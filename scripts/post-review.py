#!/usr/bin/env python3
"""
post-review.py — Post LLM suggestions as a GitHub PR review with inline comments.

Generalized version: branding and risk paths are read from config.

Comment handling follows CodeRabbit's proven pattern:
1. Summary → editable issue comment (found by HTML tag, updated in place)
2. Inline comments → posted as a COMMENT review (never carries the verdict)
3. Verdict (APPROVE / REQUEST_CHANGES / COMMENT) → posted as a separate review
   with no inline comments, so resolving threads never invalidates an approval
4. Batch fallback → if batch review 422s, post each comment individually

Supports auto-approval: safe PRs get APPROVE, risky PRs need human review,
critical findings get REQUEST_CHANGES.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import load_config

RESULT_PATH = Path("/tmp/review-result.json")
CHANGED_FILES_PATH = Path("/tmp/changed-files.txt")

SEVERITY_ICONS = {
    "critical": "\U0001f534",
    "warning": "\u26a0\ufe0f",
    "suggestion": "\U0001f4a1",
    "praise": "\u2b50",
    "consider": "\u26a0\ufe0f",  # legacy compat
}


# ---------------------------------------------------------------------------
# Config-driven branding
# ---------------------------------------------------------------------------

def _get_branding(config: dict) -> dict:
    """Get branding config with defaults."""
    branding = config.get("branding", {})
    return {
        "review_header": branding.get("review_header", "## Code Review"),
        "comment_tag": branding.get("comment_tag", "<!-- ai-review-agent -->"),
        "summary_tag": branding.get("summary_tag", "<!-- ai-review-agent-summary -->"),
    }


# ---------------------------------------------------------------------------
# Risk assessment — determines if human review is required
# ---------------------------------------------------------------------------

def has_structural_risk(changed_files: list[str], config: dict) -> tuple[bool, str | None]:
    """Changes that affect foundations."""
    risk_config = config.get("risk", {})
    structural_paths = risk_config.get("structural_paths", ["migrations/", "alembic/"])
    for risk_path in structural_paths:
        if any(risk_path in f for f in changed_files):
            return True, f"Structural risk — file matching '{risk_path}' changed"
    return False, None


def has_security_risk(changed_files: list[str], diff_content: str, config: dict) -> tuple[bool, str | None]:
    """Changes that could create security vulnerabilities."""
    risk_config = config.get("risk", {})
    security_paths = risk_config.get("security_paths", ["auth/", "security/", "middleware/"])
    if any(any(p in f for p in security_paths) for f in changed_files):
        return True, "Security-sensitive file changed"
    if any(
        (f.endswith((".env", ".env.example", ".key")) or "credentials" in f)
        for f in changed_files
    ):
        return True, "Environment/secrets file changed"
    dep_files = risk_config.get("security_dep_files", [
        "requirements.txt", "pyproject.toml", "package.json",
        "package-lock.json", "pnpm-lock.yaml",
    ])
    if any(any(f.endswith(d) for d in dep_files) for f in changed_files):
        return True, "Dependency change — supply chain risk"
    return False, None


def has_complexity_risk(changed_files: list[str], diff_stats: dict,
                        suggestions: list[dict], config: dict) -> tuple[bool, str | None]:
    """Changes too complex for automated review alone."""
    risk_config = config.get("risk", {})
    thresholds = risk_config.get("thresholds", {})
    doc_exts = config.get("files", {}).get("doc_extensions", [".md", ".mdc", ".txt", ".rst"])

    code_files = [f for f in changed_files if not any(f.endswith(ext) for ext in doc_exts)]
    code_file_count = len(code_files)

    max_files = thresholds.get("max_code_files", 15)
    if code_file_count > max_files:
        return True, f"Large PR ({code_file_count} code files) — suggest decomposition"

    code_lines = diff_stats.get("code_lines_added", 0) + diff_stats.get("code_lines_removed", 0)
    if code_lines == 0:
        code_lines = diff_stats.get("lines_added", 0) + diff_stats.get("lines_removed", 0)
    max_lines = thresholds.get("max_code_lines", 1000)
    if code_lines > max_lines:
        return True, f"Large diff ({code_lines} code lines) — suggest decomposition"

    domain_map = risk_config.get("domain_paths", {})
    if domain_map:
        domains_touched = set()
        for f in changed_files:
            for path, domain in domain_map.items():
                if path in f:
                    domains_touched.add(domain)
        cross_cutting = thresholds.get("cross_cutting_domains", 3)
        if len(domains_touched) >= cross_cutting:
            return True, f"Cross-cutting change ({', '.join(sorted(domains_touched))}) — needs architectural review"

    infra_patterns = risk_config.get("infrastructure_patterns", [
        "docker-compose", "Dockerfile", ".github/workflows/",
    ])
    if any(any(p in f for p in infra_patterns) for f in changed_files):
        return True, "Infrastructure change — affects deployment"

    return False, None


def needs_human_review(changed_files: list[str], diff_stats: dict,
                       diff_content: str, suggestions: list[dict],
                       config: dict) -> tuple[bool, list[str]]:
    """Master decision: does this PR need a human reviewer?"""
    reasons = []
    for check_fn, args in [
        (has_structural_risk, (changed_files, config)),
        (has_security_risk, (changed_files, diff_content, config)),
        (has_complexity_risk, (changed_files, diff_stats, suggestions, config)),
    ]:
        is_risky, reason = check_fn(*args)
        if is_risky:
            reasons.append(reason)
    return len(reasons) > 0, reasons


def determine_review_event(suggestions: list[dict], changed_files: list[str],
                           diff_stats: dict, diff_content: str,
                           config: dict) -> tuple[str, list[str]]:
    """Determine whether to APPROVE, REQUEST_CHANGES, or COMMENT."""
    auto_approve = config.get("review", {}).get("auto_approve_enabled", True)

    # 1. Critical findings always block
    critical = [s for s in suggestions if s.get("severity") == "critical"]
    if critical:
        reasons = [f"Critical: {s.get('title', 'unknown')}" for s in critical]
        return "REQUEST_CHANGES", reasons

    # 2. Auto-approve disabled
    if not auto_approve:
        return "COMMENT", ["Auto-approval disabled"]

    # 3. Risk assessment
    human_needed, reasons = needs_human_review(
        changed_files, diff_stats, diff_content, suggestions, config,
    )
    if human_needed:
        return "COMMENT", reasons

    # 4. All clear
    return "APPROVE", []


# ---------------------------------------------------------------------------
# Diff parsing
# ---------------------------------------------------------------------------

def get_diff_line_sets(diff_text: str) -> dict[str, set[int]]:
    """Parse diff to find which absolute line numbers are commentable."""
    line_sets: dict[str, set[int]] = {}
    current_file: str | None = None
    current_new_line = 0

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            current_file = None
        elif line.startswith("+++ b/"):
            current_file = line[6:]
            line_sets[current_file] = set()
        elif line.startswith("@@ "):
            match = re.search(r"\+(\d+)", line)
            if match:
                current_new_line = int(match.group(1)) - 1
        elif current_file is not None:
            if line.startswith("+") or line.startswith(" "):
                current_new_line += 1
                line_sets.setdefault(current_file, set()).add(current_new_line)
            elif line.startswith("-"):
                pass

    return line_sets


def get_changed_line_ranges(diff_text: str) -> dict[str, set[int]]:
    """Parse a diff to find which new-side line numbers were added/modified per file.

    Used for comment resolution: if a previous comment's file+line falls near
    a changed line, the comment is considered addressed.
    """
    ranges: dict[str, set[int]] = {}
    current_file: str | None = None
    current_new_line = 0

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            current_file = None
        elif line.startswith("+++ b/"):
            current_file = line[6:]
            ranges.setdefault(current_file, set())
        elif line.startswith("@@ "):
            match = re.search(r"\+(\d+)", line)
            if match:
                current_new_line = int(match.group(1)) - 1
        elif current_file is not None:
            if line.startswith("+"):
                current_new_line += 1
                ranges.setdefault(current_file, set()).add(current_new_line)
            elif line.startswith(" "):
                current_new_line += 1
            elif line.startswith("-"):
                pass

    return ranges


COMMENT_PROXIMITY_THRESHOLD = 5


def is_comment_addressed(
    file_path: str,
    line: int,
    changed_ranges: dict[str, set[int]],
) -> bool:
    """Check if a previous comment was addressed by changes in the new push.

    A comment is considered addressed if lines within COMMENT_PROXIMITY_THRESHOLD
    of the comment's line were modified in the same file.
    """
    if not line or line <= 0:
        return False
    changed_lines = changed_ranges.get(file_path, set())
    if not changed_lines:
        return False
    for offset in range(COMMENT_PROXIMITY_THRESHOLD + 1):
        if (line + offset) in changed_lines or (line - offset) in changed_lines:
            return True
    return False


def find_closest_commentable_line(
    commentable_lines: set[int], target_line: int,
) -> int | None:
    """Find the closest line in the diff that we can comment on."""
    if not commentable_lines:
        return None
    if target_line in commentable_lines:
        return target_line
    for offset in range(1, 6):
        if (target_line + offset) in commentable_lines:
            return target_line + offset
        if (target_line - offset) in commentable_lines:
            return target_line - offset
    return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_suggestion_body(suggestion: dict) -> str:
    """Format a suggestion into readable markdown."""
    severity = suggestion.get("severity", "suggestion")
    if severity == "consider":
        severity = "warning"
    icon = SEVERITY_ICONS.get(severity, "\U0001f4a1")
    rule = suggestion.get("rule", "")
    title = suggestion.get("title", "Suggestion")
    body = suggestion.get("body", "")
    rule_ref = f" ({rule})" if rule else ""
    return f"{icon} **{title}**{rule_ref}\n\n{body}"


def build_summary_body(
    summary: str, suggestions: list[dict], event: str,
    event_reasons: list[str], stats: dict | None = None,
    unplaced: list[dict] | None = None,
    branding: dict | None = None,
) -> str:
    """Build the summary comment body."""
    branding = branding or {}
    review_header = branding.get("review_header", "## Code Review")

    severity_counts = {}
    for s in suggestions:
        sev = s.get("severity", "suggestion")
        if sev == "consider":
            sev = "warning"
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    parts = [review_header, ""]

    if event == "APPROVE":
        parts.append("### \u2705 Auto-Approved")
        if suggestions:
            parts.append(f"No critical issues found. {len(suggestions)} suggestion{'s' if len(suggestions) != 1 else ''} for improvement.")
        else:
            parts.append(f"No issues found. {summary}")
    elif event == "REQUEST_CHANGES":
        critical_count = severity_counts.get("critical", 0)
        parts.append("### \U0001f534 Changes Requested")
        parts.append(f"Found {critical_count} critical issue{'s' if critical_count != 1 else ''} that must be resolved before merge.")
    else:
        parts.append("### \U0001f464 Human Review Required")
        parts.append("This PR requires human review. Reasons:")
        for reason in event_reasons:
            parts.append(f"- {reason}")
        if not any(s.get("severity") == "critical" for s in suggestions):
            parts.append("\nNo critical issues found by the agent.")

    parts.append("")
    parts.append(f"*{summary}*")
    parts.append("")

    parts.append("| Severity | Count |")
    parts.append("|----------|-------|")
    for sev, icon in [("critical", "\U0001f534"), ("warning", "\u26a0\ufe0f"), ("suggestion", "\U0001f4a1"), ("praise", "\u2b50")]:
        count = severity_counts.get(sev, 0)
        parts.append(f"| {icon} {sev.capitalize()} | {count} |")
    parts.append("")

    if stats and not stats.get("dry_run"):
        parts.append(f"*Reviewed in {stats.get('duration_ms', '?')}ms, {stats.get('tool_calls', '?')} tool calls*")
        parts.append("")

    if unplaced:
        parts.append("---")
        parts.append("")
        parts.append("### Findings not placed inline")
        parts.append("")
        for s in unplaced:
            formatted = format_suggestion_body(s)
            file_path = s.get("file", "")
            line = s.get("line", 0)
            parts.append(f"**{file_path}:{line}**\n\n{formatted}\n\n---\n")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# GitHub API helpers (CodeRabbit pattern)
# ---------------------------------------------------------------------------

def _gh_api(args: list[str], timeout: int = 15) -> tuple[int, str, str]:
    """Run gh api command."""
    try:
        result = subprocess.run(
            ["gh", "api"] + args,
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except subprocess.SubprocessError as e:
        return -1, "", str(e)


MAX_CLEANUP_OPS = 20


def _minimize_old_reviews(repo: str, pr_number: str, branding: dict):
    """Minimize (collapse) old agent reviews in the PR timeline."""
    review_header = branding.get("review_header", "## Code Review")
    try:
        rc, stdout, _ = _gh_api([
            f"repos/{repo}/pulls/{pr_number}/reviews",
            "--jq", f'[.[] | select(.body | contains("{review_header}")) | .node_id]',
        ])
        if rc != 0 or not stdout.strip():
            return

        node_ids = json.loads(stdout) if stdout.strip() else []
        minimized = 0
        for node_id in node_ids:
            if minimized >= MAX_CLEANUP_OPS:
                break
            mrc, _, _ = _gh_api([
                "graphql",
                "-f", f'query=mutation {{ minimizeComment(input: {{subjectId: "{node_id}", classifier: OUTDATED}}) {{ minimizedComment {{ isMinimized }} }} }}',
            ])
            if mrc == 0:
                minimized += 1
        if minimized:
            print(f"  Minimized {minimized} old review(s)")
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Warning: Review minimization failed: {e}")


def _find_summary_comment(repo: str, pr_number: str, branding: dict) -> str | None:
    """Find existing summary comment by our hidden HTML marker."""
    summary_tag = branding.get("summary_tag", "<!-- ai-review-agent-summary -->")
    rc, stdout, _ = _gh_api([
        f"repos/{repo}/issues/{pr_number}/comments",
        "--jq", f'[.[] | select(.body | contains("{summary_tag}")) | .id][0] // empty',
    ])
    comment_id = stdout.strip() if rc == 0 else ""
    return comment_id if comment_id else None


def _upsert_summary_comment(repo: str, pr_number: str, body: str, branding: dict):
    """Create or update the single summary comment."""
    summary_tag = branding.get("summary_tag", "<!-- ai-review-agent-summary -->")
    marked_body = f"{summary_tag}\n{body}"
    existing_id = _find_summary_comment(repo, pr_number, branding)

    payload_path = Path("/tmp/comment-payload.json")
    payload_path.write_text(json.dumps({"body": marked_body}, ensure_ascii=False))

    if existing_id:
        rc, _, stderr = _gh_api([
            f"repos/{repo}/issues/comments/{existing_id}",
            "--input", str(payload_path), "--method", "PATCH",
        ])
        if rc == 0:
            print(f"  Updated summary comment ({existing_id})")
        else:
            print(f"  Warning: Failed to update summary: {stderr[:100]}")
    else:
        rc, _, stderr = _gh_api([
            f"repos/{repo}/issues/{pr_number}/comments",
            "--input", str(payload_path), "--method", "POST",
        ])
        if rc == 0:
            print("  Created summary comment")
        else:
            print(f"  Warning: Failed to create summary: {stderr[:100]}")


def _list_existing_review_comments(repo: str, pr_number: str, branding: dict) -> list[dict]:
    """List all review comments from our bot on this PR."""
    comment_tag = branding.get("comment_tag", "<!-- ai-review-agent -->")
    rc, stdout, _ = _gh_api([
        f"repos/{repo}/pulls/{pr_number}/comments",
        "--jq", f'[.[] | select(.body | contains("{comment_tag}")) | {{id: .id, path: .path, line: .line, position: .position}}]',
    ])
    if rc == 0 and stdout.strip():
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return []
    return []


def _delete_review_comment(repo: str, comment_id: int):
    """Delete a single review comment."""
    _gh_api([f"repos/{repo}/pulls/comments/{comment_id}", "--method", "DELETE"])


def _delete_pending_review(repo: str, pr_number: str):
    """Delete any leftover pending review from our bot."""
    rc, stdout, _ = _gh_api([
        f"repos/{repo}/pulls/{pr_number}/reviews",
        "--jq", '[.[] | select(.state == "PENDING") | .id][0] // empty',
    ])
    review_id = stdout.strip() if rc == 0 else ""
    if review_id:
        _gh_api([
            f"repos/{repo}/pulls/{pr_number}/reviews/{review_id}",
            "--method", "DELETE",
        ])
        print(f"  Deleted pending review ({review_id})")


def _get_head_commit(repo: str, pr_number: str) -> str | None:
    """Get the HEAD commit SHA for a PR."""
    rc, stdout, _ = _gh_api([
        f"repos/{repo}/pulls/{pr_number}",
        "--jq", ".head.sha",
    ])
    sha = stdout.strip() if rc == 0 else ""
    return sha if sha else None


def _post_individual_comment(
    repo: str, pr_number: str, commit_id: str, comment: dict,
) -> bool:
    """Post a single review comment on a PR."""
    payload: dict = {
        "commit_id": commit_id,
        "path": comment["path"],
        "body": comment["body"],
        "line": comment["line"],
        "side": "RIGHT",
    }
    if comment.get("start_line") and comment["start_line"] != comment["line"]:
        payload["start_line"] = comment["start_line"]
        payload["start_side"] = "RIGHT"

    payload_path = Path("/tmp/individual-comment.json")
    payload_path.write_text(json.dumps(payload, ensure_ascii=False))
    rc, _, stderr = _gh_api([
        f"repos/{repo}/pulls/{pr_number}/comments",
        "--input", str(payload_path), "--method", "POST",
    ])
    if rc != 0:
        print(f"    Skipped comment on {comment['path']}:{comment['line']}: {stderr[:80]}")
    return rc == 0


# ---------------------------------------------------------------------------
# Review posting
# ---------------------------------------------------------------------------

def _post_inline_comment_review(
    repo: str, pr_number: str, head_sha: str | None,
    api_comments: list[dict], review_header: str,
):
    """Post inline comments as a COMMENT review (never APPROVE).

    Keeping comments separate from the verdict means resolving threads
    or minimizing old reviews won't invalidate an approval.
    """
    review_payload: dict = {
        "body": f"{review_header}\nInline comments from automated review.",
        "event": "COMMENT",
        "comments": [
            {"path": c["path"], "line": c["line"], "side": c["side"], "body": c["body"]}
            for c in api_comments
        ],
    }
    if head_sha:
        review_payload["commit_id"] = head_sha

    payload_path = Path("/tmp/review-comments-payload.json")
    payload_path.write_text(json.dumps(review_payload, ensure_ascii=False))

    rc, _, stderr = _gh_api([
        f"repos/{repo}/pulls/{pr_number}/reviews",
        "--input", str(payload_path), "--method", "POST",
    ], timeout=30)

    if rc == 0:
        print(f"  Inline comment review posted ({len(api_comments)} comments)")
        return

    # Fallback: post each comment individually
    print(f"  Warning: Batch comment review failed: {stderr[:200]}. Falling back to individual comments.")
    _delete_pending_review(repo, pr_number)

    if not head_sha:
        print("  Warning: No HEAD commit SHA — cannot post individual comments")
        return

    posted = 0
    for comment in api_comments:
        if _post_individual_comment(repo, pr_number, head_sha, comment):
            posted += 1
    print(f"  Fallback: {posted}/{len(api_comments)} comments posted individually")


def _post_verdict_review(
    repo: str, pr_number: str, head_sha: str | None,
    event: str, review_header: str, inline_count: int,
):
    """Post the verdict (APPROVE / REQUEST_CHANGES / COMMENT) as a standalone review.

    This review carries no inline comments, so minimizing or resolving
    threads on the comment review won't affect the approval state.
    """
    detail = f"{inline_count} inline comment{'s' if inline_count != 1 else ''}" if inline_count else "no inline comments"
    review_payload: dict = {
        "body": f"{review_header}\nSee summary above for details. ({detail})",
        "event": event,
    }
    if head_sha:
        review_payload["commit_id"] = head_sha

    payload_path = Path("/tmp/review-verdict-payload.json")
    payload_path.write_text(json.dumps(review_payload, ensure_ascii=False))

    rc, _, stderr = _gh_api([
        f"repos/{repo}/pulls/{pr_number}/reviews",
        "--input", str(payload_path), "--method", "POST",
    ], timeout=30)

    if rc == 0:
        print(f"  Verdict review posted (event={event})")
    else:
        print(f"  Warning: Failed to post verdict review: {stderr[:200]}")


def post_review_via_gh(
    pr_number: str,
    summary: str,
    suggestions: list[dict],
    diff_line_sets: dict[str, set[int]],
    stats: dict | None = None,
    changed_files: list[str] | None = None,
    diff_stats: dict | None = None,
    diff_content: str = "",
    config: dict | None = None,
):
    """Post review using CodeRabbit's exact pattern."""
    config = config or {}
    changed_files = changed_files or []
    diff_stats = diff_stats or {}
    branding = _get_branding(config)
    comment_tag = branding["comment_tag"]
    review_header = branding["review_header"]

    event, event_reasons = determine_review_event(
        suggestions, changed_files, diff_stats, diff_content, config,
    )

    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        print("WARNING: GITHUB_REPOSITORY not set.")
        print(build_summary_body(summary, suggestions, event, event_reasons, stats, branding=branding))
        return

    # Step 1: Build inline comments
    inline_comments: list[dict] = []
    unplaced: list[dict] = []

    for s in suggestions:
        formatted = f"{comment_tag}\n{format_suggestion_body(s)}"
        file_path = s.get("file", "")
        target_line = s.get("line", 0)

        commentable_lines = diff_line_sets.get(file_path, set())
        line = find_closest_commentable_line(commentable_lines, target_line) if target_line else None

        if line and file_path:
            inline_comments.append({
                "path": file_path,
                "line": line,
                "body": formatted,
                "_file": file_path,
                "_line": target_line,
            })
        else:
            unplaced.append(s)

    # Step 2: Minimize old reviews
    _minimize_old_reviews(repo, pr_number, branding)

    # Step 3: Upsert summary
    summary_body = build_summary_body(
        summary, suggestions, event, event_reasons, stats, unplaced, branding,
    )
    _upsert_summary_comment(repo, pr_number, summary_body, branding)

    # Step 4: Delete old bot inline comments
    existing_comments = _list_existing_review_comments(repo, pr_number, branding)
    if existing_comments:
        deleted = 0
        for ec in existing_comments:
            _delete_review_comment(repo, ec["id"])
            deleted += 1
        if deleted:
            print(f"  Deleted {deleted} old inline comment(s)")

    # Step 5: Post review
    _delete_pending_review(repo, pr_number)

    head_sha = _get_head_commit(repo, pr_number)
    if not head_sha:
        print("  Warning: Could not get HEAD commit SHA")

    api_comments = []
    for c in inline_comments:
        api_comments.append({
            "path": c["path"],
            "line": c["line"],
            "side": "RIGHT",
            "body": c["body"],
        })

    # Post inline comments as a separate COMMENT review so that resolving
    # comment threads or minimizing old reviews never invalidates the approval.
    if api_comments:
        _post_inline_comment_review(repo, pr_number, head_sha, api_comments, review_header)

    # Post the approval/request-changes/comment verdict as its own review
    _post_verdict_review(repo, pr_number, head_sha, event, review_header, len(api_comments))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_changed_files() -> list[str]:
    if CHANGED_FILES_PATH.exists():
        return [f.strip() for f in CHANGED_FILES_PATH.read_text().strip().split("\n") if f.strip()]
    return []


def compute_diff_stats(diff_text: str, config: dict) -> dict:
    doc_extensions = set(config.get("files", {}).get("doc_extensions", [".md", ".mdc", ".txt", ".rst"]))
    lines_added = 0
    lines_removed = 0
    code_lines_added = 0
    code_lines_removed = 0
    files = set()
    current_file = ""
    current_is_doc = False

    for line in diff_text.split("\n"):
        if line.startswith("+++ b/"):
            current_file = line[6:]
            files.add(current_file)
            current_is_doc = any(current_file.endswith(ext) for ext in doc_extensions)
        elif line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
            if not current_is_doc:
                code_lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_removed += 1
            if not current_is_doc:
                code_lines_removed += 1

    return {
        "files_changed": len(files),
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "code_lines_added": code_lines_added,
        "code_lines_removed": code_lines_removed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not RESULT_PATH.exists():
        print("ERROR: No review result found. Run llm-review.py first.")
        sys.exit(1)

    config = load_config()
    branding = _get_branding(config)

    result = json.loads(RESULT_PATH.read_text())

    if result.get("skip"):
        print("Review was skipped. Nothing to post.")
        return

    summary = result.get("summary", "Review complete.")
    suggestions = result.get("suggestions", [])
    stats = result.get("stats", {})
    is_dry_run = result.get("dry_run", False) or os.environ.get("REVIEW_AGENT_DRY_RUN", "false").lower() == "true"

    changed_files = load_changed_files()

    diff_path = Path("/tmp/pr.diff")
    if diff_path.exists():
        diff_text = diff_path.read_text(errors="replace")
    else:
        try:
            r = subprocess.run(
                ["git", "diff", "origin/main...HEAD"],
                capture_output=True, text=True, timeout=30,
            )
            diff_text = r.stdout
        except Exception:
            diff_text = ""

    diff_line_sets = get_diff_line_sets(diff_text)
    diff_stats = compute_diff_stats(diff_text, config)

    event, event_reasons = determine_review_event(
        suggestions, changed_files, diff_stats, diff_content=diff_text, config=config,
    )

    print("=== Posting Review ===")
    print(f"Summary: {summary}")
    print(f"Suggestions: {len(suggestions)}")
    print(f"  Critical: {sum(1 for s in suggestions if s.get('severity') == 'critical')}")
    print(f"  Warning: {sum(1 for s in suggestions if s.get('severity') in ('warning', 'consider'))}")
    print(f"  Suggestion: {sum(1 for s in suggestions if s.get('severity') == 'suggestion')}")
    print(f"  Praise: {sum(1 for s in suggestions if s.get('severity') == 'praise')}")
    print(f"Review event: {event}")
    if event_reasons:
        print(f"  Reasons: {event_reasons}")
    print(f"Auto-approve enabled: {config.get('review', {}).get('auto_approve_enabled', True)}")
    print(f"Diff stats: {diff_stats}")
    print(f"Dry run: {is_dry_run}")

    pr_number = os.environ.get("PR_NUMBER", "")
    if not pr_number:
        print("WARNING: PR_NUMBER not set.")

    if is_dry_run:
        print(f"[DRY RUN] Would post review with event: {event}")
        summary_file = os.environ.get("GITHUB_STEP_SUMMARY", "")
        if summary_file:
            with open(summary_file, "a") as f:
                f.write("\n## Code Review (Dry Run)\n\n")
                f.write(f"- Would post as: {event}\n")
                f.write(f"- Suggestions: {len(suggestions)}\n")
        return

    post_review_via_gh(
        pr_number, summary, suggestions, diff_line_sets, stats,
        changed_files=changed_files, diff_stats=diff_stats,
        diff_content=diff_text, config=config,
    )
    print(f"=== Review Posting Complete (event: {event}) ===")


if __name__ == "__main__":
    main()
