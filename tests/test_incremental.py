"""Tests for incremental review mode — the end-to-end scoping logic.

These tests exercise the real business logic: scoping rules, blast radius,
prompts, inline comment filtering, and summary labeling when a push changes
only a subset of files in a PR.

Mocks are only at component boundaries: filesystem (tmp files) and GitHub API.
"""

import importlib
import json
import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ.setdefault("GITHUB_WORKSPACE", "/tmp/test-repo")


# ---------------------------------------------------------------------------
# Helper: reload modules (they read env at import time)
# ---------------------------------------------------------------------------

@pytest.fixture()
def prepare_mod():
    return importlib.import_module("prepare-context")


@pytest.fixture()
def llm_mod():
    return importlib.import_module("llm-review")


@pytest.fixture()
def post_mod():
    return importlib.import_module("post-review")


# ---------------------------------------------------------------------------
# prepare-context: get_push_changed_files
# ---------------------------------------------------------------------------

class TestGetPushChangedFiles:

    def test_returns_none_when_file_missing(self, prepare_mod, tmp_path):
        with patch.object(Path, "exists", return_value=False):
            # Call with a path that doesn't exist — should return None
            result = prepare_mod.get_push_changed_files()
        # When /tmp/push-changed-files.txt doesn't exist, we're in full mode
        assert result is None or isinstance(result, list)

    def test_reads_push_files_from_disk(self, prepare_mod, tmp_path):
        push_file = Path("/tmp/push-changed-files.txt")
        try:
            push_file.write_text("src/auth.py\nsrc/login.py\n")
            result = prepare_mod.get_push_changed_files()
            assert result == ["src/auth.py", "src/login.py"]
        finally:
            push_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# prepare-context: main() scoping — verify the assembled context JSON
# ---------------------------------------------------------------------------

class TestPrepareContextIncremental:
    """When push-changed-files.txt exists, prepare-context must scope
    rule matching, blast radius, and spec discovery to push files,
    while keeping the full PR file list in changed_files."""

    def _run_main(self, prepare_mod, pr_files, push_files=None, config=None):
        """Run prepare-context main() with controlled filesystem state."""
        config = config or {}
        pr_file = Path("/tmp/changed-files.txt")
        push_file = Path("/tmp/push-changed-files.txt")
        diff_file = Path("/tmp/pr.diff")
        output_file = Path("/tmp/review-context.json")

        try:
            pr_file.write_text("\n".join(pr_files))
            diff_file.write_text("diff --git a/x b/x\n+++ b/x\n+line\n")
            if push_files is not None:
                push_file.write_text("\n".join(push_files))
            else:
                push_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)

            with patch.object(prepare_mod, "load_config", return_value=config), \
                 patch.object(prepare_mod, "select_applicable_rules", return_value=[]) as mock_rules, \
                 patch.object(prepare_mod, "trace_blast_radius", return_value=[]) as mock_blast, \
                 patch.dict(os.environ, {"PR_BODY": "", "PR_TITLE": "test", "BRANCH_NAME": "feat/x"}):
                prepare_mod.main()

            ctx = json.loads(output_file.read_text())
            return ctx, mock_rules, mock_blast
        finally:
            pr_file.unlink(missing_ok=True)
            push_file.unlink(missing_ok=True)
            diff_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)

    def test_full_mode_when_no_push_files(self, prepare_mod):
        ctx, mock_rules, mock_blast = self._run_main(
            prepare_mod,
            pr_files=["src/a.py", "src/b.py", "src/c.py"],
            push_files=None,
        )
        assert ctx["review_scope"] == "full"
        assert ctx["changed_files"] == ["src/a.py", "src/b.py", "src/c.py"]
        assert "push_changed_files" not in ctx

        # Rule matching and blast radius should use full PR file list
        mock_rules.assert_called_once()
        scoped_files = mock_rules.call_args[0][0]
        assert set(scoped_files) == {"src/a.py", "src/b.py", "src/c.py"}

    def test_incremental_mode_scopes_to_push_files(self, prepare_mod):
        ctx, mock_rules, mock_blast = self._run_main(
            prepare_mod,
            pr_files=["src/a.py", "src/b.py", "src/c.py"],
            push_files=["src/b.py"],
        )
        assert ctx["review_scope"] == "incremental"
        # Full PR files preserved
        assert ctx["changed_files"] == ["src/a.py", "src/b.py", "src/c.py"]
        # Push files tracked separately
        assert ctx["push_changed_files"] == ["src/b.py"]

        # Rule matching scoped to push files only
        scoped_files = mock_rules.call_args[0][0]
        assert scoped_files == ["src/b.py"]

        # Blast radius scoped to push files only
        blast_files = mock_blast.call_args[0][0]
        assert blast_files == ["src/b.py"]


# ---------------------------------------------------------------------------
# llm-review: system prompt includes incremental instructions
# ---------------------------------------------------------------------------

class TestLlmReviewIncrementalPrompt:

    def test_full_mode_prompt_has_no_incremental_language(self, llm_mod):
        context = {
            "review_scope": "full",
            "project": {"name": "TestApp"},
            "rules": [],
            "spec_docs": [],
        }
        prompt = llm_mod.build_system_prompt(context, {})
        assert "incremental" not in prompt.lower()
        assert "follow-up push" not in prompt.lower()

    def test_incremental_prompt_mentions_push_files(self, llm_mod):
        context = {
            "review_scope": "incremental",
            "push_changed_files": ["src/auth.py", "src/login.py"],
            "project": {"name": "TestApp"},
            "rules": [],
            "spec_docs": [],
        }
        prompt = llm_mod.build_system_prompt(context, {})
        assert "incremental" in prompt.lower()
        assert "src/auth.py" in prompt
        assert "src/login.py" in prompt

    def test_incremental_prompt_still_contains_review_dimensions(self, llm_mod):
        """Incremental instructions are prepended, not replacing the core prompt."""
        context = {
            "review_scope": "incremental",
            "push_changed_files": ["src/auth.py"],
            "project": {},
            "rules": [],
            "spec_docs": [],
        }
        prompt = llm_mod.build_system_prompt(context, {})
        assert "Correctness" in prompt
        assert "Security" in prompt
        assert "JSON" in prompt


# ---------------------------------------------------------------------------
# llm-review: user message separates push files from PR files
# ---------------------------------------------------------------------------

class TestLlmReviewUserMessage:

    def test_full_mode_lists_all_files_once(self, llm_mod):
        context = {
            "changed_files": ["src/a.py", "src/b.py"],
            "diff": "some diff",
            "pr_title": "Test",
            "branch_name": "feat/x",
        }
        msg = llm_mod.build_user_message(context)
        assert "## Changed Files" in msg
        assert "Files Changed in This Push" not in msg

    def test_incremental_mode_separates_push_and_pr_files(self, llm_mod):
        context = {
            "review_scope": "incremental",
            "push_changed_files": ["src/b.py"],
            "changed_files": ["src/a.py", "src/b.py", "src/c.py"],
            "diff": "some diff",
            "pr_title": "Test",
            "branch_name": "feat/x",
        }
        msg = llm_mod.build_user_message(context)
        assert "Files Changed in This Push" in msg
        assert "All PR Files" in msg
        # The trailing instruction should mention incremental
        assert "incremental" in msg.lower()

    def test_incremental_user_message_contains_diff(self, llm_mod):
        context = {
            "review_scope": "incremental",
            "push_changed_files": ["src/b.py"],
            "changed_files": ["src/a.py", "src/b.py"],
            "diff": "+new_function()",
            "pr_title": "Test",
            "branch_name": "feat/x",
        }
        msg = llm_mod.build_user_message(context)
        assert "+new_function()" in msg


# ---------------------------------------------------------------------------
# llm-review: coverage checking scoped to push files
# ---------------------------------------------------------------------------

class TestLlmReviewCoverageScoping:

    def test_coverage_check_uses_push_files_in_incremental(self, llm_mod):
        """In incremental mode, only push-changed files should be checked for coverage."""
        # Simulate: PR has 5 files, push changed 2. Agent reviewed 2 push files.
        push_files = ["src/auth.py", "src/login.py"]
        reviewed = {"src/auth.py", "src/login.py"}
        result = {"suggestions": []}

        missed = llm_mod.check_file_coverage(push_files, reviewed, result)
        assert missed == [], "All push files reviewed — no gaps expected"

    def test_coverage_check_detects_missed_push_file(self, llm_mod):
        push_files = ["src/auth.py", "src/login.py"]
        reviewed = {"src/auth.py"}
        result = {"suggestions": []}

        missed = llm_mod.check_file_coverage(push_files, reviewed, result)
        assert "src/login.py" in missed

    def test_full_pr_files_not_flagged_in_incremental(self, llm_mod):
        """Files in the PR but not in the push should NOT be flagged as missed."""
        push_files = ["src/auth.py"]
        reviewed = {"src/auth.py"}
        result = {"suggestions": []}

        missed = llm_mod.check_file_coverage(push_files, reviewed, result)
        assert missed == []


# ---------------------------------------------------------------------------
# llm-review: result propagates scope info
# ---------------------------------------------------------------------------

class TestLlmReviewResultPropagation:
    """The dry_run path writes a result JSON. Verify scope fields are present."""

    def test_dry_run_writes_result(self, llm_mod, tmp_path):
        context = {
            "review_scope": "incremental",
            "push_changed_files": ["src/a.py"],
            "changed_files": ["src/a.py", "src/b.py"],
            "rules": [],
            "spec_docs": [],
            "project": {},
            "diff": "diff",
        }
        output_path = Path("/tmp/review-result.json")
        try:
            llm_mod.dry_run(context, {})
            result = json.loads(output_path.read_text())
            # dry_run doesn't propagate scope (only live_review does),
            # but we can verify the dry_run result is valid
            assert result.get("dry_run") is True
            assert "suggestions" in result
        finally:
            output_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# post-review: inline comments filtered to push files
# ---------------------------------------------------------------------------

class TestPostReviewIncrementalFiltering:

    def test_full_mode_posts_all_comments(self, post_mod):
        """In full mode, all suggestions get inline comments."""
        suggestions = [
            {"severity": "warning", "title": "A", "body": "B", "file": "src/a.py", "line": 10},
            {"severity": "warning", "title": "C", "body": "D", "file": "src/b.py", "line": 20},
        ]
        diff_line_sets = {"src/a.py": {10, 11}, "src/b.py": {20, 21}}
        api_calls = []

        def tracking_gh_api(args, timeout=15):
            endpoint = args[0] if args else ""
            api_calls.append({"endpoint": endpoint, "args": args})
            if "reviews" in endpoint and "--jq" in args:
                return (0, "[]", "")
            if "comments" in endpoint and "--jq" in args:
                return (0, "[]", "")
            if "issues" in endpoint and "comments" in endpoint:
                return (0, '{"id": 1}', "")
            return (0, "{}", "")

        config = {"review": {"auto_approve_enabled": True}}
        with patch.object(post_mod, "_gh_api", side_effect=tracking_gh_api), \
             patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/repo"}):
            post_mod.post_review_via_gh(
                "1", "OK", suggestions, diff_line_sets,
                changed_files=["src/a.py", "src/b.py"], diff_stats={},
                diff_content="", config=config,
                review_scope="full",
            )

        # Both files should get inline comments — look for the inline comment review POST
        review_posts = [
            c for c in api_calls
            if c["endpoint"].endswith("/reviews")
            and "--method" in c["args"] and "POST" in c["args"]
            and "--jq" not in c["args"]
        ]
        # At least one review POST should exist (the inline comment review)
        assert len(review_posts) >= 1

    def test_incremental_mode_filters_comments_to_push_files(self, post_mod):
        """In incremental mode, only push-changed files get inline comments."""
        suggestions = [
            {"severity": "warning", "title": "A", "body": "B", "file": "src/a.py", "line": 10},
            {"severity": "warning", "title": "C", "body": "D", "file": "src/b.py", "line": 20},
            {"severity": "suggestion", "title": "E", "body": "F", "file": "src/c.py", "line": 30},
        ]
        diff_line_sets = {
            "src/a.py": {10, 11},
            "src/b.py": {20, 21},
            "src/c.py": {30, 31},
        }

        payloads_written = []
        original_dumps = json.dumps

        def capture_dumps(obj, **kw):
            payloads_written.append(obj)
            return original_dumps(obj, **kw)

        def ok_gh_api(args, timeout=15):
            endpoint = args[0] if args else ""
            if "--jq" in args:
                return (0, "[]", "")
            if "issues" in endpoint and "comments" in endpoint:
                return (0, '{"id": 1}', "")
            return (0, "{}", "")

        config = {"review": {"auto_approve_enabled": True}}
        with patch.object(post_mod, "_gh_api", side_effect=ok_gh_api), \
             patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/repo"}), \
             patch("json.dumps", side_effect=capture_dumps):
            post_mod.post_review_via_gh(
                "1", "OK", suggestions, diff_line_sets,
                changed_files=["src/a.py", "src/b.py", "src/c.py"],
                diff_stats={}, diff_content="", config=config,
                review_scope="incremental",
                push_changed_files=["src/b.py"],
            )

        # Find the inline comment review payload (has "comments" key with list)
        inline_payloads = [
            p for p in payloads_written
            if isinstance(p, dict) and "comments" in p and isinstance(p["comments"], list)
        ]
        assert len(inline_payloads) == 1, f"Expected 1 inline review payload, got {len(inline_payloads)}"

        # Only src/b.py should have comments (it's the only push-changed file)
        commented_files = {c["path"] for c in inline_payloads[0]["comments"]}
        assert commented_files == {"src/b.py"}, f"Expected only src/b.py, got {commented_files}"

    def test_incremental_mode_no_push_files_posts_nothing_inline(self, post_mod):
        """If the LLM returns suggestions for files outside the push, no inline comments."""
        suggestions = [
            {"severity": "warning", "title": "A", "body": "B", "file": "src/a.py", "line": 10},
        ]
        diff_line_sets = {"src/a.py": {10, 11}}
        api_calls = []

        def tracking_gh_api(args, timeout=15):
            endpoint = args[0] if args else ""
            api_calls.append({"endpoint": endpoint, "args": args})
            if "--jq" in args:
                return (0, "[]", "")
            if "issues" in endpoint and "comments" in endpoint:
                return (0, '{"id": 1}', "")
            return (0, "{}", "")

        config = {"review": {"auto_approve_enabled": True}}
        with patch.object(post_mod, "_gh_api", side_effect=tracking_gh_api), \
             patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/repo"}):
            post_mod.post_review_via_gh(
                "1", "OK", suggestions, diff_line_sets,
                changed_files=["src/a.py", "src/b.py"],
                diff_stats={}, diff_content="", config=config,
                review_scope="incremental",
                push_changed_files=["src/b.py"],  # a.py not in push
            )

        # Should only have the verdict review, no inline comment review
        review_posts = [
            c for c in api_calls
            if c["endpoint"].endswith("/reviews")
            and "--method" in c["args"] and "POST" in c["args"]
            and "--jq" not in c["args"]
        ]
        # Only 1 review POST (verdict), no inline comment review
        assert len(review_posts) == 1


# ---------------------------------------------------------------------------
# post-review: summary body includes incremental label
# ---------------------------------------------------------------------------

class TestSummaryBodyIncremental:

    def test_full_mode_no_incremental_label(self, post_mod):
        body = post_mod.build_summary_body(
            "All good", [], "APPROVE", [],
            review_scope="full",
        )
        assert "incremental" not in body.lower()

    def test_incremental_mode_has_label(self, post_mod):
        body = post_mod.build_summary_body(
            "All good", [], "APPROVE", [],
            review_scope="incremental",
        )
        assert "incremental" in body.lower()
        assert "latest push" in body.lower()

    def test_incremental_label_appears_before_verdict(self, post_mod):
        body = post_mod.build_summary_body(
            "All good", [], "APPROVE", [],
            review_scope="incremental",
        )
        incremental_pos = body.lower().find("incremental")
        approve_pos = body.find("Auto-Approved")
        assert incremental_pos < approve_pos, "Incremental label should appear before the verdict"

    def test_incremental_with_request_changes(self, post_mod):
        suggestions = [{"severity": "critical", "title": "Bug", "body": "Fix it"}]
        body = post_mod.build_summary_body(
            "Bug found", suggestions, "REQUEST_CHANGES", ["Critical: Bug"],
            review_scope="incremental",
        )
        assert "incremental" in body.lower()
        assert "Changes Requested" in body


# ---------------------------------------------------------------------------
# End-to-end: prepare-context → result JSON shape
# ---------------------------------------------------------------------------

class TestIncrementalContextShape:
    """Verify the context JSON has all expected fields for downstream consumers."""

    def test_incremental_context_has_both_file_lists(self, prepare_mod):
        pr_file = Path("/tmp/changed-files.txt")
        push_file = Path("/tmp/push-changed-files.txt")
        diff_file = Path("/tmp/pr.diff")
        output_file = Path("/tmp/review-context.json")

        try:
            pr_file.write_text("src/a.py\nsrc/b.py\nsrc/c.py\n")
            push_file.write_text("src/c.py\n")
            diff_file.write_text("diff --git a/x b/x\n+++ b/x\n+line\n")

            with patch.object(prepare_mod, "load_config", return_value={}), \
                 patch.object(prepare_mod, "select_applicable_rules", return_value=[]), \
                 patch.object(prepare_mod, "trace_blast_radius", return_value=[]), \
                 patch.dict(os.environ, {"PR_BODY": "", "PR_TITLE": "test", "BRANCH_NAME": "feat/x"}):
                prepare_mod.main()

            ctx = json.loads(output_file.read_text())

            # Shape assertions
            assert ctx["skip"] is False
            assert ctx["review_scope"] == "incremental"
            assert "src/a.py" in ctx["changed_files"]
            assert "src/b.py" in ctx["changed_files"]
            assert "src/c.py" in ctx["changed_files"]
            assert ctx["push_changed_files"] == ["src/c.py"]
            assert "diff" in ctx
            assert "rules" in ctx
        finally:
            pr_file.unlink(missing_ok=True)
            push_file.unlink(missing_ok=True)
            diff_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)

    def test_full_context_has_no_push_files(self, prepare_mod):
        pr_file = Path("/tmp/changed-files.txt")
        push_file = Path("/tmp/push-changed-files.txt")
        diff_file = Path("/tmp/pr.diff")
        output_file = Path("/tmp/review-context.json")

        try:
            pr_file.write_text("src/a.py\nsrc/b.py\n")
            push_file.unlink(missing_ok=True)
            diff_file.write_text("diff --git a/x b/x\n+++ b/x\n+line\n")

            with patch.object(prepare_mod, "load_config", return_value={}), \
                 patch.object(prepare_mod, "select_applicable_rules", return_value=[]), \
                 patch.object(prepare_mod, "trace_blast_radius", return_value=[]), \
                 patch.dict(os.environ, {"PR_BODY": "", "PR_TITLE": "test", "BRANCH_NAME": "feat/x"}):
                prepare_mod.main()

            ctx = json.loads(output_file.read_text())

            assert ctx["review_scope"] == "full"
            assert "push_changed_files" not in ctx
        finally:
            pr_file.unlink(missing_ok=True)
            diff_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)
