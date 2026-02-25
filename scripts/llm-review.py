#!/usr/bin/env python3
"""
llm-review.py — Agentic PR reviewer using Claude with tool use.

Generalized version: the system prompt is built from the project config
provided by the consuming repo. The agent reads the diff, decides what
additional context it needs, and uses tools to investigate:
  - read_file: Read a specific file or section from the repo
  - search_code: Grep the codebase for a pattern
  - read_rule: Read a specific rule by ID
  - list_directory: List files in a directory

Supports dry-run mode when ANTHROPIC_API_KEY is not set.
"""

import itertools
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import load_config, get_repo_root

CONTEXT_PATH = Path("/tmp/review-context.json")
OUTPUT_PATH = Path("/tmp/review-result.json")
REPO_ROOT = get_repo_root()
SCRIPT_DIR = Path(__file__).resolve().parent
VERIFICATION_RULES_PATH = SCRIPT_DIR.parent / "defaults" / "verification-rules.md"

# Defaults — overridable via config
MODEL = os.environ.get("REVIEW_AGENT_MODEL", "claude-sonnet-4-20250514")
MAX_TOKENS = int(os.environ.get("REVIEW_AGENT_MAX_TOKENS", "8192"))
MAX_TOOL_ROUNDS = 10
MAX_TOOL_ROUNDS_CEILING = 30
DOC_EXTENSIONS = {".md", ".mdc", ".txt", ".rst", ".mdx"}
SKIP_EXTENSIONS = {".lock", ".yaml", ".yml", ".json", ".toml"}
LOCKFILE_NAMES = {"pnpm-lock.yaml", "package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock"}


# ---------------------------------------------------------------------------
# Tool definitions (what Claude can call)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file from the repository. You can specify line ranges to read just the relevant section. Use this to check related files, test files, callers, or any code you need context on.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to repo root",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start line (1-indexed). Omit to read from beginning.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line (1-indexed). Omit to read to end. Use with start_line to read a specific section.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_code",
        "description": "Search the codebase for a pattern using grep. Returns matching files and lines. Use this to find callers of a function, usages of a class, or check if a pattern exists elsewhere.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (basic grep regex).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File glob to restrict search. E.g., '*.py', '*.tsx'. Omit to search all files.",
                },
                "directory": {
                    "type": "string",
                    "description": "Directory to search in, relative to repo root. Omit to search entire repo.",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_rule",
        "description": "Read the full content of a specific project rule by its ID. Use this when you see a potential violation and want to check the exact rule before making a suggestion.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rule_id": {
                    "type": "string",
                    "description": "Rule ID (filename without extension)",
                },
            },
            "required": ["rule_id"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files in a directory. Use this to check if test files exist, see the structure of a package, or find related files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to repo root.",
                },
            },
            "required": ["path"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _validate_path(user_path: str) -> Path | None:
    """Validate that a user-supplied path stays within REPO_ROOT."""
    try:
        resolved = (REPO_ROOT / user_path).resolve()
        if not str(resolved).startswith(str(REPO_ROOT.resolve())):
            return None
        return resolved
    except (ValueError, OSError):
        return None


def execute_tool(name: str, input_data: dict, config: dict) -> str:
    """Execute a tool and return the result as a string."""
    try:
        if name == "read_file":
            return tool_read_file(input_data)
        elif name == "search_code":
            return tool_search_code(input_data)
        elif name == "read_rule":
            return tool_read_rule(input_data, config)
        elif name == "list_directory":
            return tool_list_directory(input_data)
        else:
            return f"Unknown tool: {name}"
    except PermissionError:
        return f"Tool error: permission denied for {input_data}"
    except FileNotFoundError:
        return "Tool error: file not found"
    except subprocess.TimeoutExpired:
        return "Tool error: operation timed out"
    except Exception as e:
        return f"Tool error ({type(e).__name__}): {e}"


def tool_read_file(input_data: dict) -> str:
    filepath = _validate_path(input_data["path"])
    if filepath is None:
        return "Access denied: path is outside the repository"
    if not filepath.exists():
        return f"File not found: {input_data['path']}"
    if not filepath.is_file():
        return f"Not a file: {input_data['path']}"

    start = max(0, input_data.get("start_line", 1) - 1)
    end = input_data.get("end_line")

    with open(filepath, encoding="utf-8", errors="replace") as f:
        max_lines = 200
        if end is not None:
            max_lines = min(end - start, max_lines)
        selected = list(itertools.islice(f, start, start + max_lines))

    selected = [line.rstrip("\n") for line in selected]

    total_lines_hint = start + len(selected)
    if len(selected) >= 200:
        selected.append(f"... [truncated at 200 lines — file continues beyond line {total_lines_hint}]")

    numbered = [f"{i + start + 1:4d} | {line}" for i, line in enumerate(selected)]
    return "\n".join(numbered)


def tool_search_code(input_data: dict) -> str:
    pattern = input_data["pattern"]

    search_dir = _validate_path(input_data.get("directory", ""))
    if search_dir is None:
        search_dir = REPO_ROOT

    file_pattern = input_data.get("file_pattern", "")

    if file_pattern and not re.match(r"^[a-zA-Z0-9_.*?/-]+$", file_pattern):
        return "Invalid file pattern: only alphanumeric, _, ., *, ?, /, - allowed."

    if re.escape(pattern) != pattern:
        grep_flag = "-rnE"
    else:
        grep_flag = "-rnF"

    cmd = ["grep", grep_flag, pattern, str(search_dir)]
    if file_pattern:
        cmd.insert(2, f"--include={file_pattern}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = result.stdout.strip()
        if not output:
            return "No matches found."

        lines = output.split("\n")
        if len(lines) > 30:
            total_count = len(lines)
            lines = lines[:30]
            lines.append(f"... [{total_count} total matches, showing first 30]")

        return "\n".join(
            line.replace(str(REPO_ROOT.resolve()) + "/", "").replace(str(REPO_ROOT) + "/", "")
            for line in lines
        )
    except subprocess.TimeoutExpired:
        return "Search timed out."
    except subprocess.SubprocessError as e:
        return f"Search failed: {type(e).__name__}"


def tool_read_rule(input_data: dict, config: dict) -> str:
    rule_id = input_data["rule_id"]
    if "/" in rule_id or "\\" in rule_id or ".." in rule_id:
        return "Access denied: invalid rule ID"

    rules_dir = config.get("rules", {}).get("directory", ".cursor/rules")
    file_pattern = config.get("rules", {}).get("file_pattern", "*.mdc")
    extension = file_pattern.replace("*", "")  # e.g., "*.mdc" -> ".mdc"

    rule_path = REPO_ROOT / rules_dir / f"{rule_id}{extension}"
    if not rule_path.exists():
        # Try without assumed extension (maybe the ID already includes it)
        rule_path = REPO_ROOT / rules_dir / rule_id
        if not rule_path.exists():
            return f"Rule not found: {rule_id}"

    content = rule_path.read_text(encoding="utf-8")
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2].strip()

    if len(content) > 3000:
        content = content[:3000] + "\n... [truncated]"
    return content


def tool_list_directory(input_data: dict) -> str:
    dirpath = _validate_path(input_data["path"])
    if dirpath is None:
        return "Access denied: path is outside the repository"
    if not dirpath.exists():
        return f"Directory not found: {input_data['path']}"
    if not dirpath.is_dir():
        return f"Not a directory: {input_data['path']}"

    entries = sorted(dirpath.iterdir())
    lines = []
    for entry in entries[:50]:
        prefix = "d " if entry.is_dir() else "  "
        lines.append(f"{prefix}{entry.name}")

    if len(entries) > 50:
        lines.append(f"... [{len(entries)} total entries, showing first 50]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt — built from project config
# ---------------------------------------------------------------------------

def build_system_prompt(context: dict, config: dict) -> str:
    """Build the system prompt from project config + rule summaries."""

    project = context.get("project", {})
    project_name = project.get("name", "")
    project_desc = project.get("description", "")
    tech_stack = project.get("tech_stack", "")

    # Build project identity line
    if project_name and project_desc:
        identity = f"a PR for the {project_name} project — {project_desc}"
    elif project_name:
        identity = f"a PR for the {project_name} project"
    else:
        identity = "a PR"

    if tech_stack:
        identity += f" (tech stack: {tech_stack})"

    # Build rule summary
    rule_summaries = []
    for rule in context.get("rules", []):
        desc = rule.get("description", "")
        rule_summaries.append(f"- **{rule['name']}**: {desc}")
    rule_list = "\n".join(rule_summaries) if rule_summaries else "No project-specific rules configured."

    # Build spec reference
    spec_refs = []
    for doc in context.get("spec_docs", []):
        spec_refs.append(f"- {doc['path']}")
    spec_list = "\n".join(spec_refs) if spec_refs else "No spec/plan linked in the PR description."

    today = datetime.now(timezone.utc).strftime("%B %d, %Y")

    return f"""You are a senior software engineer reviewing {identity}.

Today's date is {today}. Use this when evaluating dates in code or documentation — do NOT flag dates as "future" if they are on or before today.

You review like a tech lead who cares deeply about code quality. You're helpful, specific, and have a good sense of humor. You are NOT a cop — you're the senior colleague everyone wants reviewing their code because you make it better.

## Review Dimensions (in priority order)

### 1. Correctness — Does it actually work?
- Logic errors: off-by-one, inverted conditions, wrong comparisons
- Missing null/None checks on values that could be absent
- Async issues: missing `await`, unhandled promises, race conditions
- Edge cases the tests might miss: empty lists, None, zero-length strings, Unicode

### 2. Security — Can this be exploited?
- Auth: new endpoints missing authentication or checking wrong user
- Data exposure: API responses leaking internal fields (passwords, tokens, internal IDs)
- Injection: raw string interpolation in SQL or shell commands
- SSRF: HTTP calls to user-supplied URLs without validation

### 3. Performance — Will this be slow?
- N+1 queries: DB call inside a loop instead of batch fetch
- Unbounded data: fetching all records without limit/pagination
- Expensive operations in hot paths: LLM calls inside loops, heavy computation per request
- Missing indexes for new query patterns

### 4. Backward Compatibility — Will this break existing consumers?
- API changes: renamed/removed fields that break clients
- State schema changes: new fields that old data doesn't have
- Import changes: moved modules that other files import
- DB changes without migration

### 5. Completeness — What's missing?
- Error handling: happy path works, but what if the external API is down?
- Logging: critical operations with no log entry for debugging
- Tests: new behavior with no test, or existing tests not updated
- Docs: API changes not reflected in documentation

### 6. Design & Patterns — Is this the right approach?
- Check what patterns exist nearby: if neighboring modules use a dispatch map, a new if/elif chain should too
- Search for duplicate logic: does a similar function already exist elsewhere?
- Coupling: does this new code import from too many unrelated modules?
- Naming: do variable/function names communicate intent clearly?
- Then check against project rules (see rule list below)

### 7. Architecture — Does this change fit the system?
- Is this code in the right module/layer?
- Does this new code have a single, clear responsibility?
- Search for similar logic elsewhere. If 3+ places do the same thing, suggest extraction.
- Count imports in new files. If a file imports from 5+ different modules across layers, it may be doing too much.

## How to Investigate

1. READ the PR diff carefully — understand what changed and why.
2. Review ALL changed files. Do not submit your final JSON until you have examined every changed file's diff. Use as many tool calls as you need.
3. USE TOOLS to get context:
   - `read_file` — check related files (callers, tests, models, the full file around a change)
   - `search_code` — find callers of changed functions, check for duplicates, verify patterns
   - `read_rule` — read the full rule before citing it in a suggestion
   - `list_directory` — check if tests exist, see module structure
4. Before suggesting a pattern change, SEARCH for how the codebase already handles it. Follow existing patterns.
5. ONLY suggest issues you've verified with context. Never guess or assume.
6. When you have examined every changed file and have enough context, produce your final review.

## Your Tone and Personality
- You're the senior colleague everyone WANTS reviewing their code — because you make it fun AND better.
- Coach, not cop. Suggestions, not demands.
- Explain the WHY with personality.
- Use humor naturally. Programming puns, movie references, gentle roasts — whatever fits the moment.
- Celebrate good code! Positive feedback matters.
- If the code is solid: "Ship it!" with genuine enthusiasm.
- Vary your tone — don't be the same joke every time.

## Limits
- **"critical" and "warning" have NO limit** — always report bugs, security issues, data loss risks, and breaking changes.
- **"suggestion" is capped at 8** — pick the highest-impact improvements.
- **"praise" is capped at 2** — only for genuinely impressive work (see criteria below).
- If no critical/warning issues and fewer than 2 suggestions, just say "Looks good."
- Don't nitpick formatting or style — linters handle that.
- Don't flag things that existing CI already catches.

## Project Rules (summaries — use read_rule tool for full content)
{rule_list}

## Linked Spec/Plan Documents (use read_file to read relevant sections)
{spec_list}

## Output Format
Return ONLY a JSON object (no text before or after, no markdown fences):
{{
  "summary": "One sentence overall assessment",
  "suggestions": [
    {{
      "file": "path/to/file.py",
      "line": 42,
      "severity": "warning",
      "rule": "B-28",
      "title": "Short title",
      "body": "Detailed explanation with context and a concrete fix suggestion."
    }}
  ]
}}

## Severity Classification Rules

### CRITICAL — Must fix before merge. This will BLOCK the PR.

A finding is CRITICAL only if it meets ONE of these criteria:

**Security:** Hardcoded secrets, SQL/command injection, missing auth, PII leaked in logs.
**Data Integrity:** DB write that could corrupt data, missing null check that will crash, cascade delete of user data.
**Code Bugs:** Off-by-one, wrong comparison, inverted boolean, wrong variable, missing await, dead code hiding a bug.
**Logic:** Exception swallowed silently, infinite loop, business logic changed without test update.
**Contract:** API schema changed without backward compat, state field renamed without migration, import path changed without updating callers.

**CALIBRATION:** IF IN DOUBT → WARNING, not critical. Critical means "this WILL cause a bug, security breach, or data loss in production." Aim for 0-1 critical per PR.

### WARNING — Should fix, doesn't block merge.
Missing type hints, suboptimal patterns, missing error context, performance concerns, missing docstrings, potential edge cases.

### SUGGESTION — Nice to have. Must have a CONCRETE ACTION.
Naming improvements, code organization ideas, documentation gaps, minor style preferences.

### PRAISE — Reserved for genuinely impressive work. Max 2 per review.

A praise must pass the "would you mention this in a team standup?" test. If you wouldn't say "hey, check out what they did here" to a colleague, it's not worth praising.

**Praise-worthy:**
- A clever abstraction that makes a complex problem simple
- Thorough edge-case handling that shows deep domain understanding
- A well-designed API or interface that will age well
- Test coverage that catches non-obvious failure modes
- A refactoring that significantly reduces complexity

**NOT praise-worthy (do not praise these):**
- Following standard patterns (that's expected, not exceptional)
- Basic error handling or logging (that's the minimum bar)
- Clean code that's just... normal clean code
- Using existing utilities correctly
- Adding type hints or docstrings

When in doubt, skip the praise. Zero praises is fine. The summary can acknowledge solid work without inline praise comments.

Allowed severity values: "critical", "warning", "suggestion", "praise"

If code is solid: {{"summary": "Clean PR, ship it!", "suggestions": []}}"""


def build_user_message(context: dict) -> str:
    """Build the initial user message with just the diff."""
    parts = []

    parts.append("## Changed Files")
    for f in context["changed_files"]:
        parts.append(f"- {f}")
    parts.append("")

    parts.append("## PR Info")
    parts.append(f"- Title: {context.get('pr_title', 'N/A')}")
    parts.append(f"- Branch: {context.get('branch_name', 'N/A')}")
    parts.append("")

    parts.append("## PR Diff")
    parts.append("```diff")
    parts.append(context["diff"])
    parts.append("```")

    parts.append("")
    parts.append(
        "Review this diff. Examine every changed file — do not skip any. "
        "Use tools to investigate anything that needs context. "
        "When you have reviewed all files, return your final JSON review."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def dry_run(context: dict, config: dict):
    """Log what the agent would do without making API calls."""
    print("=== DRY RUN MODE (no ANTHROPIC_API_KEY set or dry-run enabled) ===")
    print(f"Model: {MODEL}")
    print(f"Mode: AGENTIC (tool use)")
    print(f"Changed files: {len(context['changed_files'])}")
    print(f"Rules available: {[r['name'] for r in context['rules']]}")
    print(f"Spec docs: {[d['path'] for d in context.get('spec_docs', [])]}")
    print(f"Tools: read_file, search_code, read_rule, list_directory")
    print(f"Max tool rounds: {MAX_TOOL_ROUNDS}")

    system = build_system_prompt(context, config)
    user = build_user_message(context)
    system_tokens = len(system) // 4
    user_tokens = len(user) // 4
    print(f"Initial prompt — system: ~{system_tokens} tokens, user: ~{user_tokens} tokens")

    mock_result = {
        "summary": "[DRY RUN] Review agent would investigate this PR using tools. Set ANTHROPIC_API_KEY to enable.",
        "suggestions": [],
        "dry_run": True,
        "stats": {
            "mode": "agentic",
            "rules_count": len(context["rules"]),
            "rules": [r["name"] for r in context["rules"]],
            "spec_docs": [d["path"] for d in context.get("spec_docs", [])],
            "tools_available": ["read_file", "search_code", "read_rule", "list_directory"],
            "estimated_initial_tokens": system_tokens + user_tokens,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(mock_result, indent=2))
    print("=== DRY RUN COMPLETE ===")


def compute_max_tool_rounds(num_changed_files: int) -> int:
    """Scale tool rounds with PR size. More files = more investigation needed.

    The last round is always reserved for the final response (tools are
    withheld), so the effective tool budget is (result - 1).
    """
    base = 10
    per_file = 2
    dynamic = base + int(num_changed_files * per_file)
    return min(max(dynamic, base), MAX_TOOL_ROUNDS_CEILING)


def check_file_coverage(
    changed_files: list[str],
    files_read: set[str],
    result: dict,
) -> list[str]:
    """Return changed files that the agent hasn't reviewed.

    A file counts as reviewed if:
    - The agent used read_file on it
    - The agent mentioned it in a suggestion
    - It's a doc/lockfile that doesn't need code review
    """
    files_in_suggestions = {s.get("file", "") for s in result.get("suggestions", [])}
    reviewed = files_read | files_in_suggestions

    missed = []
    for f in changed_files:
        if f in reviewed:
            continue
        basename = Path(f).name
        if basename in LOCKFILE_NAMES:
            continue
        ext = Path(f).suffix
        if ext in DOC_EXTENSIONS:
            continue
        missed.append(f)
    return missed


def build_coverage_followup(missed_files: list[str]) -> str:
    """Build a follow-up message asking the agent to review missed files."""
    file_list = "\n".join(f"- {f}" for f in missed_files)
    return (
        f"You haven't reviewed these changed files yet:\n{file_list}\n\n"
        "Please review them now. Read each file's changes in the diff above "
        "(or use read_file if you need more context), then return an UPDATED "
        "JSON review that includes findings for ALL files — both the ones you "
        "already reviewed and these new ones."
    )


def _extract_json(raw_text: str) -> dict | None:
    """Extract a JSON object from model output, handling markdown fences."""
    raw_text = raw_text.strip()
    try:
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        json_start = raw_text.find("{")
        json_end = raw_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            return json.loads(raw_text[json_start:json_end])
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None


def run_verification_pass(client, result: dict, context: dict) -> dict:
    """Second pass: verify findings against verification rules to filter false positives."""
    suggestions = result.get("suggestions", [])
    if not suggestions:
        return result

    if not VERIFICATION_RULES_PATH.exists():
        print("  Verification rules not found, skipping verification pass.")
        return result

    verification_rules = VERIFICATION_RULES_PATH.read_text(encoding="utf-8")

    diff_excerpt = context.get("diff", "")[:8000]

    user_content = f"""## Verification Task

Below are the review findings and the diff they were generated from.
Apply the verification rules to each finding. Drop or fix any finding
that fails verification.

## Diff (for reference)
```diff
{diff_excerpt}
```

## Findings to Verify
```json
{json.dumps(result, indent=2)}
```"""

    print("  Running verification pass...")
    start = time.monotonic()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": verification_rules,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=TOOLS,
            messages=[{"role": "user", "content": user_content}],
        )

        messages = [
            {"role": "user", "content": user_content},
        ]
        verification_tool_calls = 0

        for verify_round in range(5):
            if response.stop_reason == "end_turn":
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        verification_tool_calls += 1
                        print(f"    Verify tool: {block.name}({json.dumps(block.input)[:120]})")
                        tool_result = execute_tool(block.name, block.input, config={})
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_result[:5000],
                        })
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=[{"type": "text", "text": verification_rules, "cache_control": {"type": "ephemeral"}}],
                    tools=TOOLS,
                    messages=messages,
                )
                continue
            break

        raw_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                raw_text += block.text

        verified = _extract_json(raw_text)
        elapsed = (time.monotonic() - start) * 1000

        if verified and "suggestions" in verified:
            before = len(suggestions)
            after = len(verified.get("suggestions", []))
            print(f"  Verification pass complete in {elapsed:.0f}ms "
                  f"({verification_tool_calls} tool calls): "
                  f"{before} → {after} findings")
            return verified

        print(f"  Verification pass could not parse response, keeping original findings.")
        return result

    except Exception as e:
        print(f"  Verification pass failed ({e}), keeping original findings.")
        return result


def live_review(context: dict, config: dict):
    """Run the agentic review loop with Claude tool use."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()
    system_prompt = build_system_prompt(context, config)
    user_message = build_user_message(context)

    messages = [{"role": "user", "content": user_message}]
    changed_files = context.get("changed_files", [])
    max_rounds = compute_max_tool_rounds(len(changed_files))
    files_read: set[str] = set()
    coverage_followup_sent = False

    print(f"=== Agentic Review ({MODEL}) ===")
    print(f"  Changed files: {len(changed_files)}, max rounds: {max_rounds}")
    total_input_tokens = 0
    total_output_tokens = 0
    tool_calls = 0
    start = time.monotonic()

    is_final_round = False
    for round_num in range(max_rounds):
        is_final_round = (round_num == max_rounds - 1)
        round_label = f"  Round {round_num + 1}/{max_rounds}"
        if is_final_round:
            round_label += " (final — no tools)"
            messages.append({
                "role": "user",
                "content": (
                    "You have used all available tool rounds. "
                    "Return your final JSON review now. "
                    "Do not request any more tools — just output the JSON object."
                ),
            })
        print(f"{round_label}...")

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[] if is_final_round else TOOLS,
                messages=messages,
            )
        except Exception as e:
            print(f"ERROR: API call failed: {e}")
            OUTPUT_PATH.write_text(
                json.dumps(
                    {
                        "summary": "Review agent encountered an API error.",
                        "suggestions": [],
                        "error": str(e),
                    }
                )
            )
            return

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        if response.stop_reason == "end_turn":
            # Check file coverage before accepting the result
            raw_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    raw_text += block.text
            preliminary = _extract_json(raw_text)

            if preliminary and not coverage_followup_sent:
                missed = check_file_coverage(changed_files, files_read, preliminary)
                if missed:
                    print(f"  Coverage gap: {len(missed)} files unreviewed — sending follow-up")
                    coverage_followup_sent = True
                    followup = build_coverage_followup(missed)
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": followup})
                    continue

            print(f"  Agent done after {round_num + 1} rounds, {tool_calls} tool calls.")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if hasattr(block, "text") and block.text.strip():
                    print(f"    Thinking: {block.text.strip()[:200]}")
                if block.type == "tool_use":
                    tool_calls += 1
                    print(f"    Tool: {block.name}({json.dumps(block.input)[:150]})")
                    if block.name == "read_file" and "path" in block.input:
                        files_read.add(block.input["path"])
                    result = execute_tool(block.name, block.input, config)
                    result_preview = result[:100].replace("\n", " ")
                    print(f"    Result: {result_preview}...")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result[:5000],
                        }
                    )

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            continue

        print(f"  Unexpected stop_reason: {response.stop_reason}")
        break

    elapsed_ms = (time.monotonic() - start) * 1000
    print(f"Completed in {elapsed_ms:.0f}ms")
    print(f"Tokens — input: {total_input_tokens}, output: {total_output_tokens}")
    print(f"Tool calls: {tool_calls}")

    input_cost = total_input_tokens * 0.000003
    output_cost = total_output_tokens * 0.000015
    print(f"Estimated cost: ${input_cost + output_cost:.4f}")

    # Extract the final JSON from the last response
    raw_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            raw_text += block.text

    result = _extract_json(raw_text)
    if result is None:
        print("WARNING: Could not parse final response as JSON. Posting as raw text.")
        result = {
            "summary": "Review completed (response format issue — showing raw output).",
            "suggestions": [],
            "raw_response": raw_text.strip()[:3000],
        }

    # Verification pass: re-evaluate findings against verification rules
    suggestions = result.get("suggestions", [])
    has_critical_or_warning = any(
        s.get("severity") in ("critical", "warning") for s in suggestions
    )
    if suggestions and has_critical_or_warning:
        before_count = len(suggestions)
        result = run_verification_pass(client, result, context)
        after_count = len(result.get("suggestions", []))
        if after_count < before_count:
            result["verification"] = {
                "findings_before": before_count,
                "findings_after": after_count,
                "dropped": before_count - after_count,
            }
    else:
        print("  Skipping verification pass (no critical/warning findings).")

    result["stats"] = {
        "mode": "agentic",
        "model": MODEL,
        "rounds": round_num + 1,
        "tool_calls": tool_calls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "duration_ms": round(elapsed_ms),
        "cost_estimate": round(input_cost + output_cost, 4),
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Summary: {result.get('summary', 'N/A')}")
    print(f"Suggestions: {len(result.get('suggestions', []))}")
    print("=== Agentic Review Complete ===")


def main():
    if not CONTEXT_PATH.exists():
        print("ERROR: No context file found. Run prepare-context.py first.")
        sys.exit(1)

    context = json.loads(CONTEXT_PATH.read_text())
    config = load_config()

    # Apply config overrides
    global MODEL, MAX_TOKENS, MAX_TOOL_ROUNDS
    review_config = config.get("review", {})
    MODEL = os.environ.get("REVIEW_AGENT_MODEL", review_config.get("model", MODEL))
    MAX_TOKENS = int(os.environ.get("REVIEW_AGENT_MAX_TOKENS", review_config.get("max_tokens", MAX_TOKENS)))
    MAX_TOOL_ROUNDS = review_config.get("max_tool_rounds", MAX_TOOL_ROUNDS)

    if context.get("skip"):
        print(f"Skipping: {context.get('reason', 'unknown')}")
        OUTPUT_PATH.write_text(json.dumps({"summary": "Skipped", "suggestions": [], "skip": True}))
        return

    is_dry_run = os.environ.get("REVIEW_AGENT_DRY_RUN", "false").lower() == "true"
    if is_dry_run or not os.environ.get("ANTHROPIC_API_KEY", ""):
        dry_run(context, config)
    else:
        live_review(context, config)


if __name__ == "__main__":
    main()
