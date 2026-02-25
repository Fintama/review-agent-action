"""Tests for llm-review.py — JSON extraction, coverage tracking, completeness check, prompts."""

import json
import pytest

# Module imports happen after conftest.py adds scripts/ to sys.path
# We need to set env vars before importing to avoid git calls at module level
import os
os.environ.setdefault("GITHUB_WORKSPACE", "/tmp/test-repo")

from pathlib import Path


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    """_extract_json must handle all output formats Claude might produce."""

    @pytest.fixture(autouse=True)
    def _import(self):
        # Import here so conftest.py has run
        import importlib
        mod = importlib.import_module("llm-review")
        self._extract_json = mod._extract_json

    def test_plain_json(self):
        raw = '{"summary": "Looks good", "suggestions": []}'
        result = self._extract_json(raw)
        assert result == {"summary": "Looks good", "suggestions": []}

    def test_json_in_markdown_fence(self):
        raw = '```json\n{"summary": "OK", "suggestions": []}\n```'
        result = self._extract_json(raw)
        assert result["summary"] == "OK"

    def test_json_with_leading_text(self):
        raw = 'Here is my review:\n{"summary": "Done", "suggestions": []}'
        result = self._extract_json(raw)
        assert result["summary"] == "Done"

    def test_invalid_json_returns_none(self):
        assert self._extract_json("not json at all") is None

    def test_empty_string_returns_none(self):
        assert self._extract_json("") is None

    def test_nested_json(self):
        raw = json.dumps({
            "summary": "Issues found",
            "suggestions": [{"file": "a.py", "line": 1, "severity": "warning", "title": "T", "body": "B"}],
        })
        result = self._extract_json(raw)
        assert len(result["suggestions"]) == 1


# ---------------------------------------------------------------------------
# compute_max_tool_rounds — dynamic round allocation
# ---------------------------------------------------------------------------

class TestComputeMaxToolRounds:
    """MAX_TOOL_ROUNDS should scale with the number of changed files."""

    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib
        self.mod = importlib.import_module("llm-review")

    def test_small_pr_gets_minimum_rounds(self):
        result = self.mod.compute_max_tool_rounds(2)
        assert result >= 10, "Small PRs should still get at least 10 rounds"

    def test_large_pr_gets_more_rounds(self):
        small = self.mod.compute_max_tool_rounds(3)
        large = self.mod.compute_max_tool_rounds(15)
        assert large > small, "More files should mean more rounds"

    def test_never_exceeds_hard_ceiling(self):
        result = self.mod.compute_max_tool_rounds(100)
        assert result <= 30, "Hard ceiling should cap at 30"

    def test_zero_files_gets_minimum(self):
        result = self.mod.compute_max_tool_rounds(0)
        assert result >= 10

    def test_five_files_gets_twenty_rounds(self):
        """5-file PR: 10 base + 5*2 = 20 rounds (last one reserved for response)."""
        result = self.mod.compute_max_tool_rounds(5)
        assert result == 20


# ---------------------------------------------------------------------------
# check_file_coverage — detect unreviewed files
# ---------------------------------------------------------------------------

class TestCheckFileCoverage:
    """The completeness check must identify files the agent hasn't looked at."""

    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib
        self.mod = importlib.import_module("llm-review")

    def test_all_files_covered(self):
        changed = ["src/a.py", "src/b.py"]
        reviewed = {"src/a.py", "src/b.py"}
        result = {"suggestions": [
            {"file": "src/a.py", "line": 10, "severity": "warning", "title": "T", "body": "B"},
            {"file": "src/b.py", "line": 20, "severity": "praise", "title": "T", "body": "B"},
        ]}
        missed = self.mod.check_file_coverage(changed, reviewed, result)
        assert missed == []

    def test_missed_files_detected(self):
        changed = ["src/a.py", "src/b.py", "src/c.py"]
        reviewed = {"src/a.py"}
        result = {"suggestions": [
            {"file": "src/a.py", "line": 10, "severity": "warning", "title": "T", "body": "B"},
        ]}
        missed = self.mod.check_file_coverage(changed, reviewed, result)
        assert "src/b.py" in missed
        assert "src/c.py" in missed

    def test_doc_files_excluded(self):
        """Markdown/doc files shouldn't count as unreviewed code files."""
        changed = ["src/a.py", "README.md", "docs/guide.md"]
        reviewed = {"src/a.py"}
        result = {"suggestions": []}
        missed = self.mod.check_file_coverage(changed, reviewed, result)
        assert missed == []

    def test_files_mentioned_in_suggestions_count_as_reviewed(self):
        changed = ["src/a.py", "src/b.py"]
        reviewed = set()  # agent didn't use read_file on either
        result = {"suggestions": [
            {"file": "src/a.py", "line": 10, "severity": "warning", "title": "T", "body": "B"},
            {"file": "src/b.py", "line": 5, "severity": "suggestion", "title": "T", "body": "B"},
        ]}
        missed = self.mod.check_file_coverage(changed, reviewed, result)
        assert missed == []

    def test_lockfiles_excluded(self):
        changed = ["src/a.py", "pnpm-lock.yaml", "package-lock.json"]
        reviewed = {"src/a.py"}
        result = {"suggestions": []}
        missed = self.mod.check_file_coverage(changed, reviewed, result)
        assert missed == []


# ---------------------------------------------------------------------------
# build_coverage_followup — the nudge message
# ---------------------------------------------------------------------------

class TestBuildCoverageFollowup:
    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib
        self.mod = importlib.import_module("llm-review")

    def test_contains_missed_files(self):
        msg = self.mod.build_coverage_followup(["src/b.py", "src/c.py"])
        assert "src/b.py" in msg
        assert "src/c.py" in msg

    def test_asks_to_update_json(self):
        msg = self.mod.build_coverage_followup(["src/b.py"])
        assert "JSON" in msg or "json" in msg


# ---------------------------------------------------------------------------
# Prompt content — no efficiency bias
# ---------------------------------------------------------------------------

class TestPromptContent:
    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib
        self.mod = importlib.import_module("llm-review")

    def test_system_prompt_no_efficiency_cap(self):
        context = {
            "project": {"name": "Test"},
            "rules": [],
            "spec_docs": [],
        }
        prompt = self.mod.build_system_prompt(context, {})
        assert "aim for 3-6" not in prompt.lower()
        assert "aim for 3-5" not in prompt.lower()

    def test_system_prompt_mentions_all_files(self):
        context = {
            "project": {"name": "Test"},
            "rules": [],
            "spec_docs": [],
        }
        prompt = self.mod.build_system_prompt(context, {})
        lower = prompt.lower()
        assert "all changed files" in lower or "every changed file" in lower

    def test_user_message_no_efficiency_cap(self):
        context = {
            "changed_files": ["a.py", "b.py"],
            "diff": "some diff",
            "pr_title": "Test PR",
            "branch_name": "test",
        }
        msg = self.mod.build_user_message(context)
        assert "aim for 3-5" not in msg
        assert "not exhaustive" not in msg
