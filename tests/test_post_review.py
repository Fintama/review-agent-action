"""Tests for post-review.py — diff parsing, comment resolution, risk assessment."""

import json
import pytest
import os

os.environ.setdefault("GITHUB_WORKSPACE", "/tmp/test-repo")

# ---------------------------------------------------------------------------
# get_diff_line_sets — parsing diffs into commentable line maps
# ---------------------------------------------------------------------------

class TestGetDiffLineSets:
    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib
        self.mod = importlib.import_module("post-review")

    def test_single_file_addition(self):
        diff = (
            "diff --git a/src/a.py b/src/a.py\n"
            "--- a/src/a.py\n"
            "+++ b/src/a.py\n"
            "@@ -10,3 +10,4 @@\n"
            " existing line\n"
            "+new line\n"
            " another line\n"
        )
        result = self.mod.get_diff_line_sets(diff)
        assert "src/a.py" in result
        assert 10 in result["src/a.py"]
        assert 11 in result["src/a.py"]
        assert 12 in result["src/a.py"]

    def test_multiple_files(self):
        diff = (
            "diff --git a/a.py b/a.py\n"
            "+++ b/a.py\n"
            "@@ -1,2 +1,2 @@\n"
            "+line1\n"
            "diff --git a/b.py b/b.py\n"
            "+++ b/b.py\n"
            "@@ -5,2 +5,2 @@\n"
            "+line5\n"
        )
        result = self.mod.get_diff_line_sets(diff)
        assert "a.py" in result
        assert "b.py" in result

    def test_empty_diff(self):
        result = self.mod.get_diff_line_sets("")
        assert result == {}


# ---------------------------------------------------------------------------
# get_changed_line_ranges — parse inter-push diff for comment resolution
# ---------------------------------------------------------------------------

class TestGetChangedLineRanges:
    """Parse an inter-push diff to find which lines were modified per file."""

    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib
        self.mod = importlib.import_module("post-review")

    def test_single_hunk(self):
        diff = (
            "diff --git a/src/a.py b/src/a.py\n"
            "+++ b/src/a.py\n"
            "@@ -70,5 +70,6 @@\n"
            " context\n"
            "+new line\n"
            " context\n"
        )
        result = self.mod.get_changed_line_ranges(diff)
        assert "src/a.py" in result
        assert 71 in result["src/a.py"]

    def test_multiple_hunks(self):
        diff = (
            "diff --git a/src/a.py b/src/a.py\n"
            "+++ b/src/a.py\n"
            "@@ -10,3 +10,4 @@\n"
            "+line\n"
            "@@ -50,3 +51,4 @@\n"
            "+another\n"
        )
        result = self.mod.get_changed_line_ranges(diff)
        lines = result["src/a.py"]
        assert 10 in lines
        assert 51 in lines

    def test_no_changes(self):
        result = self.mod.get_changed_line_ranges("")
        assert result == {}


# ---------------------------------------------------------------------------
# is_comment_addressed — heuristic detection
# ---------------------------------------------------------------------------

class TestIsCommentAddressed:
    """Determine if a previous bot comment was addressed by the new push."""

    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib
        self.mod = importlib.import_module("post-review")

    def test_exact_line_changed(self):
        changed_ranges = {"src/a.py": {75, 76, 77, 78}}
        assert self.mod.is_comment_addressed("src/a.py", 77, changed_ranges) is True

    def test_nearby_line_changed(self):
        """Changes within 5 lines of the comment count as addressed."""
        changed_ranges = {"src/a.py": {80}}
        assert self.mod.is_comment_addressed("src/a.py", 77, changed_ranges) is True

    def test_distant_line_not_addressed(self):
        """Changes far from the comment don't count."""
        changed_ranges = {"src/a.py": {200}}
        assert self.mod.is_comment_addressed("src/a.py", 77, changed_ranges) is False

    def test_different_file_not_addressed(self):
        changed_ranges = {"src/b.py": {77}}
        assert self.mod.is_comment_addressed("src/a.py", 77, changed_ranges) is False

    def test_file_not_in_diff(self):
        changed_ranges = {}
        assert self.mod.is_comment_addressed("src/a.py", 77, changed_ranges) is False

    def test_no_line_number_not_addressed(self):
        changed_ranges = {"src/a.py": {1, 2, 3}}
        assert self.mod.is_comment_addressed("src/a.py", 0, changed_ranges) is False


# ---------------------------------------------------------------------------
# find_closest_commentable_line
# ---------------------------------------------------------------------------

class TestFindClosestCommentableLine:
    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib
        self.mod = importlib.import_module("post-review")

    def test_exact_match(self):
        assert self.mod.find_closest_commentable_line({10, 11, 12}, 11) == 11

    def test_closest_above(self):
        assert self.mod.find_closest_commentable_line({10, 15}, 12) == 10

    def test_closest_below(self):
        assert self.mod.find_closest_commentable_line({10, 15}, 14) == 15

    def test_too_far(self):
        assert self.mod.find_closest_commentable_line({100}, 10) is None

    def test_empty_set(self):
        assert self.mod.find_closest_commentable_line(set(), 10) is None


# ---------------------------------------------------------------------------
# determine_review_event
# ---------------------------------------------------------------------------

class TestDetermineReviewEvent:
    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib
        self.mod = importlib.import_module("post-review")

    def test_critical_blocks(self):
        suggestions = [{"severity": "critical", "title": "Bug"}]
        event, _ = self.mod.determine_review_event(
            suggestions, [], {}, "", {},
        )
        assert event == "REQUEST_CHANGES"

    def test_clean_pr_approves(self):
        event, _ = self.mod.determine_review_event(
            [], [], {}, "", {"review": {"auto_approve_enabled": True}},
        )
        assert event == "APPROVE"

    def test_auto_approve_disabled(self):
        event, _ = self.mod.determine_review_event(
            [], [], {}, "", {"review": {"auto_approve_enabled": False}},
        )
        assert event == "COMMENT"


# ---------------------------------------------------------------------------
# compute_diff_stats
# ---------------------------------------------------------------------------

class TestComputeDiffStats:
    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib
        self.mod = importlib.import_module("post-review")

    def test_counts_additions_and_removals(self):
        diff = (
            "+++ b/src/a.py\n"
            "+added line\n"
            "-removed line\n"
            "+another added\n"
        )
        stats = self.mod.compute_diff_stats(diff, {})
        assert stats["lines_added"] == 2
        assert stats["lines_removed"] == 1

    def test_doc_files_separated(self):
        diff = (
            "+++ b/README.md\n"
            "+doc line\n"
            "+++ b/src/a.py\n"
            "+code line\n"
        )
        stats = self.mod.compute_diff_stats(diff, {})
        assert stats["code_lines_added"] == 1
        assert stats["lines_added"] == 2


# ---------------------------------------------------------------------------
# format_suggestion_body
# ---------------------------------------------------------------------------

class TestFormatSuggestionBody:
    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib
        self.mod = importlib.import_module("post-review")

    def test_includes_title_and_body(self):
        s = {"severity": "warning", "title": "Missing check", "body": "Add null check", "rule": "S-22"}
        result = self.mod.format_suggestion_body(s)
        assert "Missing check" in result
        assert "Add null check" in result
        assert "S-22" in result

    def test_legacy_consider_mapped_to_warning(self):
        s = {"severity": "consider", "title": "T", "body": "B"}
        result = self.mod.format_suggestion_body(s)
        assert "⚠️" in result


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------

class TestRiskAssessment:
    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib
        self.mod = importlib.import_module("post-review")

    def test_structural_risk(self):
        is_risky, _ = self.mod.has_structural_risk(
            ["alembic/versions/001.py"], {},
        )
        assert is_risky

    def test_no_structural_risk(self):
        is_risky, _ = self.mod.has_structural_risk(
            ["src/utils.py"], {},
        )
        assert not is_risky

    def test_security_risk_env_file(self):
        is_risky, _ = self.mod.has_security_risk(
            ["config/.env"], "", {},
        )
        assert is_risky

    def test_complexity_risk_many_files(self):
        files = [f"src/file{i}.py" for i in range(20)]
        is_risky, _ = self.mod.has_complexity_risk(
            files, {}, [], {},
        )
        assert is_risky
