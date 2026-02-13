#!/usr/bin/env python3
"""
prepare-context.py — Build the review prompt from rules, spec, blast radius, and diff.

Generalized version that reads project-specific configuration from the
consuming repo's config file (default: .github/review-agent/config.yaml).

It:
1. Parses rule files (e.g., .cursor/rules/*.mdc) frontmatter to extract globs
2. Matches changed files against rule globs to select applicable rules
3. Discovers spec/plan from PR description (Level 1), fuzzy match (Level 2), or directory mapping (Level 3)
4. Traces blast radius: files that import from changed modules
5. Assembles everything into a structured prompt saved to /tmp/review-context.json
"""

import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Add scripts directory to path for config_loader
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import load_config, get_repo_root

OUTPUT_PATH = Path("/tmp/review-context.json")


# ---------------------------------------------------------------------------
# Rule parsing (supports .mdc frontmatter format and plain markdown)
# ---------------------------------------------------------------------------

def parse_rule_frontmatter(filepath: Path) -> dict:
    """Extract YAML-like frontmatter from a rule file (.mdc or .md).

    Supports the common format:
        ---
        description: "..."
        globs: "*.py,*.ts"
        ---
        Rule content here...
    """
    content = filepath.read_text(encoding="utf-8")
    if not content.startswith("---"):
        return {"description": "", "globs": "", "content": content}

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {"description": "", "globs": "", "content": content}

    frontmatter = parts[1].strip()
    body = parts[2].strip()

    result = {"content": body, "description": "", "globs": ""}
    for line in frontmatter.split("\n"):
        line = line.strip()
        if line.startswith("description:"):
            result["description"] = line.split(":", 1)[1].strip().strip('"')
        elif line.startswith("globs:"):
            result["globs"] = line.split(":", 1)[1].strip().strip('"')

    return result


def match_file_to_globs(filepath: str, globs_str: str) -> bool:
    """Check if a file path matches any of the comma-separated glob patterns."""
    if not globs_str:
        return False
    for pattern in globs_str.split(","):
        pattern = pattern.strip()
        if not pattern:
            continue
        if fnmatch.fnmatch(filepath, pattern):
            return True
    return False


def select_applicable_rules(changed_files: list[str], config: dict) -> list[dict]:
    """Match changed files against rule globs, return applicable rules."""
    rules_config = config.get("rules", {})
    if not rules_config.get("enabled", True):
        return []

    repo_root = get_repo_root()
    rules_dir = repo_root / rules_config.get("directory", ".cursor/rules")
    file_pattern = rules_config.get("file_pattern", "*.mdc")
    always_include = set(rules_config.get("always_include", []))
    max_rules = config.get("review", {}).get("max_rule_files", 25)

    if not rules_dir.exists():
        print(f"  Rules directory not found: {rules_dir}")
        return []

    rules = []
    seen = set()

    # Sort: S-* first (shared foundation), then B-*/F-* (specific), then meta
    def rule_sort_key(path):
        name = path.stem
        if name.startswith("S-"):
            return (0, name)
        if name.startswith("B-") or name.startswith("F-"):
            return (1, name)
        return (2, name)

    for rule_file in sorted(rules_dir.glob(file_pattern), key=rule_sort_key):
        parsed = parse_rule_frontmatter(rule_file)
        globs = parsed["globs"]
        name = rule_file.stem

        # Always-include rules (no glob matching needed)
        if name in always_include:
            if name not in seen:
                rules.append({
                    "name": name,
                    "description": parsed["description"],
                })
                seen.add(name)
            continue

        # Rules without globs — skip unless always-included
        if not globs:
            continue

        # Check if any changed file matches this rule's globs
        for cf in changed_files:
            if match_file_to_globs(cf, globs) and name not in seen:
                rules.append({
                    "name": name,
                    "description": parsed["description"],
                })
                seen.add(name)
                break

    return rules[:max_rules]


# ---------------------------------------------------------------------------
# Spec/plan discovery (3 levels)
# ---------------------------------------------------------------------------

def discover_spec_from_pr(pr_body: str, config: dict) -> list[dict]:
    """Level 1: Parse PR description for spec/plan file paths."""
    docs = []
    if not pr_body:
        return docs

    repo_root = get_repo_root()
    spec_dirs = config.get("docs", {}).get("spec_dirs", ["docs/specs", "docs/plans"])

    # Build regex from configured spec dirs
    dir_patterns = "|".join(re.escape(d) for d in spec_dirs)
    if not dir_patterns:
        # Fallback: match any .md path under docs/
        dir_patterns = r"docs/[^\s\)\"]*"
    pattern = rf"({dir_patterns}/[^\s\)\"]+\.md)"
    matches = re.findall(pattern, pr_body)

    for match in matches:
        filepath = repo_root / match
        if filepath.exists():
            docs.append({"path": match})

    return docs


def discover_spec_fuzzy(branch_name: str, pr_title: str, config: dict) -> list[dict]:
    """Level 2: Fuzzy match branch name / PR title to spec filenames."""
    docs = []
    repo_root = get_repo_root()
    spec_dirs = config.get("docs", {}).get("spec_dirs", ["docs/specs", "docs/plans"])

    # Extract keywords from branch and title
    text = f"{branch_name} {pr_title}".lower()
    text = re.sub(r"^(feature|fix|refactor|chore|docs|test)/", "", text)
    keywords = set(re.findall(r"[a-z]{3,}", text))
    keywords -= {"the", "and", "for", "from", "with", "this", "that", "feat", "fix"}

    if not keywords:
        return docs

    for spec_dir in spec_dirs:
        doc_dir = repo_root / spec_dir
        if not doc_dir.exists():
            continue
        for md_file in doc_dir.glob("*.md"):
            name_lower = md_file.stem.lower().replace("_", " ").replace("-", " ")
            matches = sum(1 for kw in keywords if kw in name_lower)
            if matches >= 2:
                docs.append({"path": str(md_file.relative_to(repo_root))})

    return docs[:3]


def discover_spec_directory_map(changed_files: list[str], config: dict) -> list[dict]:
    """Level 3: Fall back to directory-doc-map from config."""
    docs = []
    repo_root = get_repo_root()
    doc_map = config.get("docs", {}).get("directory_doc_map", {})

    if not doc_map:
        return docs

    seen_paths = set()
    for cf in changed_files:
        for directory, doc_paths in doc_map.items():
            if directory in cf:
                for doc_path in doc_paths:
                    if doc_path in seen_paths:
                        continue
                    filepath = repo_root / doc_path
                    if filepath.exists():
                        docs.append({"path": doc_path})
                        seen_paths.add(doc_path)

    return docs[:5]


# ---------------------------------------------------------------------------
# Blast radius tracing
# ---------------------------------------------------------------------------

def _detect_language(changed_files: list[str]) -> str:
    """Detect primary language from changed files."""
    py_count = sum(1 for f in changed_files if f.endswith(".py"))
    ts_count = sum(1 for f in changed_files if f.endswith((".ts", ".tsx", ".js", ".jsx")))
    if py_count >= ts_count:
        return "python"
    return "typescript"


def trace_blast_radius(changed_files: list[str], config: dict) -> list[dict]:
    """Find files that import from the changed modules."""
    br_config = config.get("blast_radius", {})
    if not br_config.get("enabled", True):
        return []

    repo_root = get_repo_root()
    max_files = br_config.get("max_files", 10)
    language = br_config.get("language", "auto")
    prefix_strip = br_config.get("module_prefix_strip", "")
    source_dirs = br_config.get("source_dirs", [])

    if language == "auto":
        language = _detect_language(changed_files)

    blast = []
    seen = set()

    if language == "python":
        py_files = [f for f in changed_files if f.endswith(".py")]
        if not py_files:
            return blast

        search_dirs = [str(repo_root / d) for d in source_dirs] if source_dirs else [str(repo_root)]

        for cf in py_files[:5]:
            module = cf.replace("/", ".").replace(".py", "")
            if prefix_strip:
                module = re.sub(rf"^{re.escape(prefix_strip).replace('/', '\\.')}\.?", "", module)

            if not module or module in seen:
                continue
            seen.add(module)

            for search_dir in search_dirs:
                try:
                    result = subprocess.run(
                        ["grep", "-rl", "--include=*.py",
                         f"from {module} import\\|import {module}", search_dir],
                        capture_output=True, text=True, timeout=10,
                    )
                    for line in result.stdout.strip().split("\n"):
                        if line and line not in seen:
                            rel_path = str(Path(line).relative_to(repo_root))
                            if rel_path not in changed_files:
                                try:
                                    with open(line) as f:
                                        head = "".join(f.readlines()[:50])
                                except Exception:
                                    head = ""
                                blast.append({"path": rel_path, "head": head[:2000]})
                                seen.add(line)
                except (subprocess.TimeoutExpired, Exception):
                    continue

    elif language == "typescript":
        ts_files = [f for f in changed_files if f.endswith((".ts", ".tsx", ".js", ".jsx"))]
        if not ts_files:
            return blast

        search_dirs = [str(repo_root / d) for d in source_dirs] if source_dirs else [str(repo_root)]

        for cf in ts_files[:5]:
            # Strip extension for import matching
            module_path = re.sub(r"\.(ts|tsx|js|jsx)$", "", cf)
            basename = Path(cf).stem

            if basename in seen:
                continue
            seen.add(basename)

            for search_dir in search_dirs:
                try:
                    result = subprocess.run(
                        ["grep", "-rl", "--include=*.ts", "--include=*.tsx",
                         "--include=*.js", "--include=*.jsx",
                         f"from.*{basename}", search_dir],
                        capture_output=True, text=True, timeout=10,
                    )
                    for line in result.stdout.strip().split("\n"):
                        if line and line not in seen:
                            rel_path = str(Path(line).relative_to(repo_root))
                            if rel_path not in changed_files:
                                blast.append({"path": rel_path, "head": ""})
                                seen.add(line)
                except (subprocess.TimeoutExpired, Exception):
                    continue

    return blast[:max_files]


# ---------------------------------------------------------------------------
# Diff handling
# ---------------------------------------------------------------------------

def get_diff(config: dict) -> str:
    """Get the PR diff."""
    max_lines = config.get("review", {}).get("max_diff_lines", 1500)

    diff_file = Path("/tmp/pr.diff")
    if diff_file.exists():
        diff = diff_file.read_text(encoding="utf-8", errors="replace")
    else:
        try:
            result = subprocess.run(
                ["git", "diff", "origin/main...HEAD"],
                capture_output=True, text=True, timeout=30,
                cwd=str(get_repo_root()),
            )
            diff = result.stdout
        except Exception:
            diff = ""

    lines = diff.split("\n")
    if len(lines) > max_lines:
        diff = "\n".join(lines[:max_lines])
        diff += f"\n\n[... truncated — {len(lines) - max_lines} additional lines not shown ...]"

    return diff


def get_changed_files() -> list[str]:
    """Get list of changed files."""
    files_path = Path("/tmp/changed-files.txt")
    if files_path.exists():
        return [f.strip() for f in files_path.read_text().strip().split("\n") if f.strip()]

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=str(get_repo_root()),
        )
        return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Review Agent: Preparing Context ===")

    config = load_config()

    # 1. Get changed files and diff
    changed_files = get_changed_files()
    print(f"Changed files: {len(changed_files)}")

    if not changed_files:
        print("No changed files found. Exiting.")
        OUTPUT_PATH.write_text(json.dumps({"skip": True, "reason": "No changed files"}))
        return

    diff = get_diff(config)
    print(f"Diff size: {len(diff)} chars")

    # 2. Select applicable rules
    rules = select_applicable_rules(changed_files, config)
    print(f"Applicable rules: {[r['name'] for r in rules]}")

    # 3. Discover spec/plan (3 levels)
    pr_body = os.environ.get("PR_BODY", "")
    pr_title = os.environ.get("PR_TITLE", "")
    branch_name = os.environ.get("BRANCH_NAME", "")

    spec_docs = discover_spec_from_pr(pr_body, config)
    if spec_docs:
        print(f"Spec discovery (Level 1 - PR description): {[d['path'] for d in spec_docs]}")
    else:
        spec_docs = discover_spec_fuzzy(branch_name, pr_title, config)
        if spec_docs:
            print(f"Spec discovery (Level 2 - fuzzy match): {[d['path'] for d in spec_docs]}")
        else:
            spec_docs = discover_spec_directory_map(changed_files, config)
            if spec_docs:
                print(f"Spec discovery (Level 3 - directory map): {[d['path'] for d in spec_docs]}")
            else:
                print("Spec discovery: No specs found")

    # 4. Trace blast radius
    blast_radius = trace_blast_radius(changed_files, config)
    if blast_radius:
        print(f"Blast radius: {[b['path'] for b in blast_radius]}")

    # 5. Assemble context
    context = {
        "skip": False,
        "changed_files": changed_files,
        "rules": rules,
        "spec_docs": spec_docs,
        "blast_radius": blast_radius,
        "diff": diff,
        "pr_title": pr_title,
        "pr_body": pr_body[:2000],
        "branch_name": branch_name,
        "project": config.get("project", {}),
        "config": {
            "rules_dir": config.get("rules", {}).get("directory", ""),
            "branding": config.get("branding", {}),
        },
    }

    OUTPUT_PATH.write_text(json.dumps(context, indent=2, ensure_ascii=False))
    print(f"Context written to {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")
    print("=== Context Preparation Complete ===")


if __name__ == "__main__":
    main()
