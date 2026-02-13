"""
config_loader.py — Load and merge project configuration.

Configuration is resolved in this order (later overrides earlier):
1. Built-in defaults (defaults/config.yaml in the action repo)
2. Project config (.github/review-agent/config.yaml in the consuming repo)
3. Environment variable overrides

This module is imported by all three scripts.
"""

import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    # pyyaml not yet installed — happens during action setup
    yaml = None


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins for leaf values."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> dict:
    """Load configuration from defaults + project config + env overrides.

    Environment variables:
        REVIEW_AGENT_CONFIG: Path to project config (relative to repo root)
        REVIEW_AGENT_ACTION_PATH: Path to the action's own directory
    """
    if yaml is None:
        print("ERROR: pyyaml is required. Install with: pip install pyyaml")
        sys.exit(1)

    # 1. Load built-in defaults from the action repo
    action_path = Path(os.environ.get("REVIEW_AGENT_ACTION_PATH", Path(__file__).parent.parent))
    defaults_path = action_path / "defaults" / "config.yaml"

    config = {}
    if defaults_path.exists():
        config = yaml.safe_load(defaults_path.read_text(encoding="utf-8")) or {}

    # 2. Load project-specific config from the consuming repo
    repo_root = _find_repo_root()
    config_rel_path = os.environ.get("REVIEW_AGENT_CONFIG", ".github/review-agent/config.yaml")
    project_config_path = repo_root / config_rel_path

    if project_config_path.exists():
        project_config = yaml.safe_load(project_config_path.read_text(encoding="utf-8")) or {}
        config = _deep_merge(config, project_config)
        print(f"  Loaded project config from {config_rel_path}")
    else:
        print(f"  No project config at {config_rel_path} — using defaults")

    # 3. Apply environment variable overrides
    if os.environ.get("REVIEW_AGENT_AUTO_APPROVE"):
        config.setdefault("review", {})["auto_approve_enabled"] = (
            os.environ["REVIEW_AGENT_AUTO_APPROVE"].lower() == "true"
        )
    if os.environ.get("REVIEW_AGENT_MODEL"):
        config.setdefault("review", {})["model"] = os.environ["REVIEW_AGENT_MODEL"]
    if os.environ.get("REVIEW_AGENT_MAX_TOKENS"):
        config.setdefault("review", {})["max_tokens"] = int(os.environ["REVIEW_AGENT_MAX_TOKENS"])

    return config


def _find_repo_root() -> Path:
    """Find the Git repository root."""
    # In GitHub Actions, GITHUB_WORKSPACE is the repo root
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace:
        return Path(workspace)

    # Fall back to git rev-parse
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass

    # Last resort: current directory
    return Path.cwd()


def get_repo_root() -> Path:
    """Public accessor for repo root."""
    return _find_repo_root()
