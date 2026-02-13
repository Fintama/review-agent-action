# AI Code Review Agent

An agentic PR reviewer powered by Claude. It reads your project's rules, specs, and codebase to post intelligent inline review comments on pull requests.

## Features

- **Agentic review** — Claude uses tools (read files, search code, read rules) to investigate the PR, not just scan the diff
- **Inline comments** — suggestions are posted as inline comments on the exact lines, CodeRabbit-style
- **Auto-approval** — safe PRs get auto-approved; risky PRs (security, structural, complex) require human review
- **Project-aware** — reads your project rules (e.g., `.cursor/rules/`), specs, and documentation
- **Configurable** — per-repo config file lets you customize risk paths, branding, and behavior
- **Smart dedup** — old reviews are collapsed, summary comments are updated in place

## Quick Start

### 1. Add the workflow

Create `.github/workflows/review-agent.yml` in your repo:

```yaml
name: AI Code Review

on:
  pull_request:
    branches: [main]
    types: [opened, synchronize]

permissions:
  contents: read
  pull-requests: write

jobs:
  review:
    runs-on: ubuntu-latest
    if: github.actor != 'dependabot[bot]'
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: fintama/review-agent-action@v1
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

### 2. Add secrets

In your repo's Settings > Secrets and variables > Actions, add:

| Secret | Required | Description |
|--------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude |
| `REVIEW_APP_ID` | No | GitHub App ID (for custom review identity) |
| `REVIEW_APP_PRIVATE_KEY` | No | GitHub App private key (PEM) |

> **Tip:** Set these as org-level secrets in your GitHub org to share them across all repos.

### 3. (Optional) Add project config

Create `.github/review-agent/config.yaml` for project-specific settings:

```yaml
project:
  name: "My Project"
  description: "A web application built with Django and React"
  tech_stack: "Django, PostgreSQL, React, TypeScript"

rules:
  enabled: true
  directory: ".cursor/rules"

risk:
  structural_paths:
    - "migrations/"
    - "models/"
  security_paths:
    - "auth/"
    - "middleware/"
```

If you don't provide a config, sensible defaults are used.

## Configuration Reference

The config file supports these sections:

### `project` — Identity (used in the LLM prompt)

```yaml
project:
  name: "Swisper"
  description: "AI personal assistant"
  tech_stack: "FastAPI, React, PostgreSQL"
```

### `rules` — Project rules

```yaml
rules:
  enabled: true                    # Set to false if you don't have rules
  directory: ".cursor/rules"       # Where rule files live
  file_pattern: "*.mdc"            # Glob for rule files
  always_include:                  # Rules to always include (even without glob match)
    - "00-workflow"
    - "03-pull-requests"
```

### `docs` — Spec/plan discovery

```yaml
docs:
  enabled: true
  spec_dirs:                       # Directories to search for specs
    - "docs/specs"
    - "docs/plans"
  directory_doc_map:               # Fallback: map code dirs to docs
    "src/auth/":
      - "docs/auth-design.md"
```

### `blast_radius` — Import tracing

```yaml
blast_radius:
  enabled: true
  language: "auto"                 # "python", "typescript", or "auto"
  source_dirs: ["src/"]            # Where to search for importers
  module_prefix_strip: "src/"      # Strip from file paths for module names
  max_files: 10
```

### `risk` — Auto-approve risk assessment

```yaml
risk:
  structural_paths: ["migrations/", "models/"]
  security_paths: ["auth/", "middleware/"]
  security_dep_files: ["requirements.txt", "package.json"]
  infrastructure_patterns: ["Dockerfile", ".github/workflows/"]
  domain_paths:                    # For cross-cutting detection
    "src/api/": "api"
    "src/services/": "services"
  thresholds:
    max_code_files: 15
    max_code_lines: 1000
    cross_cutting_domains: 3
```

### `review` — LLM behavior

```yaml
review:
  model: "claude-sonnet-4-20250514"
  max_tokens: 8192
  max_tool_rounds: 10
  auto_approve_enabled: true
```

### `branding` — Comment appearance

```yaml
branding:
  review_header: "## Code Review"
  comment_tag: "<!-- my-review-bot -->"
  summary_tag: "<!-- my-review-bot-summary -->"
```

## Action Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `anthropic-api-key` | Yes | — | Anthropic API key |
| `app-id` | No | — | GitHub App ID |
| `app-private-key` | No | — | GitHub App private key |
| `github-token` | No | `github.token` | Fallback token if no App |
| `config-path` | No | `.github/review-agent/config.yaml` | Path to project config |
| `model` | No | `claude-sonnet-4-20250514` | Claude model |
| `max-tokens` | No | `8192` | Max tokens for response |
| `auto-approve` | No | `true` | Enable auto-approval |
| `dry-run` | No | `false` | Run without posting |

## How It Works

1. **prepare-context.py** — Parses the PR diff, matches changed files against project rules, discovers relevant specs/plans, traces blast radius
2. **llm-review.py** — Sends the context to Claude with tools (read_file, search_code, read_rule, list_directory). The agent investigates the codebase and produces a structured review.
3. **post-review.py** — Posts the review as inline comments + summary on the PR. Handles auto-approval, risk assessment, and comment dedup.

## GitHub App Setup (Recommended)

Using a GitHub App gives the review agent its own identity (e.g., "Fintama Review Agent" instead of "github-actions[bot]").

1. Create a GitHub App at https://github.com/organizations/fintama/settings/apps/new
2. Set permissions: Pull Requests (read & write), Contents (read)
3. Install it on the repos you want
4. Add the App ID and private key as org secrets (`REVIEW_APP_ID`, `REVIEW_APP_PRIVATE_KEY`)

## Examples

See the `examples/` directory for:
- `config-swisper.yaml` — Full config for a complex monorepo
- `config-minimal.yaml` — Minimal config for a simple project
- `workflow.yml` — Example GitHub Actions workflow

## License

Private — Fintama internal use.
