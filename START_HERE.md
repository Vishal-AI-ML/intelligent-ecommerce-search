# Start Here — Claude Code Pro

## 1. Install prerequisites

Install Git, Python 3.12, VS Code, Docker Desktop, and uv.

Install uv in Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 2. Install Claude Code

Windows PowerShell:

```powershell
irm https://claude.ai/install.ps1 | iex
```

macOS, Linux, or WSL:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Restart the terminal, then verify:

```powershell
claude --version
```

## 3. Open this starter project

Extract the ZIP, then open PowerShell in the extracted folder:

```powershell
cd PATH\TO\ecommerce-search-claude-starter
git init
code .
claude
```

When prompted, log in using the same account that has Claude Pro. Inside Claude Code, run:

```text
/status
```

If an `ANTHROPIC_API_KEY` environment variable is configured, Claude Code may use separately billed API access instead of your Pro allowance. Remove it if you intend to use only the subscription.

## 4. Start Milestone 0

Copy the first prompt from `FIRST_PROMPTS.md`. Ask for the plan first, approve it only after review, then send the implementation prompt.

## 5. Work milestone by milestone

Use this loop:

```text
plan -> review -> implement -> test -> review -> fix -> commit -> fresh session
```

Do not ask Claude to build the whole repository in one attempt.

## 6. Commit Milestone 0

After review:

```powershell
git status
git diff
git add .
git commit -m "docs: complete milestone 0 specification"
```

## Human responsibilities

Claude Code can implement the software, but you must personally validate human relevance labels, the 30-product catalog review, product decisions, cloud spending, API credentials, and every metric used in a CV or README.
