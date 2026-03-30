# Project Codex Defaults

Prefer the `ruflo` MCP server for multi-step or repeatable work when it is available.

Use RuFlo first for:
- multi-file changes
- bug fix plus verification
- refactors that touch tests or docs
- tasks that benefit from orchestration or stored patterns

Skip RuFlo for:
- one-line edits
- tiny wording changes
- quick local experiments

<!-- ai-policy:start -->
## Power of 10 default coding policy

Use Gerard J. Holzmann's Power of 10 as the default coding baseline for coding work in this repo.

- Treat `docs/ai/power-of-10-baseline.md` as the repo-local operational baseline.
- Treat `docs/ai/second-brain-sources.md` as the pointer to the canonical second-brain notes and local source PDFs.
- If this repo can access the source vault, the exact rule source is `C:/D Drive/Projects/Second Brain with Obsidian/knowledge/graph/holzmanns power of 10 should anchor coding defaults.md`.
- The conservative non-C interpretation source is `C:/D Drive/Projects/Second Brain with Obsidian/knowledge/graph/power of 10 can guide non-c coding without changing the original rules.md`.
- If a Power of 10 rule does not map cleanly to the current language or framework, say so explicitly instead of inventing a new "Holzmann rule."
<!-- ai-policy:end -->
