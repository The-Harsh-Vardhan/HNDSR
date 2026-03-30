---
description: "Start work in this repo by grounding in the local Power of 10 policy and the second-brain source pointers."
name: "Start With Second Brain and Ruflo"
argument-hint: "Goal or task to start"
agent: "agent"
tools:
  - ruflo/*
---
# Start work in this project

You are helping me begin work on: ${input:Goal or task to start}

Follow this workflow unless I explicitly override it:

1. Read `docs/ai/power-of-10-baseline.md` before giving coding guidance.
2. Read `docs/ai/second-brain-sources.md` if the task depends on the canonical vault notes or original PDFs.
3. If the repo has access to the second brain, use the source pointers in `docs/ai/second-brain-sources.md` rather than guessing.
4. Use Ruflo when the task benefits from orchestration, task tracking, workflows, terminal or browser automation, or project inspection.
5. If a Power of 10 rule does not map cleanly to the current language or framework, say so explicitly instead of inventing a new "Holzmann rule."
