# LeCoder: Paper-to-Code Skill

A skill for AI coding agents to implement ML research papers from scratch when no official code exists.

**Motto**: Less Code, More Reproduction

## What is This?

This skill teaches AI agents (like Claude, Cursor, etc.) how to systematically implement research papers from PDFs into working Python code. It provides a structured workflow from paper analysis to production-ready code.

## How to Use

### For Claude Desktop/API

1. **Download this skill folder** as a ZIP
2. **Upload to Claude**:
   - Claude Desktop: Place in `~/.claude/skills/paper-to-code/`
   - Claude API: Upload via the Skills API
3. **Use it**: Tell Claude "Use the paper-to-code skill to implement [paper name]"

### For Other AI Agents

1. Copy the `SKILL.md` file to your project
2. Load it into your agent's context
3. Follow the workflow: PDF → Markdown → Algorithm → Code → Test

## Workflow Overview

```
PDF → Markdown → Algorithm Extraction → Implementation → Testing → Packaging
```

## What's Included

- **SKILL.md** - Main skill instructions with complete workflow
- **references/** - Deep-dive guides:
  - `paper-analysis.md` - How to extract algorithms from papers
  - `implementation-patterns.md` - Code patterns and best practices
  - `packaging-checklist.md` - Complete packaging guide

## Example Projects

This skill was validated on:
- **Nested Learning (NeurIPS 2025)**: HOPE architecture, CMS, DGD optimizer
  - Successfully trained on A100 via Colab
  - All code correct from first generation
  - Repository: [nested-learning](https://github.com/aryateja2106/nested-learning)

## Key Features

- **Systematic approach**: Step-by-step workflow from paper to code
- **UV-first**: Modern Python packaging with `uv`
- **Production-ready**: Includes tests, demos, documentation
- **Device-agnostic**: Works on CPU, CUDA, MPS
- **Best practices**: Security, testing, documentation built-in

## Questions?

This skill is part of the LeCoder project. For questions or improvements, see the main repository or open an issue.


