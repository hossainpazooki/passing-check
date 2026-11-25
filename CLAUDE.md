@AGENTS.md

# CLAUDE.md: how the operator drives Claude Code in this repository

`AGENTS.md` is the contract for any agent; this file restates nothing from it. Program state is never
written here; `AGENTS.md` says where it lives.

## Repo override of the global working agreement
This is a paper's record, not a software project. Work lands on `main`; a branch exists only to put a
change in front of a reviewer. No process beyond what the next step calls for. This applies the global
rule that a repo file may override the global one; say so when you apply it.

## Git mechanics for Claude
- Claude writes no git history here: no checkout, pull, commit, push, rm, branch or rebase.
- Commit blocks go to the operator in Git Bash form, one commit per concern, conventional subjects, no
  attribution trailers.
- Before any block, read `git status` and `git log -1` in the same call the block is written from.
