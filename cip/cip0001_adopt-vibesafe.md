---
id: "0001"
title: "Adopt VibeSafe project governance and tooling"
status: "Accepted"
created: "2026-02-02"
last_updated: "2026-02-02"
author: "Neil Lawrennd"
related_requirements:
  - "0001"
tags:
  - "vibesafe"
  - "governance"
---

# CIP-0001: Adopt VibeSafe project governance and tooling

## Status

- [x] Proposed
- [x] Accepted
- [ ] Implemented
- [ ] Closed

## Description

Adopt VibeSafe conventions in this repository to provide lightweight governance (WHY/WHAT/HOW/DO artifacts), automated validation, and the `whats-next` status workflow.

## Motivation

We want structural/tooling changes to be traceable and reviewable. VibeSafe provides templates and a validator that can detect drift (implementation changes without matching governance updates).

## Implementation

- Install VibeSafe minimal system files (templates, validator, and `whats-next` wrapper).
- Use `requirements/` for WHAT, `cip/` for HOW, and `backlog/` for DO.
- Keep governance artifacts updated when making structural/tooling changes.

## Implementation Status

- [x] Install VibeSafe and `whats-next` wrapper in repo root
- [x] Add an initial requirement capturing governance/traceability expectations
- [x] Add an initial backlog task for follow-on adoption work
- [ ] Ensure `whats-next` and validation scripts are part of normal workflow

## References

- **Related Requirements**: REQ-0001

