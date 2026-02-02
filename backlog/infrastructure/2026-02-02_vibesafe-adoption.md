---
id: "2026-02-02_vibesafe-adoption"
title: "Adopt VibeSafe governance workflow in this repo"
status: "Proposed"
priority: "Medium"
created: "2026-02-02"
last_updated: "2026-02-02"
category: "infrastructure"
related_cips:
  - "0001"
owner: "Neil Lawrennd"
dependencies: []
tags:
  - "vibesafe"
  - "governance"
---

# Task: Adopt VibeSafe governance workflow in this repo

## Description

Follow through on the initial VibeSafe installation by integrating the governance workflow into day-to-day work (requirements/CIPs/backlog + validation).

## Acceptance Criteria

- [ ] `./whats-next` is used to review project status and next steps.
- [ ] `./scripts/validate_vibesafe_structure.py` runs clean for typical changes (or is intentionally gated with `--strict` in CI later).
- [ ] A first tenet is authored (WHY), or an explicit note exists explaining why tenets are deferred.

## Implementation Notes

- Keep linking bottom-up: Backlog → CIPs → Requirements → Tenets.
- If a task truly doesn’t warrant a CIP, use the backlog “exception path” (related_cips: [] + `no_cip_reason`), but prefer a CIP for structural/tooling changes.

## Progress Updates

### 2026-02-02

Task created as part of addressing VibeSafe validation warnings after installation.

