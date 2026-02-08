---
id: "2026-02-08_create-community-module-structure"
title: "Create fitkit/community/ module structure"
status: "Proposed"
priority: "High"
created: "2026-02-08"
last_updated: "2026-02-08"
category: "infrastructure"
related_cips: ["0006"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- community-detection
- module-structure
---

# Task: Create fitkit/community/ Module Structure

> **Note**: Backlog tasks are DOING the work defined in CIPs (HOW).  
> Use `related_cips` to link to CIPs. Don't link directly to requirements (bottom-up pattern).

## Description

Create the foundational module structure for community detection functionality in fitkit. This establishes the package architecture before implementing the core algorithms.

## Acceptance Criteria

- [ ] Create `fitkit/community/` directory
- [ ] Create `fitkit/community/__init__.py` with proper exports
- [ ] Create placeholder files for submodules:
  - `detection.py` (CommunityDetector class)
  - `analysis.py` (within_community_analysis utilities)
  - `validation.py` (statistical validation methods)
- [ ] Add `from fitkit.community import CommunityDetector` to `fitkit/__init__.py`
- [ ] Add basic module docstring explaining purpose
- [ ] Verify imports work: `from fitkit.community import CommunityDetector` doesn't crash

## Implementation Notes

Module organization:
```
fitkit/
├── community/
│   ├── __init__.py          # Exports CommunityDetector, analysis functions
│   ├── detection.py         # CommunityDetector class
│   ├── analysis.py          # within_community_analysis()
│   └── validation.py        # permutation tests, Cheeger, effective rank
```

This follows the pattern established for `fitkit/algorithms/` in CIP-0004.

## Related

- CIP: 0006
- Related CIP: 0004 (sklearn-style estimators pattern)

## Progress Updates

### 2026-02-08

Task created. Waiting for CIP-0006 acceptance.
