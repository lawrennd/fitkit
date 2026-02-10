---
author: "Neil D. Lawrence (via AI Assistant)"
created: "2026-02-10"
id: "000A"
last_updated: "2026-02-10"
status: "Proposed"
compressed: false
related_requirements: []
related_cips: ["0008", "0009"]
tags:
- cip
- validation
- r-package
- method-of-reflections
title: "Investigate R economiccomplexity Package Reflections Behavior"
---

# CIP-000A: Investigate R economiccomplexity Package Reflections Behavior

> **Note**: This CIP proposes systematic investigation of unexpected behavior in R's economiccomplexity package.

## Status

- [x] Proposed - Initial investigation plan documented
- [ ] Accepted - Approved, ready to start work
- [ ] In Progress - Actively investigating
- [ ] Implemented - Investigation complete
- [ ] Closed - Findings documented and validated
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

During CIP-0008 and CIP-0009 validation, we observed unexpected behavior in R's `economiccomplexity` package `method="reflections"`:

- **R reflections vs Fitness-Complexity**: 7% correlation (unexpectedly low)
- **R eigenvalues vs Fitness-Complexity**: 49% correlation (reasonable)
- **Python reflections vs Fitness-Complexity**: 49% correlation (matches R eigenvalues)

This CIP proposes **systematic investigation** to understand whether this represents:
1. A bug in R's implementation
2. A different algorithmic choice we don't yet understand
3. A numerical precision issue
4. An undocumented feature or edge case handling

**We should not declare it "broken" without proper investigation.**

## Motivation

### Why This Matters

1. **R package is reference implementation**: Used by researchers worldwide
2. **Scientific integrity**: Need to understand discrepancies before making claims
3. **Potential bug discovery**: If there is a bug, we should report it properly
4. **Learning opportunity**: Understanding the difference may reveal important algorithm details

### Current Unknowns

From CIP-0009 analysis:
- ✓ Degenerate eigenvalues → explains modular matrix failure (NaN)
- ✓ Alternating normalization → explains 97% vs 100% correlation
- ⚠️ **Unknown**: Why does R reflections give 7% correlation with F-C on nested matrix (good eigengap)?
- ⚠️ **Unknown**: Is this intentional behavior or a bug?

## Detailed Description

### Investigation Plan

#### Phase 1: Source Code Analysis

1. **Read R implementation**:
   - Clone: https://github.com/pachadotdev/economiccomplexity
   - Examine `balassa_index()` and `complexity_measures()` functions
   - Check if reflections implementation differs from H&H (2009) specification
   - Look for any undocumented algorithmic choices

2. **Compare with Python**:
   - Line-by-line comparison of iteration logic
   - Check for differences in:
     - Initialization
     - Normalization scheme
     - Convergence criteria
     - Filtering/masking

#### Phase 2: Controlled Experiments

1. **Test on synthetic matrices**:
   - Perfect nested (known eigengap)
   - Modular (zero eigengap)
   - Random sparse
   - Document R vs Python behavior on each

2. **Trace intermediate iterations**:
   - Export k_c values at each iteration from R
   - Compare with Python iteration-by-iteration
   - Identify where they diverge

3. **Convergence analysis**:
   - Check if R reflections is actually converging
   - Measure convergence rate
   - Test different `max_iter` values

#### Phase 3: Numerical Precision Testing

1. **Test with different data types**:
   - R's default floating point
   - Higher precision if available
   - Check for accumulation of rounding errors

2. **Test edge cases**:
   - Very sparse matrices
   - Very dense matrices
   - Matrices with extreme values

#### Phase 4: Community Engagement

1. **Search existing issues**:
   - GitHub issues on economiccomplexity repo
   - R-help mailing list archives
   - Stack Overflow

2. **If bug found, report properly**:
   - Create minimal reproducible example
   - Document expected vs observed behavior
   - Provide diagnostic information
   - Open GitHub issue with constructive tone

3. **If not a bug, document behavior**:
   - Update our documentation
   - Explain the difference
   - Provide guidance on when to use each method

## Implementation Plan

### Step 1: Setup (1-2 hours)
- [ ] Clone R economiccomplexity repository
- [ ] Set up R development environment for source inspection
- [ ] Create test harness for side-by-side comparison

### Step 2: Source Analysis (2-4 hours)
- [ ] Read and annotate R implementation
- [ ] Document algorithmic differences found
- [ ] Create comparison matrix of implementation choices

### Step 3: Controlled Testing (4-6 hours)
- [ ] Run synthetic matrix tests
- [ ] Trace iteration-by-iteration behavior
- [ ] Document divergence points

### Step 4: Analysis and Reporting (2-3 hours)
- [ ] Compile findings
- [ ] Determine if bug or feature
- [ ] Update documentation in CIP-0008, CIP-0009
- [ ] Report bug to maintainers if appropriate

**Total estimated effort**: 10-15 hours

## Success Criteria

1. **Understanding achieved**:
   - Can explain why R reflections behaves differently
   - Can predict when it will diverge from eigenvalues

2. **Documentation updated**:
   - CIP-0009 reflects accurate understanding
   - Guidance provided for users

3. **If bug found**:
   - Reported to maintainers with reproducible example
   - Workaround documented
   - Python implementation validated as correct alternative

4. **If not a bug**:
   - Algorithmic differences documented
   - Use case guidance provided
   - Confusion resolved for future researchers

## Testing Strategy

All findings will be validated by:
1. Reproducible test cases
2. Multiple synthetic matrices
3. Comparison with original H&H (2009) method description
4. Cross-validation with Fitness-Complexity as ground truth

## Backward Compatibility

Not applicable - this is an investigation CIP.

## Open Questions

1. Should we contact package maintainers preemptively?
2. Should we test other R ECI/PCI packages for comparison?
3. Do we need access to R's internal debugging tools?

## References

1. **CIP-0008**: R Package Validation - Initial discovery of discrepancy
2. **CIP-0009**: Method of Reflections implementation - Python baseline
3. **R package**: https://github.com/pachadotdev/economiccomplexity
4. **Kemp-Benedict (2014)**: Mathematical analysis of Method of Reflections
5. **Hidalgo & Hausmann (2009)**: Original algorithm specification

## Implementation Status

- [ ] Phase 1: Source code analysis
- [ ] Phase 2: Controlled experiments
- [ ] Phase 3: Numerical precision testing
- [ ] Phase 4: Community engagement and reporting

## Notes

**Guiding Principles:**
- Assume good faith in existing implementations
- Document observations without premature conclusions
- Be thorough and systematic
- Report findings constructively
- Update our own code/docs regardless of outcome
