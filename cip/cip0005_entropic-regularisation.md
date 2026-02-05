---
author: "Neil Lawrence"
created: "2026-02-05"
id: "0005"
last_updated: "2026-02-05"
status: "Proposed"
compressed: false
related_requirements: []
related_cips: []
tags:
- cip
- algorithms
- numerical-stability
- sinkhorn
- optimal-transport
title: "Optional Entropic Regularisation for Sinkhorn"
---

# CIP-0005: Optional Entropic Regularisation for Sinkhorn

> **Note**: CIPs describe HOW to achieve requirements (WHAT).  
> Use `related_requirements` to link to the requirements this CIP implements.

> **Compression Metadata**: The `compressed` field tracks whether this CIP's key decisions have been compressed into formal documentation (README, Sphinx, architecture docs). Set to `false` by default. After closing a CIP and updating formal documentation with its essential outcomes, set `compressed: true`. This enables `whats-next` to prompt for documentation compression.

## Status

- [x] Proposed - Initial idea documented
- [ ] Accepted - Approved, ready to start work
- [ ] In Progress - Actively being implemented
- [ ] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented (add reason, use superseded_by if replaced)
- [ ] Deferred - Postponed (use blocked_by field to indicate blocker)

## Summary

Add optional entropic regularisation parameter to the Sinkhorn/IPF implementation to improve numerical stability and provide explicit control over the entropy-regularised optimal transport objective.

**Context**: The current implementation is theoretically correct (unregularised IPF = maximum entropy on support), but numerical issues can arise with extreme sparsity or ill-conditioned matrices. Optional regularisation would provide an explicit smoothness parameter while maintaining backward compatibility.

## Motivation

### Current Situation

The paper `economic-fitness.tex` shows that IPF/Sinkhorn is already solving a maximum-entropy problem on the support:

```
max_{w≥0} -Σ w_{cp} log(w_{cp})
subject to: marginal constraints + w_{cp}=0 if M_{cp}=0
```

This is equivalent to entropic OT with hard support constraints (cost = 0 on support, +∞ off support). The current implementation uses unregularised IPF ratio updates:
```python
u = r / (K @ v)
v = c / (K.T @ u)
```

### Why Add Regularisation?

1. **Numerical Stability**: Very sparse matrices or extreme marginals can cause:
   - Slow convergence
   - Numerical precision issues
   - Near-zero denominators requiring `eps` guards

2. **Explicit Control**: Users may want to control the **strength of entropy penalty** explicitly, rather than having it implicitly infinite (pure max-entropy).

3. **Modern OT Practice**: Most contemporary OT implementations use explicit `λ` or `ε` regularisation parameters for stability.

4. **Flexibility**: Different applications may benefit from different entropy penalties:
   - Strong penalty (large λ): Smoother, more uniform solutions
   - Weak penalty (small λ): Sharper, more concentrated solutions
   - λ → ∞: Pure max-entropy (current behaviour)

### What We're NOT Changing

- Default behaviour remains unregularised (backward compatible)
- The paper's theory is correct for the unregularised case
- Optional parameter only for users who need it

## Detailed Description

### Theoretical Background

**Entropic-regularised optimal transport** seeks:
```
min_{w∈Π(r,c)} Σ C_{cp}w_{cp} + λ Σ w_{cp}(log w_{cp} - 1)
```

For our **masked setting** with `C_{cp} = 0` on support and `C_{cp} = +∞` elsewhere:
```
min_{w} λ Σ w_{cp}(log w_{cp} - 1)
subject to: marginal constraints + support constraint
```

Equivalently (maximising entropy):
```
max_{w} -λ⁻¹ Σ w_{cp} log w_{cp}
subject to: marginal constraints + support constraint
```

**Current implementation**: Effectively λ → ∞ (pure max-entropy, no regularisation)  
**Proposed**: Allow finite λ for controlled regularisation

### Implementation Approaches

**Option A: Explicit Regularisation Parameter** (Recommended)

Add `reg_lambda` parameter (default: `None` for unregularised):
```python
def sinkhorn_masked(M_bin, r, c, n_iter=2000, tol=1e-12, eps=1e-30, reg_lambda=None):
    if reg_lambda is None:
        # Current unregularised IPF (pure max-entropy)
        u_new = r / np.maximum(K @ v, eps)
        v_new = c / np.maximum(K.T @ u_new, eps)
    else:
        # Entropic-regularised updates (work in log space for stability)
        log_K = np.log(K.toarray() + eps)  # Support mask in log-space
        log_u = np.log(r) - reg_lambda * logsumexp(log_K + log_v, axis=1)
        log_v = np.log(c) - reg_lambda * logsumexp(log_K.T + log_u, axis=1)
        u_new = np.exp(log_u)
        v_new = np.exp(log_v)
```

**Option B: Temperature/Smoothness Parameter**

Use `temperature` or `smoothness` naming (more intuitive):
- `temperature = 1.0`: Default unregularised
- `temperature < 1.0`: More regularisation (smoother)
- `temperature > 1.0`: Less regularisation (sharper)

**Option C: Full Log-Space Implementation**

Always work in log-space for numerical stability, with optional regularisation:
```python
# Always use log-space (more stable)
log_u = log(r) - logsumexp_masked(log(K) + log(v))
log_v = log(c) - logsumexp_masked(log(K.T) + log(u))
```

### Recommended Approach

**Option A** with these specifics:

1. **Parameter name**: `reg_lambda` (regularisation strength)
   - `None` (default): Unregularised IPF (current behaviour, backward compatible)
   - Float > 0: Entropic regularisation strength

2. **Implementation**: Conditional logic (keep simple ratio updates as default)

3. **Log-space**: Only when `reg_lambda` is specified

4. **API**:
   ```python
   # Unregularised (default, current behaviour)
   W = SinkhornScaler().fit_transform(M, row_marginals=r, col_marginals=c)
   
   # Regularised (new option)
   W = SinkhornScaler(reg_lambda=0.1).fit_transform(M, row_marginals=r, col_marginals=c)
   ```

### Design Decisions

**Q: Why not always use log-space?**  
A: The simple ratio updates are:
- Faster (no log/exp operations)
- More intuitive (direct IPF formulation)
- Match the paper's presentation
- Work well for most cases with our auto-correction

**Q: What about numerical stability?**  
A: Our recent auto-correction fix (setting marginals to zero for isolated nodes) addresses the main stability issue. Regularisation is for users who need **explicit entropy control** for modelling reasons, not just stability.

**Q: Relationship to paper's theory?**  
A: Paper describes unregularised case (λ → ∞). Adding finite λ is a **generalisation** that doesn't contradict the theory.

## Implementation Plan

### Phase 1: Core Regularisation (Minimal)

1. **Add `reg_lambda` parameter to `sinkhorn_masked` function**:
   - Default: `None` (unregularised, backward compatible)
   - Type: `float | None`
   - Range: > 0 when specified

2. **Implement conditional logic**:
   - If `reg_lambda is None`: Use current ratio updates
   - If `reg_lambda is not None`: Use log-space regularised updates

3. **Log-space implementation**:
   - Implement `logsumexp_masked` helper for sparse matrices
   - Handle support constraint in log-space (log(0) = -inf)
   - Convert back to u, v via exp for output

4. **Update `SinkhornScaler` class**:
   - Add `reg_lambda` to `__init__` parameters
   - Pass through to `sinkhorn_masked`

### Phase 2: Testing & Validation

1. **Unit tests**:
   - Test unregularised matches current behavior (backward compat)
   - Test regularised converges and is stable
   - Test λ → ∞ approximates unregularised
   - Test extreme sparsity with regularisation

2. **Convergence diagnostics**:
   - Compare convergence speed (regularised vs unregularised)
   - Check solution quality (KL divergence from unregularised)

3. **Numerical stability tests**:
   - Test on ill-conditioned matrices
   - Test with extreme marginals
   - Test with very sparse supports

### Phase 3: Documentation & Examples

1. **Update docstrings**:
   - Document `reg_lambda` parameter
   - Explain regularisation effect
   - Provide usage examples

2. **Add example notebook or section**:
   - Compare regularised vs unregularised
   - Show when regularisation helps
   - Visualise smoothing effect

3. **Update paper connections**:
   - Document that paper uses λ → ∞ (unregularised)
   - Explain generalisation to finite λ
   - Reference OT literature (Cuturi, Peyré)

## Backward Compatibility

✅ **Fully backward compatible**:
- Default `reg_lambda=None` preserves current behaviour
- All existing code continues to work unchanged
- Only users who explicitly request regularisation get it

No breaking changes to:
- Function signatures (optional parameter)
- Return types (same u, v, W, history)
- Convergence behaviour (without regularisation)

## Testing Strategy

### Unit Tests

```python
def test_sinkhorn_regularisation_backward_compat():
    """Test that unregularised mode matches current behaviour."""
    # Compare with/without explicit reg_lambda=None
    
def test_sinkhorn_regularised_convergence():
    """Test that regularised Sinkhorn converges."""
    # Test with various λ values
    
def test_sinkhorn_regularisation_smoothness():
    """Test that higher λ produces smoother solutions."""
    # Compare entropy of solutions for different λ
    
def test_sinkhorn_regularisation_limit():
    """Test that very large λ approximates unregularised."""
    # λ=1e6 should ≈ λ=None
```

### Integration Tests

```python
def test_regularised_vs_unregularised_on_sparse():
    """Compare performance on very sparse matrix."""
    
def test_regularisation_with_isolated_nodes():
    """Test that regularisation helps with isolated nodes."""
    # Should converge faster/more stably than auto-correction alone
```

### Benchmark Tests

- Convergence speed comparison
- Solution quality (KL divergence from unregularised)
- Numerical stability on edge cases

## Related Requirements

No existing requirements directly address regularisation. This CIP is motivated by:
- Numerical stability improvements (related to general code quality)
- User flexibility (aligns with user autonomy principles)
- Modern OT practice (technical excellence)

Consider creating requirement if this becomes a recurring theme.

## Implementation Status

- [ ] Add `reg_lambda` parameter to function signature
- [ ] Implement `logsumexp_masked` helper function
- [ ] Implement log-space regularised updates
- [ ] Add conditional logic (reg_lambda is None vs specified)
- [ ] Update SinkhornScaler class
- [ ] Write unit tests for regularisation
- [ ] Write integration tests
- [ ] Update docstrings and documentation
- [ ] Add example usage (notebook section or docstring)
- [ ] Benchmark performance comparison

## References

### Theoretical Foundations

- Csiszár, I. (1975). "I-divergence geometry of probability distributions and minimisation problems". Annals of Probability.
- Haberman, S.J. (1974). "The analysis of frequency data". University of Chicago Press.
- Bishop, Y., Fienberg, S., Holland, P. (1975). "Discrete Multivariate Analysis". MIT Press.

### Entropic Optimal Transport

- Cuturi, M. (2013). "Sinkhorn Distances: Lightspeed Computation of Optimal Transport". NIPS.
- Peyré, G. & Cuturi, M. (2019). "Computational Optimal Transport". Foundations and Trends in Machine Learning.

### Project Context

- Lawrence, N.D. (2024). "Conditional Likelihood Interpretation of Economic Fitness" (working paper).
  - Section: "Maximum entropy on a fixed support (dual view)"
  - Shows IPF is pure max-entropy (λ → ∞ in OT formulation)
  - This CIP generalises to finite λ for numerical/modeling flexibility

### Related Implementations

- POT (Python Optimal Transport): https://pythonot.github.io/
  - See `ot.sinkhorn()` for regularised implementation
- GeomLoss: https://www.kernel-operations.io/geomloss/
  - Provides various regularisation schedules

## Design Notes

### Naming Conventions

**Parameter name options**:
- `reg_lambda`: Mathematical (λ in OT literature) ✓ Recommended
- `epsilon`: Common in OT code, but confusing with `eps` (numerical guard)
- `reg_strength`: More intuitive, less standard
- `temperature`: Common in statistical mechanics, but inverted semantics

**Recommendation**: Use `reg_lambda` for consistency with OT literature.

### Implementation Complexity

**Simple**: Add parameter + conditional logic → ~50 lines of code  
**Medium**: Add + log-space helpers + validation → ~150 lines  
**Complex**: Always use log-space + optimisation → ~300 lines (refactor)

**Recommendation**: Start with **Simple** approach (conditional logic). Can refactor to log-space later if needed.

### Performance Considerations

- Log-space operations are slower (log/exp computations)
- Only pay cost when regularisation is requested
- For most use cases, unregularised is sufficient and faster
- Regularisation mainly for:
  - Very ill-conditioned problems
  - Explicit modelling of entropy penalty
  - Comparison with regularised OT methods

### Open Questions

1. **Default value**: Should `reg_lambda=None` or `reg_lambda=1.0`?
   - `None`: Unregularised (current behaviour, backward compat) ✓
   - `1.0`: Regularised by default (breaking change)

2. **Log-space always?**: Should we always use log-space for stability?
   - No: Adds overhead for common case (simple matrices work fine)
   - Maybe: If we find stability issues persist

3. **Relationship to `eps` parameter**: Current `eps=1e-30` is numerical guard
   - Keep separate from regularisation
   - `eps`: numerical stability (technical)
   - `reg_lambda`: modeling choice (semantic)

## Notes

### Why This Matters

From testing ECI implementation (see `tests/README_ECI_VERIFICATION.md`):
- Discovered importance of robust handling of isolated nodes
- Current auto-correction fix works well for feasibility
- Regularisation adds **modeling flexibility** beyond just fixing errors

### Connection to Fitness-Complexity

The paper shows Fitness-Complexity is equivalent to IPF on the support with chosen marginals. Adding regularisation:
- Doesn't break this equivalence
- Provides generalisation: FC ⇔ IPF(λ→∞)
- Allows exploring smoothness trade-offs

### User Feedback Integration

This CIP addresses user question: "Can't we regularise the sinkhorn or run in log space to help?"
- Short-term fix: Auto-correction (already done)
- Long-term enhancement: Optional regularisation (this CIP)
