# fitkit (work in progress)

This repository contains **in-development software accompanying the paper** *Conditional Likelihood Interpretation of Economic Fitness*.

## Scientific contribution

The paper reframes the **Fitness–Complexity** algorithm as a statistical / information-theoretic construction on a bipartite support graph:

- **RCA thresholding induces structural zeros**: the observed binary incidence matrix \(M\) is treated as a *support constraint* (edges allowed / forbidden).
- **IPF = Sinkhorn matrix scaling on the support**: fitting a log-linear independence model *conditional on that support* leads to **iterative proportional fitting (IPF)**, which is the same procedure as **Sinkhorn/Knopp scaling** (and can be viewed as a masked entropic-OT feasibility/maximum-entropy problem).
- **Fitness and Complexity are the dual scaling parameters**: the familiar fixed point updates are a reparameterisation of the diagonal scaling solution $w_{cp}=M_{cp} A_c B_p$ (with $A_c \equiv 1/F_c$, $B_p \equiv Q_p$), so $(\log F,\log Q)$ correspond to the Lagrange multipliers / dual potentials enforcing the marginal constraints.

This perspective explains key empirical properties (scale freedom, weakest-link effects, bottlenecks) as consequences of feasibility + maximum-entropy completion on a sparse support.

## What’s in this repo

- `wikipedia_editing_fitness_complexity.ipynb`: a worked notebook demonstrating Fitness–Complexity alongside the equivalent masked IPF/Sinkhorn viewpoint and “flow-native” visualisations.

## Status

The code and API are not yet stabilised; expect changes.

