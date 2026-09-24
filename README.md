# MTT GPU Solver (8-Max ICM)
High-performance, zero-abstraction Counterfactual Regret Minimization (CFR+) solver for 8-Max MTT Texas Hold'em.
Architected specifically for NVIDIA Turing (sm_75) / Tesla T4 GPUs.

## Verified Benchmarks (Tesla T4)
- 1,755 Canonical Flop EV Dumper: 3.53 minutes (8.28 flops/sec across 2x T4)
- NN-OMP Flop Subset Optimizer: 0.68 seconds (100 flops, R^2 = 0.999984)
- Preflop DCFR Anchor Generator: 45.4 ms (300 iterations across 100 subset flops)
- Honest Multi-Street HU Solver: 3.47 seconds (49 Turn Matrices, 289 MB VRAM)
- Multiway 4-Way Engine: 120 ms (1,353 nodes, 2 bets + 1 raise, 200 iterations)
