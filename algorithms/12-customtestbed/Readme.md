# SOTA Memetic Algorithm for Large-Scale 0/1 Knapsack

This repository implements a state-of-the-art, multi-stage memetic algorithm (knapsack_ma.cpp) designed to solve massive-scale 0/1 Knapsack instances (e.g., N ≈ 10^9). The central philosophy is: reduce first, then solve — variable fixing via Lagrangian Relaxation produces a small, tractable Core; a sparse, high-performance Memetic Algorithm then solves the Core.

File: `knapsack_ma.cpp`

## Architectural overview
- Two-stage hybrid: Reduction (full N) → Sparse Memetic Algorithm (Core N').
- Stage 1: Problem Reduction (Lagrangian Relaxation + Variable Fixing).
- Stage 2: Sparse MA solver operating on N' with sparse chromosomes, island model, and memetic operators.

## Stage 1 — Problem reduction (preprocessing)
- Solve the Lagrangian dual with subgradient descent on the full N-item instance to obtain u*, Z_UB, Z_LB and reduced costs r_i = v_i - u*w_i.
- Variable fixing rules:
    - If Z_UB - r_i < Z_LB, fix x_i = 1.
    - If Z_UB + r_i < Z_LB, fix x_i = 0.
- All x_i=1 items are accumulated into the global solution; the remaining un-fixed items form the Core of size N'.

## Stage 2 — Sparse Memetic Algorithm (solver)
- Sparse chromosome: Individuals store only selected indices using std::vector<size_t> selectedItemIndices (O(k) memory).
- Island model: Population split across NUM_ISLANDS to maintain diversity and allow parallelism; occasional migration of best individuals between islands.
- Deterministic Crowding per-island: Children compete with the most similar parent (sparse Hamming), preserving niches.
- Heuristic seeding: Multi-seeding using v/w, RCBO (reduced cost), max value and min weight orders, plus mutated clones.
- Lagrangian Greedy Crossover: Constructive O(k log k + N' log k) union-based crossover that uses reduced costs; fills from global RCBO population.
- Guided mutation: CORE-biased mutation flips only ambiguous items with high probability; X1_HIGH/X0_LOW items get near-zero probability.
- Sparse Simulated Annealing (memetic local search): 1-Opt moves on the sparse list (add/remove), SA acceptance to escape local optima.
- Two-phase greedy repair for feasibility: remove worst v/w until feasible, then fill best v/w from Core.

## Operators & complexity (sparse)
- calculateMetrics, mutate, crossover, and localSearch operate on sets (O(k) or O(k log k)).
- Repair scans N' for fill candidates (O(N' log k)).
- Lagrangian Greedy Crossover sorts union pool O(k log k) then optionally uses global RCBO O(N' log k).

## Comparison to base GA
Feature | Standard GA | SOTA MA (knapsack_ma.cpp)
---|---:|:---
Problem Scope | Full N | Reduce to N' then solve N'
Chromosome | Dense std::vector<bool> (O(N)) | Sparse std::vector<size_t> (O(k))
Search model | Single-population GA | Island MA + migration
Preprocessing | None | Lagrangian reduction + variable fixing
Seeding | Random | Heuristic multi-seeding (v/w, RCBO, etc.)
Crossover | Single-point | Lagrangian Greedy (sparse constructive)
Mutation | Uniform random | CORE-focused guided mutation
Local search | None | Simulated Annealing per child (sparse)
Replacement | Generational | Deterministic Crowding (per island)
Repair | Simple random | Two-phase greedy (sparse)

## Complexity analysis
- Preprocessing: O(T*N + N log N) — one-time cost to analyze full N items (T Lagrangian iters + sorting).
- Evolution per gen: O(G * P_total * (T_cross + T_meme + T_repair)) with:
    - T_cross = O(k log k + N' log k)
    - T_meme = O(SA_ITERATIONS * (k + N' log k))
    - T_repair = O(N' log k)
- Space: O(N + N' + P_total * k) — reading full input, storing Core metadata, and sparse population storage.
- Designed so the heavy cost on N is a fixed preprocessing step; evolutionary cost depends on N' and k, enabling N ≈ 10^9 scale.

This design preserves solution quality and guarantees scalability by focusing expensive evolutionary work only on the compressed Core.
