# SOTA Memetic Algorithm for Large-Scale 0/1 Knapsack

This repository implements a state-of-the-art, multi-stage memetic algorithm (knapsack_ma.cpp) designed to solve massive-scale 0/1 Knapsack instances (e.g., N ≈ 10^9). The central philosophy is: reduce first, then solve — variable fixing via Lagrangian Relaxation produces a small, tractable Core; a sparse, high-performance Memetic Algorithm then solves the Core.

File: `knapsack_ma.cpp`

## Architectural overview
- Two-stage hybrid: Reduction (full N) → Sparse Memetic Algorithm (Core N').
- Stage 1: Problem Reduction (Lagrangian Relaxation + Variable Fixing).
- Stage 2: Sparse MA solver operating on N' with sparse chromosomes, island model, and memetic operators.

## Stage 1 — Problem reduction (preprocessing)
 Solve the Lagrangian dual with subgradient descent on the full N-item instance to obtain a robust dual multiplier and fractional Lagrangian bound. We now track the best dual value (best_u) found across iterations and compute a fractional upper bound bestZUB. The integer upper bound is taken as ceil(bestZUB) to avoid integer truncation errors that could cause incorrect variable fixing or premature optimality detection; reduced costs are then calculated with best_u.
 Variable fixing rules:
     - If Z_UB - r_i < Z_LB, fix x_i = 1.
     - If Z_UB + r_i < Z_LB, fix x_i = 0.
     - Note: comparisons apply a small epsilon when using bestZUB to avoid rounding edge-cases and reduce incorrect fixes.
- All x_i=1 items are accumulated into the global solution; the remaining un-fixed items form the Core of size N'.
 Island model: Population split across NUM_ISLANDS to maintain diversity and allow parallelism; NUM_ISLANDS and ISLAND_SIZE are adapted for core size with a lowered island threshold to avoid unnecessary parallelism overhead on smaller cores; occasional migration of best individuals between islands.
 Lagrangian Greedy Crossover: Constructive O(k log k + N' log k) union-based crossover that uses reduced costs; fills from global RCBO population. The current solver can apply a fast union-style crossover (O(k)) for speed on large cores, followed by optional RCBO filling.
 Guided mutation: CORE-biased mutation flips only ambiguous items with high probability; X1_HIGH/X0_LOW items get near-zero probability. Mutation rates are adaptively scaled using CORE size and gap ratio to improve reliability without heavy computational cost.
 Sparse Simulated Annealing (memetic local search): 1-Opt moves on the sparse list (add/remove), SA acceptance to escape local optima. Local search is adaptive and only runs often enough to aid convergence — a LOCAL_SEARCH_PROB and SA_ITERATIONS governance is used: SA is disabled for very large cores to keep the evolution fast.
 Two-phase greedy repair for feasibility: remove worst v/w until feasible, then fill best v/w from Core. Repairs are scheduled less often during evolution (REPAIR_INTERVAL) to avoid per-generation overhead; we also perform a final deterministic check and repair on the assembled global solution to guarantee feasibility before returning solutions to the caller.
 Deterministic Crowding per-island: Children compete with the most similar parent (sparse Hamming), preserving niches. We also use lightweight tournament selection and elitism for faster convergence and improved runtime.
     - T_meme = O(SA_ITERATIONS * (k + N' log k)) — SA is adaptive and often disabled for very large N' to keep runtime low.
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
