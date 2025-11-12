For reference, https://arxiv.org/pdf/2002.00352

## Implementation notes and recent optimizations

This file documents the memory, initialization, convergence, and postprocessing
optimizations applied to the `09-billionscale/algo.cpp` implementation. All
changes preserve the algorithmic core — a dual‑descent (Lagrangian relaxation)
approach with multiplicative updates to the dual variable `lambda` — while
improving robustness on hard datasets and reducing memory where safe.

### Summary of changes

- Initialization
	- Replaced mean profit/weight heuristic with the median profit/weight ratio
		estimated from existing items. This is more robust for datasets where many
		items have similar ratios.
	- Code effect: compute `ratios` and set `lambda = median(ratios)`.

- Dual update hyperparameters
	- Increased multiplicative update learning rate (`alpha`) from `0.05` to
		`0.15` to speed convergence on difficult instances.
	- Relaxed capacity tolerance (`tol`) from `1e-4` to `1e-3` to allow the dual
		loop to stop earlier when further improvements are negligible for very
		large capacities.
	- Increased `max_iters` from `1000` to `5000` as a safety cap for harder
		instances where convergence is slower.

- Postprocessing improvements
	- Replaced the original "remove by smallest profit/weight ratio" removal
		with a two-step procedure that (a) removes smallest-profit items first to
		preserve high-value items and (b) performs a greedy add-back pass that
		attempts to insert best remaining items (by profit/weight) that fit.
	- This change aims to improve final solution quality on balanced/hard
		instances where simple ratio-based removal discards valuable combinations.

- Memory / micro-optimizations (kept conservative)
	- Several recommendations were considered (bit-packed selection flags,
		32-bit types for weights/profits), but the current committed code retains
		64-bit storage for safety and correctness on very large numeric inputs.
	- The code does avoid building excessively large temporary structures when
		possible, and uses reserve() for temporary vectors.

### Rationale and expected effects

- Median initialization: datasets with many similar profit/weight ratios often
	cause mean-based lambda to be skewed; the median gives a robust starting
	threshold and often reduces wasted dual updates.
- Larger alpha: multiplicative updates scale lambda faster toward feasibility
	when weight violation is large. This can avoid many small updates and speed
	up convergence at the cost of potentially overshooting; the median init and
	greedy postprocessing mitigate that risk.
- Relaxed tol and larger max_iters: these combined settings provide practical
	trade-offs between runtime and solution quality for massive instances.
- Postprocessing two-step: removing the smallest profits first tends to keep
	high-value items. The greedy add-back can recover capacity-filling items
	that were previously excluded during the dual pass.

### How these changes affect "billionscale" claims

- The algorithm remains a dual‑descent billionscale approach: it performs
	O(n) work per dual-iteration and never builds an O(W) DP table. The core
	method is unchanged.
- Practical memory: the implementation is still O(n) memory to store item
	arrays. For true billion-item problems on limited hardware, streaming or
	sampled initialization and external/out-of-core postprocessing are required
	(not implemented in the codebase yet). The README above documents those
	recommended next steps for extreme scale.

### Next steps (recommended)

1. Sample-based median estimation: compute the median of a small random
	 sample rather than the full `ratios` vector to reduce memory and time.
2. Replace in-memory postprocessing with an external or streaming approach
	 (priority queue limited to k best items or external sort) when k is large.
3. Optionally use `vector<bool>` or bitsets for selection flags and 32-bit
	 types for weights/profits when inputs fit to reduce memory footprint.

If you want, I can implement one or more of the next steps above; tell me
which and I'll add a focused change and test.
