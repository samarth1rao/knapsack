
## Meet-in-the-Middle Knapsack: Optimizations and Trade-offs

This implementation is based on the classic meet-in-the-middle approach for 0-1 knapsack ([GeeksforGeeks reference](https://www.geeksforgeeks.org/dsa/meet-in-the-middle/)), with several optimizations for time and memory efficiency:

### Optimizations Applied

- **Data Type Optimization:**
	- Switched from `long long` to `int` for weights, values, and capacity, reducing memory usage and improving speed (safe for datasets where values fit in 32-bit integer).
- **Mask Storage Reduction:**
	- Only store bitmasks for the left half; for the right half, store only weight and value, reconstructing the mask for the best solution only.
- **Dominance Filtering:**
	- After sorting the right half by weight, keep only non-dominated pairs (where value increases as weight increases), reducing search and memory overhead.
- **Efficient Merging:**
	- Use binary search to merge left and right halves, finding the best fit quickly.
- **Brute-force Mask Reconstruction (Right Half):**
	- For the best right half solution, reconstruct the mask only once, saving memory at the cost of a negligible time increase for small n.

### Comparison Table: Base vs. Optimized Algorithm

| Feature                        | Base Meet-in-the-Middle | Optimized Version         | Improvement / Loss         |
|--------------------------------|-------------------------|--------------------------|----------------------------|
| Data type                      | long long               | int                      | Lower memory, faster (if safe) |
| Mask storage (right half)      | All masks stored        | Only best mask reconstructed | Major memory reduction     |
| Dominance filtering            | Yes                     | Yes                      | Same (essential)           |
| Binary search for merging      | Yes                     | Yes                      | Same (essential)           |
| Output selected indices        | All indices             | All indices              | Same (no loss)             |
| Memory usage                   | High (O(2^(n/2)))       | Lower (O(2^(n/2)), but less per subset) | Improvement              |
| Time usage                     | Fast for small n        | Faster for small n       | Slight improvement         |
| Large value support            | Yes                     | No (unless int is enough)| Loss (overflow risk if values exceed int) |

**Note:**
- These optimizations make the algorithm more practical for n ≤ 40 and moderate value ranges. For very large values, revert to `long long` to avoid overflow.

---