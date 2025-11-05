# Random Permutation


### Reference Article
For reference, section 2 of https://arxiv.org/pdf/2308.11307  


### Randomized 0–1 Knapsack in ˜O(n³ᐟ² wₘₐₓ) Time

This project implements a randomized approximation algorithm for the classical 0–1 knapsack problem.  
It achieves ˜O(n³ᐟ² wₘₐₓ) runtime, improving upon the standard O(nW) dynamic programming approach when W is large but the maximum item weight (wₘₐₓ) is moderate.

---

### Algorithm Intuition

1. **Standard DP Recap**  
   The classical DP computes optimal values for each sub-capacity up to W, leading to O(nW) time complexity.

2. **Randomized Reduction**  
   This algorithm samples item weights, partitions them into groups, and uses probabilistic rounding to approximate achievable weights, significantly reducing the effective state space.

3. **Runtime**  
   The resulting expected runtime is ˜O(n³ᐟ² wₘₐₓ), maintaining good accuracy while improving scalability for large instances.

---

### Applications
Useful for large-scale resource allocation or subset selection problems where full DP becomes computationally expensive.
