# Dynamic Programming Solution for 0/1 Knapsack Problem

## Theoretical Analysis

### An Explanation

#### Problem Statement

The **0/1 Knapsack Problem** is a classic optimization problem where we are given:
- A set of `n` items, each with a weight `w_i` and value `v_i`
- A knapsack with maximum capacity `W`

The goal is to select a subset of items to maximize the total value while ensuring the total weight does not exceed the capacity. Each item can either be included (1) or excluded (0), hence the name "0/1 Knapsack."

**Mathematically**, we want to find:

$$
\max \sum_{i=1}^{n} v_i \cdot x_i
$$

subject to:

$$
\sum_{i=1}^{n} w_i \cdot x_i \leq W
$$

where $x_i \in \{0, 1\}$ for all $i \in \{1, 2, ..., n\}$.

#### The Dynamic Programming Approach

The algorithm uses **bottom-up dynamic programming (tabulation)** to solve this problem optimally. The key insight is to break down the problem into overlapping subproblems and build solutions incrementally.

We create a table where each cell `dp[i][w]` represents "the maximum value achievable using the first `i` items with a knapsack capacity of `w`." We start with no items and build up, considering each item one by one. For each item, at each possible capacity, we make a decision: should we include this item or not? We pick whichever choice gives us more value.

**Mathematically**, the recurrence relation is:

$$
dp[i][w] = \begin{cases}
0 & \text{if } i = 0 \text{ or } w = 0 \\
dp[i-1][w] & \text{if } w_i > w \\
\max(dp[i-1][w], \, v_i + dp[i-1][w - w_i]) & \text{if } w_i \leq w
\end{cases}
$$

Where:
- `dp[i-1][w]` represents the value if we **exclude** item `i`
- `v_i + dp[i-1][w - w_i]` represents the value if we **include** item `i`

**Backtracking:** After filling the table, we trace back from `dp[n][W]` to determine which items were selected by comparing values in adjacent cells.

---

### Time Complexity

#### Analysis

The algorithm consists of two main phases:

1. **Table Filling Phase:**
   - We have two nested loops: one iterating over `n` items, another over `W+1` capacity values
   - Each cell computation involves constant-time operations (comparison, addition, max)
   - Total operations: $O(n \times W)$

2. **Backtracking Phase:**
   - We iterate backward through at most `n` items
   - Each iteration performs constant-time operations
   - Total operations: $O(n)$

**Overall Time Complexity:**

- **Best Case:** $\Theta(n \times W)$
- **Average Case:** $\Theta(n \times W)$
- **Worst Case:** $\Theta(n \times W)$

All cases have the same complexity because the algorithm must fill the entire DP table regardless of input values. The number of operations is deterministic and depends only on `n` and `W`.

**Important Note:** This is a **pseudo-polynomial time** algorithm because the complexity depends on the *value* of `W` (the capacity), not just the *size* of the input. If `W` is exponentially large relative to `n`, the algorithm becomes impractical.

---

### Space Complexity

#### Analysis

The algorithm uses several data structures:

1. **DP Table:** `vector<vector<int64>> dp(n+1, vector<int64>(W+1, 0))`
   - Dimensions: $(n+1) \times (W+1)$
   - Space: $O(n \times W)$

2. **Input Vectors:** `weights` and `values`
   - Space: $O(n)$ each, total $O(n)$

3. **Output Vector:** `selectedItems`
   - Worst case: all items selected
   - Space: $O(n)$

4. **Other Variables:** Constant space for loop counters, temporary variables
   - Space: $O(1)$

**Total Space Complexity:**

- **Auxiliary Space:** $O(n \times W)$ (dominated by the DP table)
- **Total Space:** $O(n \times W)$

**Space Optimization Note:** This implementation uses a 2D table for clarity and to support backtracking. It's possible to optimize space to $O(W)$ using a 1D array if we only need the maximum value (without tracking selected items), though backtracking would require additional techniques.

---

### Correctness

#### Proof of Correctness

We prove correctness using **mathematical induction** on the number of items considered.

**Base Case:** When $i = 0$ (no items) or $w = 0$ (zero capacity):
- $dp[0][w] = 0$ for all $w$, which is correct (no items means no value)
- $dp[i][0] = 0$ for all $i$, which is correct (zero capacity means nothing can be taken)

**Inductive Hypothesis:** Assume that for all $k < i$, the value $dp[k][w]$ correctly represents the maximum value achievable using the first $k$ items with capacity $w$.

**Inductive Step:** We prove that $dp[i][w]$ is correct.

For item $i$ with weight $w_i$ and value $v_i$:

1. **If $w_i > w$:** The item cannot fit, so we cannot include it. The only option is to exclude it:
   $$dp[i][w] = dp[i-1][w]$$
   By the inductive hypothesis, $dp[i-1][w]$ is correct, so $dp[i][w]$ is correct.

2. **If $w_i \leq w$:** We have two choices:
   - **Exclude item $i$:** Value is $dp[i-1][w]$ (correct by hypothesis)
   - **Include item $i$:** We get value $v_i$, plus the best we can do with remaining capacity $w - w_i$ using items $1$ to $i-1$, which is $dp[i-1][w - w_i]$ (correct by hypothesis)
   
   Taking the maximum of these two options gives us the optimal value:
   $$dp[i][w] = \max(dp[i-1][w], \, v_i + dp[i-1][w - w_i])$$

**Conclusion:** By induction, $dp[n][W]$ correctly represents the maximum value achievable with all $n$ items and capacity $W$.

**Backtracking Correctness:** The backtracking phase reconstructs the solution by checking whether $dp[i][w] \neq dp[i-1][w]$. If they differ, item $i$ must have been included (since including it gave a better value). This correctly identifies all selected items.

---

### Model of Computation/Assumptions

#### Computational Model

- **RAM Model (Random Access Machine):** The algorithm assumes a standard computational model where:
  - Array/vector access takes $O(1)$ time
  - Arithmetic operations (addition, subtraction, comparison) take $O(1)$ time
  - Memory access is uniform cost

#### Assumptions

1. **Input Format:**
   - `n` items with non-negative integer weights and values
   - Capacity `W` is a non-negative integer
   - All values fit within 64-bit signed integers (`int64`)

2. **Data Types:**
   - Uses `long long` (int64) to prevent overflow for large values
   - Assumes arithmetic operations on 64-bit integers are constant time

3. **Memory:**
   - Sufficient memory is available to allocate the $(n+1) \times (W+1)$ DP table
   - For very large `W`, this could require gigabytes of memory

4. **Optimality:**
   - The algorithm guarantees an **optimal solution** (maximum value)
   - Unlike greedy or heuristic approaches, this is an exact algorithm

5. **Input Validity:**
   - No validation is performed on input data
   - Assumes well-formed input (correct counts, non-negative values)

---

### Case Analysis (Best/Average/Worst)

#### Best Case

**Input Characteristics:** Any valid input with `n` items and capacity `W`.

**Time Complexity:** $\Theta(n \times W)$

**Explanation:** The dynamic programming algorithm always fills the entire $(n+1) \times (W+1)$ table, regardless of the specific values of weights and items. There is no "early termination" condition. Even if all items are too heavy (all $w_i > W$), the algorithm still iterates through all cells, though many will simply copy the value from the previous row.

**Space Complexity:** $O(n \times W)$ (same as average/worst case)

#### Average Case

**Input Characteristics:** Random weights and values with no special structure.

**Time Complexity:** $\Theta(n \times W)$

**Explanation:** The average case is identical to the best and worst cases. The algorithm's behavior is deterministic and independent of the distribution of weights and values. Each cell requires the same constant-time operations regardless of input values.

**Space Complexity:** $O(n \times W)$ (same as best/worst case)

#### Worst Case

**Input Characteristics:** Any valid input with `n` items and capacity `W`.

**Time Complexity:** $\Theta(n \times W)$

**Explanation:** Like the best and average cases, the worst case also requires filling the entire DP table. There are no inputs that cause the algorithm to perform additional work beyond the standard table filling and backtracking.

**Space Complexity:** $O(n \times W)$ (same as best/average case)

**Note on "Worst" in Practice:** While the asymptotic complexity is the same across all cases, certain inputs might be "worse" in practice:
- **Large Capacity:** If $W$ is very large (e.g., billions), the algorithm becomes impractical due to memory and time requirements
- **All Items Selected:** If all items fit in the knapsack, backtracking visits all `n` items, but this is still $O(n)$

---

### Summary

The **Dynamic Programming solution** for the 0/1 Knapsack problem is:

| Aspect | Complexity |
|--------|------------|
| **Time Complexity** | $\Theta(n \times W)$ (pseudo-polynomial) |
| **Space Complexity** | $O(n \times W)$ |
| **Correctness** | Proven optimal via induction |
| **Cases** | Best = Average = Worst |

**Strengths:**
- Guarantees optimal solution
- Efficient for moderate values of `W`
- Handles arbitrary weight and value combinations

**Weaknesses:**
- Pseudo-polynomial (depends on magnitude of `W`, not just input size)
- High memory usage for large capacities
- Impractical for very large `W` (e.g., $W > 10^7$)

### For reference:
https://www.w3schools.com/dsa/dsa_ref_knapsack.php