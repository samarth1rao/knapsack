# Memoization Approach for 0/1 Knapsack Problem

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

#### The Memoization Approach

The memoization approach is a **top-down dynamic programming** technique that combines the recursive structure of brute force with intelligent caching to avoid redundant computations.

**In English:**
We start with the original problem and recursively break it down into smaller subproblems, just like brute force. However, before solving each subproblem, we check if we've already solved it before. If we have, we simply return the cached result. If not, we compute it, store the result in a memo table, and return it. This way, each unique subproblem is solved exactly once.

**Key Insight:** The brute force approach has **overlapping subproblems** - the same subproblem `(i, remaining_capacity)` is encountered multiple times through different recursive paths. Memoization eliminates this redundancy by caching results.

**Mathematically**, the recurrence relation with memoization is:

$$
M[i][w] = \begin{cases}
0 & \text{if } i = 0 \text{ or } w = 0 \\
M[i][w] & \text{if already computed (cached)} \\
M[i-1][w] & \text{if } w_i > w \text{ (doesn't fit)} \\
\max(M[i-1][w], \, v_i + M[i-1][w - w_i]) & \text{if } w_i \leq w \text{ (fits)}
\end{cases}
$$

Where $M[i][w]$ represents the memoized result for the maximum value achievable using the first `i` items with capacity `w`.

**Process Flow:**
1. Check if `memo[i][remaining_capacity]` is already computed (not `-1`)
2. If yes, return the cached value immediately (O(1) lookup)
3. If no, recursively compute the result:
   - If item doesn't fit: recurse with `(i-1, capacity)`
   - If item fits: take max of including vs excluding
4. Store the result in `memo[i][remaining_capacity]`
5. Return the computed result

**Comparison with Brute Force:**
- **Brute Force:** Explores $2^n$ nodes, solving the same subproblems repeatedly
- **Memoization:** Solves each unique subproblem once, with at most $n \times W$ unique subproblems

---

### Time Complexity

#### Analysis

The time complexity depends on the number of unique subproblems and the work done per subproblem.

1. **Number of Unique Subproblems:**
   - Each subproblem is defined by two parameters: item index `i` (range: $0$ to $n$) and capacity `w` (range: $0$ to $W$)
   - Total unique subproblems: $(n+1) \times (W+1) = O(n \times W)$

2. **Work Per Subproblem:**
   - **Cache hit:** If already computed, lookup is $O(1)$
   - **Cache miss:** Perform computation (comparison, addition, max operation, recursive calls)
   - The actual computation is $O(1)$ (constant operations)
   - Recursive calls don't add to the count because they either hit the cache or solve a new subproblem (counted separately)

3. **Total Computation:**
   - At most $n \times W$ subproblems are computed (cache misses)
   - Each computation takes $O(1)$ time
   - Total: $O(n \times W)$

4. **Reconstruction Phase:**
   - Backtracking through the memo table to find selected items
   - Visits at most $n$ items
   - Time: $O(n)$

**Overall Time Complexity:**

- **Best Case:** $O(n \times W)$
  - Even if many branches terminate early, we still need to fill reachable states in the memo table
  - The recursive structure ensures we compute necessary subproblems

- **Average Case:** $\Theta(n \times W)$
  - For typical inputs, most states in the memo table are visited
  - The algorithm explores the state space systematically

- **Worst Case:** $\Theta(n \times W)$
  - All states in the memo table are computed
  - This occurs when all items can potentially fit

**Comparison with Other Approaches:**
- **Brute Force:** $O(2^n)$ - exponential
- **Memoization:** $O(n \times W)$ - pseudo-polynomial
- **Dynamic Programming (Bottom-Up):** $O(n \times W)$ - same asymptotic complexity

**Pseudo-Polynomial Nature:** The complexity depends on the *value* of `W`, not just its *size* (number of bits). If `W` is exponentially large, the algorithm becomes impractical.

---

### Space Complexity

#### Analysis

The algorithm uses space for several components:

1. **Memoization Table:**
   - Dimensions: $(n+1) \times (W+1)$
   - Each entry stores an `int64` value
   - Space: $O(n \times W)$

2. **Recursion Call Stack:**
   - Maximum recursion depth: $O(n + W)$ in the worst case
   - In practice, depth is typically $O(n)$ as we process items sequentially
   - Each stack frame stores: item index, capacity, return address
   - Stack space: $O(n)$ typically, $O(n + W)$ worst case

3. **Input Storage:**
   - `weights` and `values` vectors: $O(n)$ each
   - Total: $O(n)$

4. **Output Storage:**
   - `selected_items` vector: $O(n)$ in worst case

**Total Space Complexity:**

- **Auxiliary Space:** $O(n \times W)$ (dominated by memo table)
- **Total Space:** $O(n \times W + n) = O(n \times W)$

**Comparison with Bottom-Up DP:**
- **Memoization:** Requires $O(n \times W)$ for memo table + $O(n)$ for recursion stack
- **Bottom-Up DP:** Requires $O(n \times W)$ for DP table (no recursion stack overhead)
- **Optimized Bottom-Up:** Can use $O(W)$ space with rolling array technique
- **Memoization Advantage:** Only allocates space for *reachable* states if using sparse data structures (e.g., hash maps), though this implementation uses a full 2D array

---

### Correctness

#### Proof of Correctness

We prove correctness using **strong induction** combined with the **memoization invariant**.

**Memoization Invariant:** Once `memo[i][w]` is computed and stored, it contains the correct maximum value for the subproblem of using the first `i` items with capacity `w`.

**Claim:** The function `knapsack_memoization(i, w)` correctly returns the maximum value achievable using the first `i` items with capacity `w`.

**Base Case:** When $i = 0$ (no items) or $w = 0$ (zero capacity):
- Returns $0$
- This is correct: no items or no capacity means zero value ✓

**Inductive Hypothesis:** Assume that for all pairs $(k, c)$ where either $k < i$ or $c < w$ (or both), if `knapsack_memoization(k, c)` is called, it returns the correct maximum value.

**Inductive Step:** We prove correctness for `knapsack_memoization(i, w)`.

**Case 1: Result is already memoized** (`memo[i][w] != -1`):
- Return the cached value
- By the memoization invariant, this value was correctly computed earlier ✓

**Case 2: Item doesn't fit** ($w_i > w$):
- Compute `memo[i][w] = knapsack_memoization(i-1, w)`
- By the inductive hypothesis, the recursive call returns the correct value for items $1$ to $i-1$ with capacity $w$
- Since item $i$ cannot be included, this is the optimal value ✓

**Case 3: Item fits** ($w_i \leq w$):
- Compute two options:
  - **Include:** `include_item = values[i-1] + knapsack_memoization(i-1, w - weights[i-1])`
  - **Exclude:** `exclude_item = knapsack_memoization(i-1, w)`
- By the inductive hypothesis, both recursive calls return correct values for their respective subproblems
- The maximum of these two options is the optimal choice ✓
- Store this in `memo[i][w]`

**Conclusion:** By strong induction and the memoization invariant, the algorithm correctly computes the optimal value for all valid inputs.

**Reconstruction Correctness:** The `reconstruct_solution()` function backtracks through the memo table:
- If `memo[i][w] != memo[i-1][w]`, then item $i$ must have been included (its inclusion changed the value)
- This correctly identifies all selected items ✓

---

### Model of Computation/Assumptions

#### Computational Model

- **RAM Model (Random Access Machine):** The algorithm assumes:
  - Array/vector access takes $O(1)$ time
  - Arithmetic operations (addition, subtraction, comparison) take $O(1)$ time
  - Function calls take $O(1)$ time (though recursion has overhead)
  - Memory allocation is pre-computed (memo table allocated upfront)

#### Assumptions

1. **Input Format:**
   - `n` items with non-negative integer weights and values
   - Capacity `W` is a non-negative integer
   - All values fit within 64-bit signed integers (`int64`)

2. **Data Types:**
   - Uses `long long` (int64) to prevent overflow
   - Assumes 64-bit arithmetic operations are constant time

3. **Memory:**
   - Sufficient memory to allocate $(n+1) \times (W+1)$ memo table
   - For large `W` (e.g., $W > 10^7$), memory requirements can be prohibitive
   - Stack space is sufficient for recursion depth $O(n)$

4. **Initialization:**
   - Memo table initialized to `-1` to indicate uncomputed states
   - Assumes `-1` is not a valid result value (which is true for non-negative values)

5. **Optimality:**
   - Guarantees **optimal solution** (maximum value)
   - Exact algorithm, not an approximation

6. **Recursion Overhead:**
   - Function call overhead exists (stack frame creation/destruction)
   - This makes memoization slightly slower than iterative DP in practice
   - However, asymptotic complexity remains the same

7. **Cache Efficiency:**
   - Memory access patterns are less cache-friendly than bottom-up DP
   - Top-down recursion has irregular memory access patterns

---

### Case Analysis (Best/Average/Worst)

#### Best Case

**Input Characteristics:** Items with very large weights relative to capacity, causing early termination in many branches.

**Time Complexity:** $O(n \times W)$

**Explanation:** Even with early termination, the memoization approach still needs to explore and cache results for reachable states. In the best case, fewer states might be visited if many branches terminate early (items don't fit), but the algorithm still needs to compute results for all visited states. The asymptotic complexity remains $O(n \times W)$ because the memo table structure dictates the maximum number of unique subproblems.

**Space Complexity:** $O(n \times W)$ - memo table is fully allocated upfront.

**Practical Performance:** Slightly faster than average due to more cache hits from early termination patterns, but the difference is marginal.

#### Average Case

**Input Characteristics:** Random weights and values with typical distribution.

**Time Complexity:** $\Theta(n \times W)$

**Explanation:** For typical inputs, most of the $(n+1) \times (W+1)$ states are visited and computed. The recursive structure systematically explores the state space, computing each unique subproblem once. The memoization ensures that each state is computed at most once, resulting in the characteristic $O(n \times W)$ complexity.

**Space Complexity:** $O(n \times W)$ for the memo table.

**Practical Performance:** 
- Slightly slower than bottom-up DP due to function call overhead
- Better than bottom-up if many states are unreachable (sparse state space)
- Memory access patterns are less cache-friendly

#### Worst Case

**Input Characteristics:** All items can potentially fit; all states in the memo table are visited.

**Time Complexity:** $\Theta(n \times W)$

**Explanation:** When all items can fit within various capacity constraints, the algorithm explores the entire state space. Every combination of `(i, w)` for $i \in [0, n]$ and $w \in [0, W]$ is visited. Each state is computed once and cached, resulting in exactly $n \times W$ computations, each taking $O(1)$ time.

**Example:** If `W = 1000` and all weights are small (e.g., `≤ 10`), then all capacity values from `0` to `W` are reachable for each item, maximizing the number of states visited.

**Space Complexity:** $O(n \times W)$ - full memo table is used.

**Practical Performance:** This represents the maximum work the algorithm performs, matching the complexity of bottom-up DP but with additional recursion overhead.

---

### Comparison of Cases

| Case | Time Complexity | Space Complexity | States Visited |
|------|----------------|------------------|----------------|
| **Best** | $O(n \times W)$ | $O(n \times W)$ | Fewer (some pruning) |
| **Average** | $\Theta(n \times W)$ | $O(n \times W)$ | Most states |
| **Worst** | $\Theta(n \times W)$ | $O(n \times W)$ | All states |

**Key Insight:** Unlike brute force (which is exponential in all cases), memoization reduces complexity to pseudo-polynomial $O(n \times W)$ by eliminating redundant computation. The difference between cases is primarily in the *constant factors* (exact number of states visited) rather than asymptotic complexity.

---

### Comparison: Memoization vs Bottom-Up Dynamic Programming

Both approaches solve the same problem with the same asymptotic complexity, but differ in implementation style and practical performance:

| Feature | Top-Down (Memoization) | Bottom-Up (Tabulation) |
|---------|------------------------|------------------------|
| **Time Complexity** | $O(n \times W)$ | $O(n \times W)$ |
| **Space Complexity** | $O(n \times W)$ + $O(n)$ stack | $O(n \times W)$ (reducible to $O(W)$) |
| **Approach** | Recursive with caching | Iterative with table filling |
| **Computation Order** | On-demand (lazy) | Systematic (eager) |
| **Function Overhead** | Higher (recursion) | Lower (loops) |
| **Cache Efficiency** | Lower (irregular access) | Higher (sequential access) |
| **Code Intuition** | More natural (matches recurrence) | Less intuitive initially |
| **Sparse States** | Efficient (computes only needed) | Inefficient (computes all) |
| **Stack Overflow Risk** | Possible for large `n` | None |
| **Space Optimization** | Harder | Easier (rolling array) |

**When to Prefer Memoization:**
- When the state space is sparse (many unreachable states)
- When the recursive formulation is more intuitive
- For educational purposes (closer to mathematical recurrence)
- When prototyping or debugging (easier to trace)

**When to Prefer Bottom-Up DP:**
- When all states need to be computed anyway
- When maximum performance is critical
- When space optimization is needed
- For very large `n` (avoid stack overflow)

---

### Summary

The **Memoization solution** for the 0/1 Knapsack problem is:

| Aspect | Complexity |
|--------|------------|
| **Time Complexity** | $\Theta(n \times W)$ (pseudo-polynomial) |
| **Space Complexity** | $O(n \times W)$ + $O(n)$ recursion stack |
| **Correctness** | Proven optimal via induction + memoization invariant |
| **Cases** | Best = Average = Worst (asymptotically) |

**Strengths:**
- Guarantees optimal solution
- Eliminates exponential redundancy of brute force
- More intuitive than bottom-up DP (follows natural recursion)
- Efficient for sparse state spaces
- Easier to understand and debug

**Weaknesses:**
- Pseudo-polynomial (impractical for very large `W`)
- Function call overhead (slower than bottom-up in practice)
- Stack overflow risk for very large `n`
- Less cache-friendly memory access patterns
- Space optimization is harder than bottom-up

**Complexity Improvement Over Brute Force:**
- **Brute Force:** $O(2^n)$ - explores all $2^n$ subsets repeatedly
- **Memoization:** $O(n \times W)$ - solves each of $n \times W$ subproblems once

**Example:** For `n = 20` and `W = 1000`:
- **Brute Force:** ~1 million recursive calls (impractical for `n > 25`)
- **Memoization:** ~20,000 unique subproblems (very practical)

For reference: https://www.w3schools.com/dsa/dsa_ref_knapsack.php, under the section "Memoization"