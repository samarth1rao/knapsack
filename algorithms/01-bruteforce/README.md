# Brute Force Approach for 0/1 Knapsack Problem

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

#### The Brute Force Approach

The brute force approach is the most straightforward solution: **try every possible combination of items** and select the one with maximum value that fits within the capacity constraint.

**In English:**
We use a recursive strategy that, for each item, explores two possibilities: either include the item in our knapsack or exclude it. We recursively solve the problem for the remaining items with the updated capacity, then compare which choice (include vs. exclude) gives us a better total value. We keep track of which items we selected and return the best solution found.

**Mathematically**, the recursive relation is:

$$
K(n, W) = \begin{cases}
0 & \text{if } n < 0 \text{ or } W = 0 \\
K(n-1, W) & \text{if } w_n > W \\
\max(K(n-1, W), \, v_n + K(n-1, W - w_n)) & \text{if } w_n \leq W
\end{cases}
$$

Where:
- $K(n, W)$ represents the maximum value achievable using items $0$ to $n$ with capacity $W$
- $K(n-1, W)$ represents **excluding** item $n$
- $v_n + K(n-1, W - w_n)$ represents **including** item $n$

**Key Steps:**
1. For each item, make a recursive call to exclude it
2. If the item fits, make another recursive call to include it
3. Compare both options and return the better one
4. Base case: when no items remain or capacity is zero, return 0

This approach exhaustively explores the entire solution space, forming a **binary decision tree** of depth `n`, where each level corresponds to an item and each node represents a decision (include/exclude).

---

### Time Complexity

#### Analysis

The algorithm explores a binary tree of decisions:

1. **Recursive Tree Structure:**
   - At each level (item), we make up to 2 recursive calls (include and exclude)
   - The tree has depth `n` (one level per item)
   - In the worst case, we explore all $2^n$ possible subsets

2. **Work Per Node:**
   - Each recursive call performs constant-time operations: comparison, addition, vector operations
   - Work per node: $O(1)$

3. **Total Number of Nodes:**
   - A complete binary tree of depth `n` has $2^n$ leaf nodes
   - Total nodes in the tree: $O(2^n)$

**Overall Time Complexity:**

- **Best Case:** $O(2^n)$
  - Even in the best case, the algorithm explores the exponential decision tree
  - Early pruning (when items don't fit) reduces some branches but doesn't change asymptotic complexity

- **Average Case:** $\Theta(2^n)$
  - On average, most of the exponential tree is explored
  - The recursion doesn't have significant pruning for typical inputs

- **Worst Case:** $\Theta(2^n)$
  - When all items fit within capacity, every possible subset must be considered
  - The algorithm explores all $2^n$ combinations

**Why Exponential?**
For `n` items, there are $2^n$ possible subsets (each item can be in or out). The brute force approach must examine each subset to determine which one is optimal.

**Example:** With `n = 20` items, there are $2^{20} = 1,048,576$ combinations to check. With `n = 30`, there are over 1 billion combinations!

---

### Space Complexity

#### Analysis

The algorithm uses space for:

1. **Recursive Call Stack:**
   - Maximum recursion depth: `n` (when processing items sequentially)
   - Each stack frame stores: capacity, item index, local variables
   - Stack space: $O(n)$

2. **Solution Tracking:**
   - Each recursive call may create a vector to store selected items
   - In the worst case, vectors are copied and merged at each level
   - This adds significant overhead: $O(n)$ per recursive call

3. **Input Storage:**
   - `weights` and `values` vectors: $O(n)$

4. **Output Storage:**
   - `selectedItems` vector: $O(n)$ in the worst case

**Total Space Complexity:**

- **Auxiliary Space:** $O(n)$ for the recursion stack (not counting solution tracking)
- **Total Space:** $O(n)$ for a clean recursive implementation

**Important Note:** The current implementation creates and copies vectors at each recursive level, which in practice can lead to $O(n \cdot 2^n)$ space in the worst case due to vector copying. However, with proper optimization (using references and in-place modification), the space can be reduced to $O(n)$ for the call stack alone.

---

### Correctness

#### Proof of Correctness

We prove correctness using **strong induction** on the number of remaining items.

**Claim:** The function `knapsackBruteForceRecursive(W, weights, values, n)` correctly returns the maximum value achievable using items $0$ to $n$ with capacity $W$, along with the selected items.

**Base Case:** When $n < 0$ (no items) or $W = 0$ (zero capacity):
- Returns $(0, \{\})$ (value 0, empty set)
- This is correct: with no items or no capacity, the maximum value is 0

**Inductive Hypothesis:** Assume that for all $k < n$, the function correctly computes the maximum value for items $0$ to $k$ with any capacity $w \leq W$.

**Inductive Step:** We prove correctness for item $n$.

For item $n$ with weight $w_n$ and value $v_n$:

1. **Case 1: Item doesn't fit** ($w_n > W$)
   - The algorithm returns `excludeResult = knapsackBruteForceRecursive(W, weights, values, n-1)`
   - By the inductive hypothesis, this is the optimal solution for items $0$ to $n-1$
   - Since item $n$ cannot be included, this is also optimal for items $0$ to $n$ ✓

2. **Case 2: Item fits** ($w_n \leq W$)
   - **Exclude option:** `excludeResult = knapsackBruteForceRecursive(W, weights, values, n-1)`
     - By hypothesis, this is optimal for items $0$ to $n-1$ with capacity $W$
   
   - **Include option:** `includeResult = knapsackBruteForceRecursive(W - w_n, weights, values, n-1)`
     - By hypothesis, this is optimal for items $0$ to $n-1$ with capacity $W - w_n$
     - Adding $v_n$ gives the value when including item $n$
   
   - The algorithm returns `max(includeResult, excludeResult)`
   - This is correct because the optimal solution either includes item $n$ or doesn't
   - By considering both options and taking the maximum, we get the optimal solution ✓

**Conclusion:** By strong induction, the algorithm correctly computes the optimal solution for all valid inputs.

**Completeness:** The algorithm is complete because it exhaustively explores all $2^n$ possible subsets of items, guaranteeing that the optimal solution is found.

---

### Model of Computation/Assumptions

#### Computational Model

- **RAM Model (Random Access Machine):** The algorithm assumes:
  - Array/vector access takes $O(1)$ time
  - Arithmetic operations (addition, subtraction, comparison) take $O(1)$ time
  - Function calls and returns take $O(1)$ time
  - Memory allocation and deallocation take $O(1)$ time per element

#### Assumptions

1. **Input Format:**
   - `n` items with non-negative integer weights and values
   - Capacity `W` is a non-negative integer
   - All values fit within 64-bit signed integers (`int64`)

2. **Data Types:**
   - Uses `long long` (int64) to prevent overflow for large accumulated values
   - Assumes 64-bit arithmetic operations are constant time

3. **Memory:**
   - Sufficient stack space is available for recursion depth up to `n`
   - Stack overflow is not a concern (for reasonable `n`, typically $n < 10^4$)
   - For very large `n`, this could cause stack overflow

4. **Optimality:**
   - The algorithm guarantees an **optimal solution** (maximum value)
   - It's an exact algorithm, not an approximation

5. **Input Validity:**
   - No validation is performed on input data
   - Assumes well-formed input (correct counts, non-negative values)

6. **Practicality:**
   - Due to exponential time complexity, this approach is only practical for small `n`
   - Typically feasible for $n \leq 20$ to $25$ items

---

### Case Analysis (Best/Average/Worst)

#### Best Case

**Input Characteristics:** All items have weights greater than the capacity (none fit).

**Time Complexity:** $O(2^n)$

**Explanation:** Even when no items fit, the algorithm still makes recursive calls to explore both the "exclude" and "include" branches. The "include" branch immediately returns when checking if the item fits, but the recursive tree is still explored. While there's some pruning, the exponential nature remains.

**Example:** If `W = 10` and all weights are `> 10`, the algorithm still recurses through the decision tree, though the "include" branches terminate quickly.

**Space Complexity:** $O(n)$ for the recursion stack.

**Practical Note:** This case is slightly faster in practice due to early termination of "include" branches, but asymptotically still exponential.

#### Average Case

**Input Characteristics:** Random weights and values with some items fitting and some not.

**Time Complexity:** $\Theta(2^n)$

**Explanation:** For typical inputs, the algorithm explores most of the exponential decision tree. Some branches may be pruned when items don't fit, but on average, a significant portion of the $2^n$ subsets are examined. The recursive structure ensures that the algorithm visits an exponential number of nodes.

**Space Complexity:** $O(n)$ for the recursion stack.

**Practical Behavior:** The actual number of nodes visited depends on how many items can fit, but it remains exponential for any reasonable distribution of weights.

#### Worst Case

**Input Characteristics:** All items have weights less than or equal to the capacity (all fit).

**Time Complexity:** $\Theta(2^n)$

**Explanation:** This is the true worst case where the algorithm must explore every possible subset. Both "include" and "exclude" branches are fully explored for every item, resulting in a complete binary tree of depth `n` with $2^n$ leaf nodes. No pruning occurs.

**Example:** If `W = 1000` and all weights are `≤ 10`, then for `n = 20` items, all $2^{20} = 1,048,576$ combinations must be checked.

**Space Complexity:** $O(n)$ for the recursion stack (though vector copying in the current implementation may increase this).

**Critical Problem:** This exponential growth makes the algorithm **infeasible for large inputs**. Even modern computers cannot handle $n > 30$ in reasonable time.

---

### Comparison of Cases

| Case | Time Complexity | Space Complexity | Input Characteristics |
|------|----------------|------------------|----------------------|
| **Best** | $O(2^n)$ | $O(n)$ | Items don't fit (some pruning) |
| **Average** | $\Theta(2^n)$ | $O(n)$ | Mixed fitting items |
| **Worst** | $\Theta(2^n)$ | $O(n)$ | All items fit (full tree) |

**Key Insight:** Unlike some algorithms where best/worst cases differ significantly, the brute force approach is exponential **in all cases**. The difference is only in the constant factors (how much of the tree is explored), not in the asymptotic complexity.

---

### Summary

The **Brute Force solution** for the 0/1 Knapsack problem is:

| Aspect | Complexity |
|--------|------------|
| **Time Complexity** | $\Theta(2^n)$ (exponential) |
| **Space Complexity** | $O(n)$ (recursion stack) |
| **Correctness** | Proven optimal via induction |
| **Practical Limit** | $n \leq 20$ to $25$ items |

**Strengths:**
- Simple and intuitive implementation
- Guarantees optimal solution
- Easy to understand and verify correctness
- No dependency on capacity value (unlike DP)

**Weaknesses:**
- **Exponential time complexity** - impractical for large `n`
- Explores redundant subproblems (no memoization)
- Significant recursive overhead
- Stack overflow risk for very large `n`

**When to Use:**
- Very small problem instances ($n \leq 20$)
- Educational purposes to understand the problem
- As a baseline to compare other algorithms
- When optimality is required and `n` is guaranteed to be small

**Why Better Algorithms Exist:**
The brute force approach has **overlapping subproblems** (the same subproblems are solved multiple times). Dynamic programming and memoization exploit this structure to reduce complexity from $O(2^n)$ to $O(n \times W)$, making much larger problems tractable.

For reference: https://www.w3schools.com/dsa/dsa_ref_knapsack.php