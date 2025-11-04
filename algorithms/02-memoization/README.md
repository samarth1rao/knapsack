# Memoization Approach

This is a pseudopolynomial-time exact algorithm that uses a top-down memoization approach.
It follows same idea as dynamic programming, where subproblem results are stored and reused to avoid redundant computation.

The key difference lies in how results are computed and stored:
in bottom-up dynamic programming, values are filled iteratively in a table using loops;
in memoization, values are computed recursively through function calls, and results are cached as the recursion unwinds.

This causes function overheads in this approach.

How it differs apart from the algorithm:

| Feature                | Top-Down (Memoization)                 | Bottom-Up (Tabulation)                          |
| ---------------------- | -------------------------------------- | ----------------------------------------------- |
| Time Complexity        | O(n × capacity)                        | O(n × capacity)                                 |
| Space Complexity       | O(n × capacity)                        | O(n × capacity) (can be reduced to O(capacity)) |
| Function Call Overhead | Slightly higher                        | Minimal                                         |
| Good for Sparse States | Yes (only computes reachable states) | No (computes entire table)                    |

  
    


For reference: https://www.w3schools.com/dsa/dsa_ref_knapsack.php, under the section "Memoization"