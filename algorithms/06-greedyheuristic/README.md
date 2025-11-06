# Greedy Heuristic for 0/1 Knapsack

This implements the classic greedy heuristic for the 0-1 Knapsack problem.

## Algorithm Description

The greedy heuristic works by:
1. Calculate the value-to-weight ratio (density) for each item
2. Sort items in descending order of density
3. Greedily select items in this order until no more items fit in the knapsack

## Time Complexity
- **Time**: O(n log n) due to sorting
- **Space**: O(n) for storing items and their densities

## Approximation Quality
- This is a heuristic approach that does not guarantee optimal solutions
- Performance depends on the instance characteristics
- Generally performs well when high-density items are clearly better choices
- May perform poorly when there are complex trade-offs between items

## Implementation Details

- Uses `std::sort` with custom comparator for density-based sorting
- Handles edge cases (zero weights, though unlikely in valid inputs)
- Outputs selected items in sorted index order for consistency
- Measures execution time and memory usage as per project requirements

## References
- https://www.biyanicolleges.org/knapsack-problem-using-greedy-method/
- Standard greedy algorithm for fractional knapsack (adapted for 0-1 variant)  