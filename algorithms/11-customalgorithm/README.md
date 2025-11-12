# Custom (Modified Genetic) Algorithm for 0/1 Knapsack

## About the Algorithm

The Genetic Algorithm (GA) is a heuristic inspired by Charles Darwin's theory of natural evolution. This algorithm reflects the process of natural selection where the fittest individuals are selected for reproduction in order to produce offspring of the next generation.

As a heuristic, the GA does not guarantee the optimal solution. However, it is very effective at finding high-quality, near-optimal solutions in a fraction of the time required by exact algorithms, especially for large problem sizes.

## Implementation

### Modifications

In this version of the GA, we introduce several modifications aimed at improving the quality of solutions and the rate of convergence towards them. We do this by implementing intelligent heuristics in place of pure randomness, such as in individual repair and generational mutation.

#### Greedy Repair Strategy

The base GA repairs overweight individuals by randomly removing items till the knapsack is valid. This version instead iteratively removes the items with the lowest value/weight ratio. This helps to guide the population towards higher-quality solutions.

#### Elitism

To ensure that the best solution found so far is not lost in subsequent generations, this algorithm incorporates elitism. In each generation, the best individual (the one with the highest fitness) from the current population is automatically carried over to the next generation. This guarantees that the quality of the solution in the population can only increase or stay the same over time, leading to better and more consistent results.

#### Dynamic Hyperparameters

Instead of using fixed values, the `POPULATION_SIZE` and `MAX_GENERATIONS` are dynamically determined based on `N`. This allows the algorithm to adapt its effort to the problem's scale. For smaller `N`, it runs more generations with a smaller population, and for larger `N`, it uses a larger population for fewer generations to cover the solution space more effectively. These can also be overridden via command-line arguments.

### Complexity

#### Individual Operations

| Function | Time Complexity | Space Complexity | Notes |
|----------|----------------|------------------|-------|
| `Individual::calculateMetrics()` | O(N) | O(1) | Iterates through all N items once |
| `Individual::getFitness()` | O(1) | O(1) | Amortized constant time with caching |
| `Individual::repair()` | O(N) | O(1) | Iterates through v/w-sorted items; worst case removes all |

#### Preprocessing

| Function | Time Complexity | Space Complexity | Notes |
|----------|----------------|------------------|-------|
| `preSortItems()` | O(N log N) | O(N) | Computes value/weight ratios O(N), sorts N items O(N log N) |

#### Population Initialisation

| Function | Time Complexity | Space Complexity | Notes |
|----------|----------------|------------------|-------|
| `generateInitialPopulation()` | O(P × N) | O(P × N) | Creates P individuals, each initialised and repaired |

#### Selection

| Function | Time Complexity | Space Complexity | Notes |
|----------|----------------|------------------|-------|
| `selection()` | O(1) | O(1) | Tournament selection with 4 random individuals, fitness cached |

#### Genetic Operators

| Function | Time Complexity | Space Complexity | Notes |
|----------|----------------|------------------|-------|
| `crossover()` | O(N) | O(1) | Single-point crossover with metric recalculation for both children |
| `mutate()` | O(N) | O(1) | Iterates through all N bits, expected mutations: MUTATION_RATE × N |

#### Generational Evolution

| Function | Time Complexity | Space Complexity | Notes |
|----------|----------------|------------------|-------|
| `nextGeneration()` | O(P × N) | O(1) | Processes P individuals through selection, crossover/reproduction, mutation, and repair |

#### Overall Algorithm

**`solveKnapsackGenetic()`**

* **Time Complexity**: **O(G × P × N + N log N) ≈ O(G × P × N)**
  * Pre-sort items: O(N log N)
  * Initial population generation: O(P × N)
  * G generations of evolution: G × O(P × N)
  * Finding best individual: O(P)
  * Collecting selected items: O(N)
  * Total: dominated by O(G × P × N)

* **Space Complexity**: **O(P × N)**
  * Two populations: 2 × P individuals × N bits each
  * SORTED_ITEMS array: O(N)
  * Item arrays (ITEM_WEIGHTS, ITEM_VALUES): O(N)
  * Result selectedItems: O(N) worst case
  * Total: dominated by O(P × N)

Where:

* **N** = number of items
* **P** = `POPULATION_SIZE` (dynamically set: 20–150 based on N)
* **G** = `MAX_GENERATIONS` (dynamically set: 30–200 based on N)

#### Dynamic Scaling Analysis

The implementation uses adaptive hyperparameters to maintain practical efficiency:

| N Range | P (Population) | G (Generations) | G × P | Effective Complexity |
|---------|----------------|-----------------|-------|---------------------|
| N < 100 | 20 | 200 | 4,000 | O(4,000N + N log N) |
| 100 ≤ N < 1,000 | 50 | 100 | 5,000 | O(5,000N + N log N) |
| 1,000 ≤ N < 10,000 | 100 | 50 | 5,000 | O(5,000N + N log N) |
| N ≥ 10,000 | 150 | 30 | 4,500 | O(4,500N + N log N) |

By keeping the product G × P approximately constant as N grows, the algorithm achieves **pseudo-linear O(N) scaling** in practice for large problem instances. This is a key optimization that makes the GA feasible for large-scale knapsack problems where exact algorithms would be computationally prohibitive.

## References

1. <https://www.youtube.com/watch?v=MacVqujSXWE>
2. <https://arpitbhayani.me/blogs/genetic-knapsack> - `Solving the Knapsack Problem with Evolutionary Algorithms` _(Fun fact: Arpit Bhayani is a IIIT alum!)_
3. <https://github.com/arpitbbhayani/genetic-knapsack> - Python code for GA _(by Arpit Bhayani, linked in his blog)_
4. <https://doi.org/10.47191/etj/v9i07.10> - `0-1 Knapsack Problem Solving using Genetic Optimization Algorithm`
5. <https://doi.org/10.1145/3230905.3230907> - `A hybrid genetic algorithm for solving 0/1 Knapsack Problem`
6. <https://doi.org/10.1109/ICACCCT.2014.7019272> - `Solving the 0-1 Knapsack Problem Using Genetic Algorithm and Rough Set Theory`
