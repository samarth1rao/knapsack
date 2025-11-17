#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <numeric>
#include <cstring>
#include <cmath>
#include <limits>
#include <map>
#include <set> // For sparse set operations

// Use int64 for large numbers
using int64 = long long;

// Struct to hold the results of the memetic algorithm.
struct Result {
    int64 maxValue;                 // The maximum value found for the knapsack.
    std::vector<int> selectedItems; // The indices of the items selected to achieve the max value.
    long long executionTime;        // Total execution time in microseconds.
    size_t memoryUsed;              // Approximate memory usage in bytes.
};

// Struct to hold item properties for sorting
struct ItemProperty {
    int id;
    double ratio; // Can be v/w or reduced cost (v_i - u*w_i)
    int64 weight;
    int64 value;

    // Sort by ratio descending
    bool operator<(const ItemProperty& other) const {
        return ratio > other.ratio;
    }
};

// --- Memetic Algorithm Hyperparameters ---
size_t POPULATION_SIZE = 0;   // Total population, will be divided among islands.
size_t MAX_GENERATIONS = 0;   // Number of generations.
size_t NUM_ISLANDS = 8;       // Number of independent populations (2.4)
size_t ISLAND_SIZE = 50;      // Population size per island (2.4)
size_t MIGRATION_INTERVAL = 25; // How often to migrate best individual (2.4)
unsigned int SEED = std::random_device()(); // Seed for random number generator.

// Lagrangian & SA Parameters
size_t LAGRANGIAN_ITERATIONS = 100; // Iterations for preprocessing (Part 1.4)
double BASE_SA_TEMP = 100.0;      // Base temperature for local search (Part 4.3)
double SA_COOLING_RATE = 0.90;     // Cooling rate for local search
size_t SA_ITERATIONS = 50;         // Steps per local search per child

// Guided Mutation Parameters
double BASE_MUTATION_RATE = 0.1; // Per-individual probability of a mutation event
double CORE_MUTATION_PROB;
double NON_CORE_MUTATION_PROB;

// --- Global variables for the FULL knapsack problem (N) ---
int64 FULL_KNAPSACK_CAPACITY;
std::vector<int64> FULL_ITEM_WEIGHTS;
std::vector<int64> FULL_ITEM_VALUES;
size_t FULL_NUM_ITEMS;

// --- Global variables for the CORE knapsack problem (N') ---
int64 KNAPSACK_CAPACITY; // This will be FULL_CAPACITY - fixed_weight
std::vector<int64> ITEM_WEIGHTS;
std::vector<int64> ITEM_VALUES;
size_t NUM_ITEMS; // This is N'
int64 GLOBAL_LOWER_BOUND = 0; // Best feasible value found (Z_LB)

// --- Preprocessing & Heuristic Data ---
std::vector<double> REDUCED_COSTS; // v_i - u*w_i, for the CORE problem
enum ItemType { CORE, X1_HIGH, X0_LOW };
std::vector<ItemType> ITEM_CLASSIFICATION;

// Helper vectors for sorted orders (for the CORE problem)
std::vector<ItemProperty> v_w_order;    // Sorted by value/weight
std::vector<ItemProperty> rcbo_order;   // Sorted by reduced cost
std::vector<ItemProperty> value_order;  // Sorted by value
std::vector<ItemProperty> weight_order; // Sorted by weight (ascending)

// Preprocessing results (Part 1.1)
std::vector<size_t> core_to_original_index_map; // Maps core index -> full index
std::vector<int> fixed_one_items; // Original indices of items fixed to 1
int64 fixed_items_value = 0;
int64 fixed_items_weight = 0;
double optimal_lagrangian_multiplier = 0.0;

// Global random number generator
std::mt19937 rng;


/**
 * @class Individual
 * @brief Represents a sparse-chromosome individual (Part 2.1)
 *
 * Uses a sorted vector of indices (std::vector<size_t>) to represent
 * the set of selected items, enabling O(k) space.
 */
class Individual {
public:
    // Sparse chromosome: sorted list of selected item indices (from core problem)
    std::vector<size_t> selectedItemIndices; // (Part 2.1)
    int64 totalValue = 0;
    int64 totalWeight = 0;
    mutable int64 cachedFitness = -1;
    mutable bool fitnessValid = false;

    // Default constructor
    Individual() {}

    // Calculates total value and weight from scratch from the sparse vector. O(k)
    void calculateMetrics() { // (Part 2.2)
        totalValue = 0;
        totalWeight = 0;
        for (size_t item_index : selectedItemIndices) {
            totalValue += ITEM_VALUES[item_index];
            totalWeight += ITEM_WEIGHTS[item_index];
        }
        fitnessValid = false;
    }

    // Get fitness. O(1) if cached, O(k) otherwise.
    int64 getFitness() const {
        if (fitnessValid) {
            return cachedFitness;
        }
        if (totalWeight > KNAPSACK_CAPACITY) {
            cachedFitness = 0; // Penalize invalid solutions
        } else {
            cachedFitness = totalValue;
        }
        fitnessValid = true;
        return cachedFitness;
    }
    
    // Helper: adds an item, maintains sorted order. O(log k + k)
    void add(size_t item_index) {
        auto it = std::lower_bound(selectedItemIndices.begin(), selectedItemIndices.end(), item_index);
        if (it == selectedItemIndices.end() || *it != item_index) {
            selectedItemIndices.insert(it, item_index);
            // Metrics updated externally by caller (e.g., calculateMetrics)
        }
    }

    // Helper: removes an item, maintains sorted order. O(log k + k)
    void remove(size_t item_index) {
        auto it = std::lower_bound(selectedItemIndices.begin(), selectedItemIndices.end(), item_index);
        if (it != selectedItemIndices.end() && *it == item_index) {
            selectedItemIndices.erase(it);
            // Metrics updated externally
        }
    }
    
    // Two-phase greedy repair. (Part 2.2)
    void repair() {
        if (totalWeight <= KNAPSACK_CAPACITY) {
            // Phase 2 (Add): O(N' log k)
            for (const auto& item : v_w_order) {
                if (totalWeight + item.weight > KNAPSACK_CAPACITY) continue;
                // Check if item is already present O(log k)
                auto it = std::lower_bound(selectedItemIndices.begin(), selectedItemIndices.end(), item.id);
                if (it == selectedItemIndices.end() || *it != (size_t)item.id) {
                    selectedItemIndices.insert(it, item.id); // O(k)
                    totalWeight += item.weight;
                    totalValue += item.value;
                }
            }
        } else {
            // Phase 1 (Remove): O(k * N') or O(k log k)
            // This is O(k * N') which is slow. A better O(k log k) way:
            std::vector<ItemProperty> in_knapsack;
            in_knapsack.reserve(selectedItemIndices.size());
            for(size_t item_index : selectedItemIndices) {
                in_knapsack.push_back(v_w_order[item_index]);
            }
            // Sort by v/w ascending (worst items first)
            std::sort(in_knapsack.begin(), in_knapsack.end(), 
                [](const ItemProperty& a, const ItemProperty& b){ return a.ratio < b.ratio; });

            std::set<size_t> to_remove;
            for(const auto& item : in_knapsack) {
                if(totalWeight <= KNAPSACK_CAPACITY) break;
                totalWeight -= item.weight;
                totalValue -= item.value;
                to_remove.insert(item.id);
            }
            
            // Rebuild selectedItemIndices O(k)
            std::vector<size_t> new_indices;
            new_indices.reserve(selectedItemIndices.size());
            for(size_t item_index : selectedItemIndices) {
                if(to_remove.find(item_index) == to_remove.end()) {
                    new_indices.push_back(item_index);
                }
            }
            selectedItemIndices = std::move(new_indices);
        }

        fitnessValid = false;
        if (totalWeight <= KNAPSACK_CAPACITY && totalValue > GLOBAL_LOWER_BOUND) {
            GLOBAL_LOWER_BOUND = totalValue;
        }
    }

    // Guided Mutation. (Part 5.2)
    void mutate() {
        if (NUM_ITEMS == 0) return;
        std::uniform_real_distribution<double> probDist(0.0, 1.0);

        // Pick a random item
        std::uniform_int_distribution<size_t> dist(0, NUM_ITEMS - 1);
        size_t item_index = dist(rng);
        
        // Find its mutation probability
        double prob = (ITEM_CLASSIFICATION[item_index] == CORE) ? CORE_MUTATION_PROB : NON_CORE_MUTATION_PROB;
        
        if (probDist(rng) < prob) {
            // Perform a "flip" (Add or Remove)
            auto it = std::lower_bound(selectedItemIndices.begin(), selectedItemIndices.end(), item_index);
            if (it != selectedItemIndices.end() && *it == item_index) {
                selectedItemIndices.erase(it); // Remove
            } else {
                selectedItemIndices.insert(it, item_index); // Add
            }
            calculateMetrics(); // Full O(k) recalculation
        }
    }

    // Memetic part: Simulated Annealing local search (Part 4.3)
    void localSearch() {
        if (NUM_ITEMS == 0) return;
        double temp = BASE_SA_TEMP;
        std::uniform_real_distribution<double> prob(0.0, 1.0);
        std::uniform_int_distribution<size_t> index_dist(0, NUM_ITEMS - 1);

        calculateMetrics();
        repair();
        int64 currentFitness = getFitness();

        for (size_t i = 0; i < SA_ITERATIONS; ++i) {
            Individual neighbor = *this;
            
            // 1-Opt move: flip a random bit (Add or Remove) (Part 4.1)
            size_t flip_idx = index_dist(rng);
            auto it = std::lower_bound(neighbor.selectedItemIndices.begin(), neighbor.selectedItemIndices.end(), flip_idx);
            if (it != neighbor.selectedItemIndices.end() && *it == flip_idx) {
                neighbor.selectedItemIndices.erase(it); // Remove
            } else {
                neighbor.selectedItemIndices.insert(it, flip_idx); // Add
            }
            
            // Recalculate, repair, and get fitness
            neighbor.calculateMetrics();
            neighbor.repair();
            int64 neighborFitness = neighbor.getFitness();

            int64 delta = neighborFitness - currentFitness;
            if (delta > 0 || prob(rng) < std::exp(delta / temp)) {
                *this = neighbor; // Accept the neighbor
                currentFitness = neighborFitness;
            }
            temp *= SA_COOLING_RATE; // Cool down
        }
    }
};

// Helper: Hamming distance for sparse vectors (Symmetric Difference) O(k)
size_t hammingDist(const Individual& a, const Individual& b) {
    std::vector<size_t> diff;
    std::set_symmetric_difference(
        a.selectedItemIndices.begin(), a.selectedItemIndices.end(),
        b.selectedItemIndices.begin(), b.selectedItemIndices.end(),
        std::back_inserter(diff)
    );
    return diff.size();
}


// --- 1. Lagrangian Relaxation & Problem Reduction (Part 1.4) ---

// Solves the Lagrangian subproblem O(N) on the FULL problem
std::pair<int64, int64> lagrangianSubproblem(double u, const std::vector<int64>& weights, const std::vector<int64>& values) {
    int64 subproblem_value = 0;
    int64 subproblem_weight = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] - u * weights[i] > 0) {
            subproblem_value += (int64)(values[i] - u * weights[i]);
            subproblem_weight += weights[i];
        }
    }
    return {subproblem_value, subproblem_weight};
}

// Simple greedy heuristic (v/w) on the FULL problem O(N log N)
int64 greedyHeuristic(const std::vector<int64>& weights, const std::vector<int64>& values, int64 capacity) {
    std::vector<ItemProperty> items(values.size());
    for(size_t i = 0; i < values.size(); ++i) {
        items[i] = { (int)i, (weights[i] > 0 ? (double)values[i] / weights[i] : 0), weights[i], values[i] };
    }
    std::sort(items.begin(), items.end()); // Sort by v/w desc

    int64 current_weight = 0;
    int64 current_value = 0;
    for(const auto& item : items) {
        if (current_weight + item.weight <= capacity) {
            current_weight += item.weight;
            current_value += item.value;
        }
    }
    return current_value;
}

// Main preprocessing function (Part 1.1)
void preprocess() {
    // 1. Get initial Lower Bound (Z_LB) on FULL problem
    GLOBAL_LOWER_BOUND = greedyHeuristic(FULL_ITEM_WEIGHTS, FULL_ITEM_VALUES, FULL_KNAPSACK_CAPACITY);
    
    // 2. Run Subgradient Descent to find u* and Z_UB
    double u = 0.0;
    int64 Z_UB = std::numeric_limits<int64>::max();
    double step_alpha = 2.0;
    
    std::vector<double> full_reduced_costs(FULL_NUM_ITEMS);

    for (size_t iter = 0; iter < LAGRANGIAN_ITERATIONS; ++iter) {
        auto [sub_val, sub_weight] = lagrangianSubproblem(u, FULL_ITEM_WEIGHTS, FULL_ITEM_VALUES);
        int64 Z_u = sub_val + (int64)(u * FULL_KNAPSACK_CAPACITY);
        if (Z_u < Z_UB) Z_UB = Z_u;
        
        int64 g = sub_weight - FULL_KNAPSACK_CAPACITY;
        if (g == 0) break;

        double step = step_alpha * (Z_u - GLOBAL_LOWER_BOUND) / (double)(g * g);
        u = std::max(0.0, u - step * g);
        if(iter % 10 == 0) step_alpha *= 0.9;
    }
    
    optimal_lagrangian_multiplier = u;

    // 3. Variable Fixing (Part 1.4)
    fixed_items_value = 0;
    fixed_items_weight = 0;
    std::vector<int64> core_weights, core_values;

    for (size_t i = 0; i < FULL_NUM_ITEMS; ++i) {
        double surplus = FULL_ITEM_VALUES[i] - u * FULL_ITEM_WEIGHTS[i];
        
        // Z_UB - surplus < Z_LB  => Fix to 1
        if (Z_UB - surplus < GLOBAL_LOWER_BOUND) {
            fixed_items_value += FULL_ITEM_VALUES[i];
            fixed_items_weight += FULL_ITEM_WEIGHTS[i];
            fixed_one_items.push_back(i);
        }
        // Z_UB + surplus < Z_LB  => Fix to 0
        else if (Z_UB + surplus < GLOBAL_LOWER_BOUND) {
            // Fix to 0: do nothing
        }
        // Cannot fix: add to core problem
        else {
            core_weights.push_back(FULL_ITEM_WEIGHTS[i]);
            core_values.push_back(FULL_ITEM_VALUES[i]);
            core_to_original_index_map.push_back(i);
        }
    }

    // 4. Set globals to the new CORE problem (N')
    ITEM_WEIGHTS = std::move(core_weights);
    ITEM_VALUES = std::move(core_values);
    NUM_ITEMS = ITEM_WEIGHTS.size(); // N'
    KNAPSACK_CAPACITY = FULL_KNAPSACK_CAPACITY - fixed_items_weight;

    if (KNAPSACK_CAPACITY < 0) {
        // Pathological case: over-fixed.
        NUM_ITEMS = 0;
        ITEM_WEIGHTS.clear();
        ITEM_VALUES.clear();
    }
    
    // 5. Build heuristic data for the CORE problem
    REDUCED_COSTS.resize(NUM_ITEMS);
    ITEM_CLASSIFICATION.resize(NUM_ITEMS);
    v_w_order.resize(NUM_ITEMS);
    rcbo_order.resize(NUM_ITEMS);
    value_order.resize(NUM_ITEMS);
    weight_order.resize(NUM_ITEMS);

    double core_count = 0;
    for (size_t i = 0; i < NUM_ITEMS; ++i) {
        REDUCED_COSTS[i] = ITEM_VALUES[i] - u * ITEM_WEIGHTS[i];
        double ratio = (ITEM_WEIGHTS[i] > 0) ? (double)ITEM_VALUES[i] / ITEM_WEIGHTS[i] : 0;
        
        // Populate sorted lists
        v_w_order[i] = ItemProperty{(int)i, ratio, ITEM_WEIGHTS[i], ITEM_VALUES[i]};
        rcbo_order[i] = ItemProperty{(int)i, REDUCED_COSTS[i], ITEM_WEIGHTS[i], ITEM_VALUES[i]};
        value_order[i] = ItemProperty{(int)i, (double)ITEM_VALUES[i], ITEM_WEIGHTS[i], ITEM_VALUES[i]};
        weight_order[i] = ItemProperty{(int)i, (double)-ITEM_WEIGHTS[i], ITEM_WEIGHTS[i], ITEM_VALUES[i]}; // Negative for ascending

        // Classify items (Part 5.2)
        if (Z_UB - REDUCED_COSTS[i] < GLOBAL_LOWER_BOUND) ITEM_CLASSIFICATION[i] = X1_HIGH;
        else if (Z_UB + REDUCED_COSTS[i] < GLOBAL_LOWER_BOUND) ITEM_CLASSIFICATION[i] = X0_LOW;
        else { ITEM_CLASSIFICATION[i] = CORE; core_count++; }
    }
    
    // Sort heuristic lists
    std::sort(v_w_order.begin(), v_w_order.end());
    std::sort(rcbo_order.begin(), rcbo_order.end());
    std::sort(value_order.begin(), value_order.end());
    std::sort(weight_order.begin(), weight_order.end());
    
    // Set mutation probabilities (Part 5.2)
    if (core_count == 0) core_count = 1;
    double non_core_count = NUM_ITEMS - core_count;
    if (non_core_count == 0) non_core_count = 1;
    CORE_MUTATION_PROB = (NUM_ITEMS * BASE_MUTATION_RATE) / (core_count + non_core_count / 100.0);
    NON_CORE_MUTATION_PROB = CORE_MUTATION_PROB / 100.0;
    if (CORE_MUTATION_PROB > 1.0) CORE_MUTATION_PROB = 1.0;
}


// --- 2. Evolutionary Core Functions (on Sparse Chromosomes) ---

// Creates an individual from a sorted item list (Part 3.1)
Individual createIndividualFromOrder(const std::vector<ItemProperty>& order) {
    Individual ind;
    for (const auto& item : order) {
        if (ind.totalWeight + item.weight <= KNAPSACK_CAPACITY) {
            ind.add(item.id);
            ind.totalWeight += item.weight;
            ind.totalValue += item.value;
        }
    }
    ind.fitnessValid = false;
    return ind;
}

// Generates the initial population using Heuristic Multi-Seeding (Part 3)
std::vector<std::vector<Individual>> generateInitialPopulation() {
    std::vector<std::vector<Individual>> islands(NUM_ISLANDS);
    
    // Create seed individuals
    std::vector<Individual> seeds;
    seeds.push_back(createIndividualFromOrder(v_w_order));
    seeds.push_back(createIndividualFromOrder(rcbo_order));
    seeds.push_back(createIndividualFromOrder(value_order));
    seeds.push_back(createIndividualFromOrder(weight_order));

    for (size_t i = 0; i < NUM_ISLANDS; ++i) {
        islands[i].resize(ISLAND_SIZE);
        for (size_t j = 0; j < ISLAND_SIZE; ++j) {
            // Fill 50% with mutated seeds, 50% with seeds
            if (j < seeds.size()) {
                islands[i][j] = seeds[j];
            } else {
                islands[i][j] = seeds[j % seeds.size()];
                islands[i][j].mutate();
                islands[i][j].repair();
            }
        }
    }
    return islands;
}

// Lagrangian Greedy Crossover for SPARSE chromosomes (Part 5.1)
void lagrangianGreedyCrossover(const Individual &p1, const Individual &p2,
                               Individual &c1, Individual &c2) {
    // Pool items from both parents O(k)
    std::set<size_t> pool;
    pool.insert(p1.selectedItemIndices.begin(), p1.selectedItemIndices.end());
    pool.insert(p2.selectedItemIndices.begin(), p2.selectedItemIndices.end());

    // Get properties from RCBO order O(k)
    std::vector<ItemProperty> pool_items;
    for (size_t item_index : pool) {
        pool_items.push_back(rcbo_order[item_index]);
    }

    // Sort pool by reduced cost O(k log k)
    std::sort(pool_items.begin(), pool_items.end());

    // Build Child 1 (from pool) O(k)
    c1 = Individual();
    for (const auto& item : pool_items) {
        if (c1.totalWeight + item.weight <= KNAPSACK_CAPACITY) {
            c1.add(item.id);
            c1.totalWeight += item.weight;
            c1.totalValue += item.value;
        }
    }
    // Fill Child 1 (from global RCBO) O(N' log k)
    for (const auto& item : rcbo_order) {
        if (c1.totalWeight + item.weight > KNAPSACK_CAPACITY) continue;
        auto it = std::lower_bound(c1.selectedItemIndices.begin(), c1.selectedItemIndices.end(), item.id);
        if (it == c1.selectedItemIndices.end() || *it != (size_t)item.id) {
            c1.selectedItemIndices.insert(it, item.id);
            c1.totalWeight += item.weight;
            c1.totalValue += item.value;
        }
    }
    c1.fitnessValid = false;

    // Build Child 2 (symmetric)
    c2 = c1;
}

// Generates the next generation for a single island using DC (Part 6.2)
void nextGeneration(std::vector<Individual>& currentPop, std::vector<Individual>& nextPop) {
    
    std::vector<size_t> indices(ISLAND_SIZE);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    // No elitism, as per DC
    for (size_t i = 0; i < ISLAND_SIZE; i += 2) {
        if (i + 1 == ISLAND_SIZE) { // Odd population size
            nextPop[indices[i]] = currentPop[indices[i]];
            break; 
        }

        // 1. Select parents
        const Individual& p1 = currentPop[indices[i]];
        const Individual& p2 = currentPop[indices[i+1]];

        // 2. Create children
        Individual c1, c2;
        lagrangianGreedyCrossover(p1, p2, c1, c2); // c1 and c2 are identical here

        // 3. Mutate
        std::uniform_real_distribution<double> prob(0.0, 1.0);
        if(prob(rng) < BASE_MUTATION_RATE) c1.mutate();
        if(prob(rng) < BASE_MUTATION_RATE) c2.mutate();

        // 4. Local Search (Memetic Step)
        c1.localSearch();
        c2.localSearch();
        // Repair is handled inside localSearch and at end

        // 5. Final Repair (SA might leave it slightly invalid)
        c1.repair();
        c2.repair();

        // 6. Deterministic Crowding Replacement
        size_t d_p1c1 = hammingDist(p1, c1);
        size_t d_p2c2 = hammingDist(p2, c2);
        size_t d_p1c2 = hammingDist(p1, c2);
        size_t d_p2c1 = hammingDist(p2, c1);

        if (d_p1c1 + d_p2c2 < d_p1c2 + d_p2c1) {
            nextPop[indices[i]]   = (c1.getFitness() > p1.getFitness()) ? c1 : p1;
            nextPop[indices[i+1]] = (c2.getFitness() > p2.getFitness()) ? c2 : p2;
        } else {
            nextPop[indices[i]]   = (c2.getFitness() > p1.getFitness()) ? c2 : p1;
            nextPop[indices[i+1]] = (c1.getFitness() > p2.getFitness()) ? c1 : p2;
        }
    }
}

// Migration function for Island Model
void migrate(std::vector<std::vector<Individual>>& islands) {
    for (size_t i = 0; i < NUM_ISLANDS; ++i) {
        // Find best in island i
        auto best_it = std::max_element(islands[i].begin(), islands[i].end(),
            [](const Individual& a, const Individual& b) {
                return a.getFitness() < b.getFitness();
            });
        
        // Find worst in island (i+1) % N
        size_t target_island = (i + 1) % NUM_ISLANDS;
        auto worst_it = std::min_element(islands[target_island].begin(), islands[target_island].end(),
            [](const Individual& a, const Individual& b) {
                return a.getFitness() < b.getFitness();
            });

        // Replace worst in target with best from source
        *worst_it = *best_it;
    }
}

// Main Memetic Algorithm function
Result solveKnapsackMemetic() {
    Result result;
    auto start = std::chrono::high_resolution_clock::now();

    // 1. Preprocessing (Part 1)
    preprocess();

    // 2. Initialize Islands (Part 3)
    std::vector<std::vector<Individual>> islands = generateInitialPopulation();
    std::vector<std::vector<Individual>> nextIslands = islands; // Pre-allocate

    // 3. Evolve the population
    for (size_t gen = 0; gen < MAX_GENERATIONS; ++gen) {
        for(size_t i = 0; i < NUM_ISLANDS; ++i) {
            nextGeneration(islands[i], nextIslands[i]);
        }
        islands.swap(nextIslands);

        if (gen > 0 && gen % MIGRATION_INTERVAL == 0) {
            migrate(islands);
        }
    }

    // 4. Find the best individual across all islands
    Individual bestOverall;
    int64 maxFitness = -1;
    for (const auto& island : islands) {
        for (const auto& ind : island) {
            if (ind.getFitness() > maxFitness) {
                maxFitness = ind.getFitness();
                bestOverall = ind;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // 5. Store results, mapping back to original problem
    result.maxValue = bestOverall.getFitness() + fixed_items_value;
    
    // Add items fixed to 1
    for (int original_index : fixed_one_items) {
        result.selectedItems.push_back(original_index);
    }
    // Add items from the core solution
    for (size_t core_index : bestOverall.selectedItemIndices) {
        result.selectedItems.push_back(core_to_original_index_map[core_index]);
    }
    std::sort(result.selectedItems.begin(), result.selectedItems.end());

    result.executionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Approximate memory usage
    size_t populationMemory = 0;
    for(const auto& island : islands) {
        for (const auto& ind : island) {
            populationMemory += sizeof(Individual) + (sizeof(size_t) * ind.selectedItemIndices.capacity());
        }
    }
    populationMemory *= 2; // For nextIslands
    
    size_t vectorMemory =
        (sizeof(int64) * (FULL_ITEM_WEIGHTS.capacity() + FULL_ITEM_VALUES.capacity())) +
        (sizeof(int64) * (ITEM_WEIGHTS.capacity() + ITEM_VALUES.capacity())) +
        (sizeof(double) * REDUCED_COSTS.capacity()) +
        (sizeof(ItemType) * ITEM_CLASSIFICATION.capacity()) +
        (sizeof(ItemProperty) * (v_w_order.capacity() + rcbo_order.capacity() + value_order.capacity() + weight_order.capacity())) +
        (sizeof(int) * result.selectedItems.capacity()) +
        (sizeof(size_t) * core_to_original_index_map.capacity()) +
        (sizeof(int) * fixed_one_items.capacity());

    result.memoryUsed = populationMemory + vectorMemory;

    return result;
}


// Parses command-line arguments
void parseArguments(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--population_size" && i + 1 < argc) {
            POPULATION_SIZE = std::stoul(argv[++i]);
        } else if (arg == "--max_generations" && i + 1 < argc) {
            MAX_GENERATIONS = std::stoul(argv[++i]);
        } else if (arg == "--mutation_rate" && i + 1 < argc) {
            BASE_MUTATION_RATE = std::stod(argv[++i]);
        } else if (arg == "--sa_temp" && i + 1 < argc) {
            BASE_SA_TEMP = std::stod(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            SEED = static_cast<unsigned int>(std::stoi(argv[++i]));
        } else if (arg == "--help" || arg == "-h") {
            std::cerr << "Usage: " << argv[0] << " [options]" << "\n";
            std::cerr << "Options:" << "\n";
            std::cerr << "  --population_size <int>    Total population size (overrides ISLAND_SIZE)" << "\n";
            std::cerr << "  --max_generations <int>    Max generations" << "\n";
            std::cerr << "  --mutation_rate <float>    Base mutation rate" << "\n";
            std::cerr << "  --sa_temp <float>          Base SA temperature" << "\n";
            std::cerr << "  --seed <unsigned int>      Seed for random number generator" << "\n";
            exit(0);
        }
    }
}

int main(int argc, char *argv[]) {
    // Use fast I/O.
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    parseArguments(argc, argv);
    rng.seed(SEED);

    std::cin >> FULL_NUM_ITEMS >> FULL_KNAPSACK_CAPACITY;

    // --- Heuristics for parameters ---
    if (POPULATION_SIZE == 0) {
        // Use default NUM_ISLANDS * ISLAND_SIZE
        POPULATION_SIZE = NUM_ISLANDS * ISLAND_SIZE;
    } else {
        // User provided total size, so adjust ISLAND_SIZE
        ISLAND_SIZE = std::max((size_t)2, POPULATION_SIZE / NUM_ISLANDS);
        POPULATION_SIZE = ISLAND_SIZE * NUM_ISLANDS; // Ensure consistency
    }

    if (MAX_GENERATIONS == 0) {
        if (FULL_NUM_ITEMS < 100) MAX_GENERATIONS = 200;
        else if (FULL_NUM_ITEMS < 1000) MAX_GENERATIONS = 100;
        else if (FULL_NUM_ITEMS < 10000) MAX_GENERATIONS = 50;
        else MAX_GENERATIONS = 30;
    }
    // --- End Heuristics ---

    FULL_ITEM_WEIGHTS.resize(FULL_NUM_ITEMS);
    FULL_ITEM_VALUES.resize(FULL_NUM_ITEMS);

    for (size_t i = 0; i < FULL_NUM_ITEMS; i++) {
        std::cin >> FULL_ITEM_WEIGHTS[i];
    }
    for (size_t i = 0; i < FULL_NUM_ITEMS; ++i) {
        std::cin >> FULL_ITEM_VALUES[i];
    }

    // Solve the knapsack problem.
    Result result = solveKnapsackMemetic();

    // Print the results in the required format.
    std::cout << result.maxValue << "\n";
    std::cout << result.selectedItems.size() << "\n";
    if (!result.selectedItems.empty()) {
        for (size_t i = 0; i < result.selectedItems.size(); ++i) {
            std::cout << result.selectedItems[i] << (i == result.selectedItems.size() - 1 ? "" : " ");
        }
        std::cout << "\n";
    }
    std::cout << result.executionTime << "\n";
    std::cout << result.memoryUsed << "\n";

    return 0;
}