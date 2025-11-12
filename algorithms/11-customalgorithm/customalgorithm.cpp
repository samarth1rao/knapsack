#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <numeric>
#include <cstring>

// Use int64 for large numbers
using int64 = long long;

// Struct to hold the results of the genetic algorithm.
struct Result {
    int64 maxValue;                 // The maximum value found for the knapsack.
    std::vector<int> selectedItems; // The indices of the items selected to achieve the max value.
    int64 executionTime;            // Total execution time in microseconds.
    size_t memoryUsed;              // Approximate memory usage in bytes.
};

// Struct to hold item properties for sorting.
struct ItemProperty {
    int id;
    double ratio;
};


// Genetic Algorithm hyperparameters. These can be overridden by command-line arguments.
int POPULATION_SIZE = -1;                   // Size of the population. If -1, set heuristically.
int MAX_GENERATIONS = -1;                   // Number of generations. If -1, set heuristically.
double CROSSOVER_RATE = 0.53;               // Probability of crossover.
double MUTATION_RATE = 0.013;               // Probability of mutation.
double REPRODUCTION_RATE = 0.15;            // Probability of direct reproduction.
unsigned int SEED = std::random_device()(); // Seed for random number generator.

// Global variables for the knapsack problem instance.
int64 KNAPSACK_CAPACITY;                // Maximum weight the knapsack can hold.
std::vector<int64> ITEM_WEIGHTS;        // Weights of the items.
std::vector<int64> ITEM_VALUES;         // Values of the items.
int NUM_ITEMS;                          // Total number of items.
std::vector<ItemProperty> SORTED_ITEMS; // Pre-sorted list of items by value-to-weight ratio.

// Mersenne Twister random number generator, will be seeded in main.
std::mt19937 rng;


// Class to represent an individual in the population.
// Each individual is a potential solution, represented by a bit string.
class Individual {
public:
    std::vector<bool> bits;             // A bit string representing item selection.
    int64 totalValue = 0;               // Cached total value of selected items.
    int64 totalWeight = 0;              // Cached total weight of selected items.
    mutable int64 cachedFitness = -1;   // Cached fitness value to avoid re-computation.
    mutable bool fitnessValid = false;  // Flag to check if the cached fitness is valid.

    // Constructor to create an individual with a given number of items, initialised to false.
    Individual(int n) : bits(n, false) {}

    // Calculates total value and weight from scratch. To be called after major changes.
    void calculateMetrics() {
        totalValue = 0;
        totalWeight = 0;
        for (int i = 0; i < NUM_ITEMS; ++i) {
            if (bits[i]) {
                totalValue += ITEM_VALUES[i];
                totalWeight += ITEM_WEIGHTS[i];
            }
        }
        // Invalidate fitness cache as metrics have been recalculated.
        fitnessValid = false;
    }

    // Calculates the fitness of the individual.
    // Fitness is the total value of selected items or 0 if over capacity.
    int64 getFitness() const {
        // Return cached fitness if it's valid.
        if (fitnessValid) {
            return cachedFitness;
        }
        // Use pre-computed metrics to determine fitness.
        if (totalWeight > KNAPSACK_CAPACITY) {
            cachedFitness = 0;
        }
        else {
            cachedFitness = totalValue;
        }
        // Cache the fitness value.
        fitnessValid = true;
        return cachedFitness;
    }

    // Repairs an individual that is overweight (total weight > knapsack capacity).
    // It removes items with the lowest value-to-weight ratio until the individual is valid.
    void repair() {
        // Repaired flag.
        bool repaired = false;
        // Remove the worst items till the weight is within capacity.
        for (const auto &itemProp : SORTED_ITEMS) {
            // If the individual is within capacity, stop repair.
            if (totalWeight <= KNAPSACK_CAPACITY) {
                break;
            }
            // Remove the item if it is included in the individual.
            if (bits[itemProp.id]) {
                bits[itemProp.id] = false;                  // Remove item by setting its bit to false.
                totalWeight -= ITEM_WEIGHTS[itemProp.id];   // Update the total weight.
                totalValue -= ITEM_VALUES[itemProp.id];     // Update the total value.
                repaired = true;                            // Set repaired flag.
            }
        }
        // Invalidate the fitness cache if the individual has been modified.
        if (repaired) {
            fitnessValid = false;
        }
    }
};


// Pre-sorts items by their value-to-weight ratio in ascending order.
void preSortItems() {
    // Resize SORTED_ITEMS.
    SORTED_ITEMS.resize(NUM_ITEMS);

    // Compute value-to-weight ratio for each item. Handle zero weight case.
    for (int i = 0; i < NUM_ITEMS; ++i) {
        SORTED_ITEMS[i].id = i;
        if (ITEM_WEIGHTS[i] > 0) {
            SORTED_ITEMS[i].ratio = static_cast<double>(ITEM_VALUES[i]) / ITEM_WEIGHTS[i];
        }
        else if (ITEM_VALUES[i] > 0) {
            SORTED_ITEMS[i].ratio = std::numeric_limits<double>::infinity();
        }
        else {
            SORTED_ITEMS[i].ratio = 0.0;
        }
    }

    // Sort items by their value-to-weight ratio in ascending order.
    std::sort(
        SORTED_ITEMS.begin(),
        SORTED_ITEMS.end(),
        [](const ItemProperty &a, const ItemProperty &b) {
            return a.ratio < b.ratio;
        }
    );
}


// Generates the initial population of random individuals.
std::vector<Individual> generateInitialPopulation() {
    // Init population vector.
    std::vector<Individual> population;
    population.reserve(POPULATION_SIZE);
    // Distribution for generating random bits.
    std::uniform_int_distribution<int> bitDist(0, 1);

    // Create POPULATION_SIZE individuals with random bit strings.
    for (int p = 0; p < POPULATION_SIZE; ++p) {
        Individual ind(NUM_ITEMS);
        for (int i = 0; i < NUM_ITEMS; ++i) {
            ind.bits[i] = (bitDist(rng) == 1);
        }
        // Calculate metrics once from scratch.
        ind.calculateMetrics();
        population.push_back(ind);
    }

    // Repair any individuals in the initial population that are overweight.
    for (auto &individual : population) {
        individual.repair();
    }

    return population;
}


// Selects two parent individuals from the population using tournament selection.
std::pair<const Individual *, const Individual *> selection(const std::vector<Individual> &population) {
    // Distribution for generating random indices.
    std::uniform_int_distribution<int> indexDist(0, POPULATION_SIZE - 1);

    // Select four random individuals from the population.
    int idx1 = indexDist(rng);
    int idx2 = indexDist(rng);
    int idx3 = indexDist(rng);
    int idx4 = indexDist(rng);

    // Ensure distinct indices (individuals) for tournament selection.
    while (idx2 == idx1) {
        idx2 = (idx2 + 1) % POPULATION_SIZE;
    }
    while (idx3 == idx1 || idx3 == idx2) {
        idx3 = (idx3 + 1) % POPULATION_SIZE;
    }
    while (idx4 == idx1 || idx4 == idx2 || idx4 == idx3) {
        idx4 = (idx4 + 1) % POPULATION_SIZE;
    }

    // Tournament 1: Select the fitter individual between the first two.
    const Individual *parent1 = &population[idx1];
    const Individual *parent2 = &population[idx2];
    if (parent2->getFitness() > parent1->getFitness()) {
        parent1 = parent2;
    }

    // Tournament 2: Select the fitter individual between the last two.
    const Individual *parent3 = &population[idx3];
    const Individual *parent4 = &population[idx4];
    if (parent4->getFitness() > parent3->getFitness()) {
        parent3 = parent4;
    }

    return { parent1, parent3 };
}


// Performs single-point crossover on two parent individuals to create two children.
// The bit strings of the parents are split at the midpoint and swapped.
void crossover(const Individual &parent1, const Individual &parent2,
    Individual &child1, Individual &child2) {
    // Crossover point is the midpoint of the bit string.
    int midpoint = NUM_ITEMS / 2;

    // Child 1: First half from parent1, second half from parent2.
    std::copy(parent1.bits.begin(), parent1.bits.begin() + midpoint, child1.bits.begin());
    std::copy(parent2.bits.begin() + midpoint, parent2.bits.end(), child1.bits.begin() + midpoint);

    // Child 2: First half from parent2, second half from parent1.
    std::copy(parent2.bits.begin(), parent2.bits.begin() + midpoint, child2.bits.begin());
    std::copy(parent1.bits.begin() + midpoint, parent1.bits.end(), child2.bits.begin() + midpoint);

    // Recalculate metrics from scratch for the children.
    child1.calculateMetrics();
    child2.calculateMetrics();
}


// Applies mutation to an individual.
// Each bit in the individual's bit string has a chance to be flipped.
void mutate(Individual &individual) {
    // Distribution for mutation probability.
    std::uniform_real_distribution<double> probDist(0.0, 1.0);

    // Mutation flag.
    bool mutated = false;
    // Apply mutation to each bit.
    for (int i = 0; i < NUM_ITEMS; ++i) {
        // Flip the bit with probability MUTATION_RATE and update metrics accordingly.
        if (probDist(rng) < MUTATION_RATE) {
            // Flipping true-> false, removing item.
            if (individual.bits[i]) {
                individual.totalValue -= ITEM_VALUES[i];
                individual.totalWeight -= ITEM_WEIGHTS[i];
            }
            // Flipping false-> true, adding item.
            else {
                individual.totalValue += ITEM_VALUES[i];
                individual.totalWeight += ITEM_WEIGHTS[i];
            }
            // Flip the bit and set mutated flag.
            individual.bits[i].flip();
            mutated = true;
        }
    }

    // Invalidate fitness cache if mutation occurred.
    if (mutated) {
        individual.fitnessValid = false;
    }
}


// Generates the next generation of the population and swaps it with the current population.
// Uses elitism, selection followed by reproduction or crossover, and mutation.
void nextGeneration(const std::vector<Individual> &currentPop, std::vector<Individual> &nextPop) {
    std::uniform_real_distribution<double> probDist(0.0, 1.0);

    // Elitism: The best individual from the current population is carried over to the next generation.
    auto bestIt = max_element(
        currentPop.begin(),
        currentPop.end(),
        [](const Individual &a, const Individual &b) { return a.getFitness() < b.getFitness(); });
    nextPop[0] = *bestIt;

    // Fill the rest of the next population.
    for (size_t i = 1; i < static_cast<size_t>(POPULATION_SIZE); ) {
        // Select two parents using tournament selection.
        auto [parent1, parent2] = selection(currentPop);

        // Decide whether to reproduce directly or create offspring.
        if (probDist(rng) < REPRODUCTION_RATE) {
            // Reproduction: Directly copy parents to the next population.
            nextPop[i++] = *parent1;
            if (i < static_cast<size_t>(POPULATION_SIZE)) {
                nextPop[i++] = *parent2;
            }
        }
        else {
            // Select two children slots in the next population.
            Individual &child1 = nextPop[i];
            Individual &child2 = (i + 1 < static_cast<size_t>(POPULATION_SIZE)) ? nextPop[i + 1] : child1;

            // Decide whether to perform crossover.
            if (probDist(rng) < CROSSOVER_RATE) {
                // Crossover: Create two children from the parents.
                crossover(*parent1, *parent2, child1, child2);
            }
            else {
                // No crossover: Children are copies of the parents.
                child1 = *parent1;
                if (i + 1 < static_cast<size_t>(POPULATION_SIZE)) {
                    child2 = *parent2;
                }
            }

            // Mutate and repair child1.
            mutate(child1);
            child1.repair();
            i++;

            // Mutate and repair child2 if there's space in the population.
            if (i < static_cast<size_t>(POPULATION_SIZE) && &child1 != &child2) {
                mutate(child2);
                child2.repair();
                i++;
            }
        }
    }
}


// Main Genetic Algorithm function to solve the knapsack problem.
Result solveKnapsackGenetic() {
    // Init result struct.
    Result result;

    // Start the timer.
    auto start = std::chrono::high_resolution_clock::now();

    // Pre-sort items by value-to-weight ratio.
    preSortItems();

    // Create the initial population.
    std::vector<Individual> population = generateInitialPopulation();
    std::vector<Individual> nextPopulation(POPULATION_SIZE, Individual(NUM_ITEMS));

    // Evolve the population over generations.
    for (int gen = 0; gen < MAX_GENERATIONS; ++gen) {
        nextGeneration(population, nextPopulation);
        population.swap(nextPopulation);
    }

    // Find the best individual in the final population.
    Individual bestIndividual = *max_element(
        population.begin(),
        population.end(),
        [](const Individual &a, const Individual &b) { return a.getFitness() < b.getFitness(); });

    // Stop the timer.
    auto end = std::chrono::high_resolution_clock::now();

    // Store the results.
    result.maxValue = bestIndividual.getFitness();

    // Collect the indices of the selected items.
    for (int i = 0; i < NUM_ITEMS; ++i) {
        if (bestIndividual.bits[i]) {
            result.selectedItems.push_back(i);
        }
    }

    // Calculate execution time in microseconds.
    result.executionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Approximate memory usage.
    size_t populationMemory = 0;
    for (const auto &ind : population) {
        populationMemory += sizeof(Individual) + ((ind.bits.capacity() + 7) / 8);
    }
    for (const auto &ind : nextPopulation) {
        populationMemory += sizeof(Individual) + ((ind.bits.capacity() + 7) / 8);
    }
    size_t vectorMemory =
        (sizeof(int64) * (ITEM_WEIGHTS.capacity() + ITEM_VALUES.capacity())) +
        (sizeof(int) * result.selectedItems.capacity()) +
        (sizeof(ItemProperty) * SORTED_ITEMS.capacity());
    result.memoryUsed = populationMemory + vectorMemory;

    return result;
}

// Parses command-line arguments to override default hyperparameters.
void parseArguments(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--population_size" && i + 1 < argc) {
            POPULATION_SIZE = atoi(argv[++i]);
        }
        else if (arg == "--max_generations" && i + 1 < argc) {
            MAX_GENERATIONS = atoi(argv[++i]);
        }
        else if (arg == "--crossover_rate" && i + 1 < argc) {
            CROSSOVER_RATE = atof(argv[++i]);
        }
        else if (arg == "--mutation_rate" && i + 1 < argc) {
            MUTATION_RATE = atof(argv[++i]);
        }
        else if (arg == "--reproduction_rate" && i + 1 < argc) {
            REPRODUCTION_RATE = atof(argv[++i]);
        }
        else if (arg == "--seed" && i + 1 < argc) {
            SEED = static_cast<unsigned int>(atoi(argv[++i]));
        }
        else if (arg == "--help" || arg == "-h") {
            std::cerr << "Usage: " << argv[0] << " [options]" << "\n";
            std::cerr << "Options:" << "\n";
            std::cerr << "  --population_size <int>      Population size" << "\n";
            std::cerr << "  --max_generations <int>      Max generations" << "\n";
            std::cerr << "  --crossover_rate <float>     Crossover rate" << "\n";
            std::cerr << "  --mutation_rate <float>      Mutation rate" << "\n";
            std::cerr << "  --reproduction_rate <float>  Reproduction rate" << "\n";
            std::cerr << "  --seed <unsigned int>        Seed for random number generator" << "\n";
            exit(0);
        }
    }
}

int main(int argc, char *argv[]) {
    // Use fast I/O.
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Parse command-line arguments for hyperparameters.
    parseArguments(argc, argv);

    // Seed the random number generator.
    rng.seed(SEED);

    // Read problem instance from standard input.
    int n;
    int64 capacity;
    std::cin >> n >> capacity;

    // Set instance variables.
    NUM_ITEMS = n;
    KNAPSACK_CAPACITY = capacity;

    // If population size is not provided, compute it using a heuristic.
    if (POPULATION_SIZE == -1) {
        if (n < 100) {
            POPULATION_SIZE = 20;
        }
        else if (n < 1000) {
            POPULATION_SIZE = 50;
        }
        else if (n < 10000) {
            POPULATION_SIZE = 100;
        }
        else {
            POPULATION_SIZE = 150;
        }
    }

    // If max generations is not provided, compute it using a heuristic.
    if (MAX_GENERATIONS == -1) {
        if (n < 100) {
            MAX_GENERATIONS = 200;
        }
        else if (n < 1000) {
            MAX_GENERATIONS = 100;
        }
        else if (n < 10000) {
            MAX_GENERATIONS = 50;
        }
        else {
            MAX_GENERATIONS = 30;
        }
    }

    // Resize vectors to hold item data.
    ITEM_WEIGHTS.resize(n);
    ITEM_VALUES.resize(n);

    // Read item weights.
    for (int i = 0; i < n; i++) {
        std::cin >> ITEM_WEIGHTS[i];
    }

    // Read item values.
    for (int i = 0; i < NUM_ITEMS; ++i) {
        std::cin >> ITEM_VALUES[i];
    }

    // Solve the knapsack problem.
    Result result = solveKnapsackGenetic();

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
