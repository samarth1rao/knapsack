#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <numeric>
#include <cstring>


// Struct to hold the results of the genetic algorithm.
struct Result {
    int maxValue;                   // The maximum value found for the knapsack.
    std::vector<int> selectedItems; // The indices of the items selected to achieve the max value.
    long long executionTime;        // Total execution time in microseconds.
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
int KNAPSACK_CAPACITY;                  // Maximum weight the knapsack can hold.
std::vector<int> ITEM_WEIGHTS;          // Weights of the items.
std::vector<int> ITEM_VALUES;           // Values of the items.
int NUM_ITEMS;                          // Total number of items.
std::vector<ItemProperty> SORTED_ITEMS; // Pre-sorted list of items by value-to-weight ratio.

// Mersenne Twister random number generator, will be seeded in main.
std::mt19937 rng;


// Class to represent an individual in the population.
// Each individual is a potential solution, represented by a bit string.
class Individual {
public:
    std::vector<char> bits;             // A bit string (0 or 1) representing item selection.
    mutable int cachedFitness = -1;     // Cached fitness value to avoid re-computation.
    mutable bool fitnessValid = false;  // Flag to check if the cached fitness is valid.

    // Constructor to create an individual with a given number of items, initialised to 0.
    Individual(int n) : bits(n, 0) {}

    // Constructor to create an individual from an existing bit string.
    Individual(const std::vector<char> &b) : bits(b) {}

    // Calculates the fitness of the individual.
    // Fitness is the total value of selected items or 0 if over capacity.
    int fitness() const {
        // Return cached fitness if it's valid.
        if (fitnessValid) {
            return cachedFitness;
        }

        // Calculate total value and weight of selected items.
        int totalValue = 0;
        int totalWeight = 0;
        for (int i = 0; i < NUM_ITEMS; ++i) {
            if (bits[i] == 1) {
                totalValue += ITEM_VALUES[i];
                totalWeight += ITEM_WEIGHTS[i];
            }
        }

        // If total weight exceeds capacity, the solution is invalid, so fitness is 0.
        if (totalWeight > KNAPSACK_CAPACITY) {
            cachedFitness = 0;
        }
        else {
            cachedFitness = totalValue;
        }

        // Cache the new fitness value.
        fitnessValid = true;
        return cachedFitness;
    }

    // Calculates the total weight of the items selected by the individual.
    int totalWeight() const {
        int weight = 0;
        for (int i = 0; i < NUM_ITEMS; ++i) {
            if (bits[i] == 1) {
                weight += ITEM_WEIGHTS[i];
            }
        }
        return weight;
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


// Repairs an individual that is overweight (total weight > knapsack capacity).
// It removes items with the lowest value-to-weight ratio until the individual is valid.
void repairIndividual(Individual &individual) {
    // Calculate the current weight of the individual.
    int currentWeight = individual.totalWeight();

    // Remove the worst items till the weight is within capacity.
    for (const auto &itemProp : SORTED_ITEMS) {
        // If the individual is within capacity, stop repair.
        if (currentWeight <= KNAPSACK_CAPACITY) {
            break;
        }
        // Remove the item if it is included in the individual.
        if (individual.bits[itemProp.id] == 1) {
            individual.bits[itemProp.id] = 0;           // Remove item by setting its bit to 0.
            currentWeight -= ITEM_WEIGHTS[itemProp.id]; // Update the current weight.
        }
    }

    // Invalidate the fitness cache as the individual has been modified.
    individual.fitnessValid = false;
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
            ind.bits[i] = bitDist(rng);
        }
        population.push_back(ind);
    }

    // Repair any individuals in the initial population that are overweight.
    for (auto &individual : population) {
        repairIndividual(individual);
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
    if (parent2->fitness() > parent1->fitness()) {
        parent1 = parent2;
    }

    // Tournament 2: Select the fitter individual between the last two.
    const Individual *parent3 = &population[idx3];
    const Individual *parent4 = &population[idx4];
    if (parent4->fitness() > parent3->fitness()) {
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

    // Invalidate fitness caches for the children.
    child1.fitnessValid = false;
    child2.fitnessValid = false;
}


// Applies mutation to an individual.
// Each bit in the individual's bit string has a chance to be flipped.
void mutate(Individual &individual) {
    // Distribution for mutation probability.
    std::uniform_real_distribution<double> probDist(0.0, 1.0);

    // Apply mutation to each bit.
    bool mutated = false;
    for (int i = 0; i < NUM_ITEMS; ++i) {
        // Flip the bit with probability MUTATION_RATE.
        if (probDist(rng) < MUTATION_RATE) {
            individual.bits[i] = 1 - individual.bits[i];
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
        [](const Individual &a, const Individual &b) { return a.fitness() < b.fitness(); });
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
            repairIndividual(child1);
            i++;

            // Mutate and repair child2 if there's space in the population.
            if (i < static_cast<size_t>(POPULATION_SIZE) && &child1 != &child2) {
                mutate(child2);
                repairIndividual(child2);
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
        [](const Individual &a, const Individual &b) { return a.fitness() < b.fitness(); });

    // Stop the timer.
    auto end = std::chrono::high_resolution_clock::now();

    // Store the results.
    result.maxValue = bestIndividual.fitness();

    // Collect the indices of the selected items.
    for (int i = 0; i < NUM_ITEMS; ++i) {
        if (bestIndividual.bits[i] == 1) {
            result.selectedItems.push_back(i);
        }
    }

    // Calculate execution time in microseconds.
    result.executionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Approximate memory usage.
    size_t populationMemory = 0;
    for (const auto &ind : population) {
        populationMemory += sizeof(Individual) + (ind.bits.capacity() * sizeof(char));
    }
    for (const auto &ind : nextPopulation) {
        populationMemory += sizeof(Individual) + (ind.bits.capacity() * sizeof(char));
    }
    size_t vectorMemory =
        (sizeof(int) * (ITEM_WEIGHTS.capacity() + ITEM_VALUES.capacity())) +
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
            std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cerr << "Options:" << std::endl;
            std::cerr << "  --population_size <int>      Population size" << std::endl;
            std::cerr << "  --max_generations <int>      Max generations" << std::endl;
            std::cerr << "  --crossover_rate <float>     Crossover rate" << std::endl;
            std::cerr << "  --mutation_rate <float>      Mutation rate" << std::endl;
            std::cerr << "  --reproduction_rate <float>  Reproduction rate" << std::endl;
            std::cerr << "  --seed <unsigned int>        Seed for random number generator" << std::endl;
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
    int n, capacity;
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
    std::cout << result.maxValue << std::endl;
    std::cout << result.selectedItems.size() << std::endl;
    if (!result.selectedItems.empty()) {
        for (size_t i = 0; i < result.selectedItems.size(); ++i) {
            std::cout << result.selectedItems[i] << (i == result.selectedItems.size() - 1 ? "" : " ");
        }
        std::cout << std::endl;
    }
    std::cout << result.executionTime << std::endl;
    std::cout << result.memoryUsed << std::endl;

    return 0;
}
