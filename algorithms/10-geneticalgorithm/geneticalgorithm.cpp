#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <numeric>
#include <cstring>


// Genetic Algorithm hyperparameters. These can be overridden by command-line arguments.
int POPULATION_SIZE = -1;       // Size of the population. If -1, it's set based on the number of items.
int MAX_GENERATIONS = -1;       // Number of generations to evolve. If -1, it's set based on the number of items.
double CROSSOVER_RATE = 0.53;   // Probability of crossover.
double MUTATION_RATE = 0.013;   // Probability of mutation.
double REPRODUCTION_RATE = 0.15;// Probability of direct reproduction.

// Global variables for the knapsack problem instance.
int KNAPSACK_CAPACITY;          // Maximum weight the knapsack can hold.
std::vector<int> ITEM_WEIGHTS;  // Weights of the items.
std::vector<int> ITEM_VALUES;   // Values of the items.
int NUM_ITEMS;                  // Total number of items.


// Mersenne Twister random number generator, seeded with a random device.
std::random_device rd;
std::mt19937 rng(rd());


// Struct to hold the results of the genetic algorithm.
struct Result {
    int maxValue;                   // The maximum value found for the knapsack.
    std::vector<int> selectedItems; // The indices of the items selected to achieve the max value.
    long long executionTime;        // Total execution time in microseconds.
    size_t memoryUsed;              // Approximate memory usage in bytes.
};


// Represents an individual in the population.
// Each individual is a potential solution, represented by a bit string.
class Individual {
public:
    std::vector<int> bits;              // A bit string (0 or 1) representing item selection. 1 means the item is taken.
    mutable int cachedFitness = -1;     // Cached fitness value to avoid re-computation.
    mutable bool fitnessValid = false;  // Flag to check if the cached fitness is valid.

    // Constructor to create an individual with a given number of items, initialized to 0 (not selected).
    Individual(int n) : bits(n, 0) {}

    // Constructor to create an individual from an existing bit string.
    Individual(const std::vector<int> &b) : bits(b) {}

    // Calculates the fitness of the individual.
    // Fitness is the total value of selected items. If the total weight exceeds the knapsack capacity, fitness is 0.
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


// Repairs an individual that is overweight (total weight > knapsack capacity).
// It removes items with the lowest value-to-weight ratio until the individual is valid.
void repairIndividual(Individual &individual) {
    // Calculate the current weight of the individual.
    int currentWeight = individual.totalWeight();

    // If the individual is not overweight, no repair is needed.
    if (currentWeight <= KNAPSACK_CAPACITY) {
        return;
    }

    // Store selected items with their value-to-weight ratio and index.
    // Format: {value/weight ratio, index}
    std::vector<std::pair<double, int>> selectedItems;
    for (int i = 0; i < NUM_ITEMS; ++i) {
        if (individual.bits[i] == 1) {
            // Calculate value-to-weight ratio. Handle zero weight case.
            double ratio;
            if (ITEM_WEIGHTS[i] > 0) {
                ratio = static_cast<double>(ITEM_VALUES[i]) / ITEM_WEIGHTS[i];
            }
            else {
                ratio = 0.0;
            }
            selectedItems.push_back({ ratio, i });
        }
    }

    // Sort items by their value-to-weight ratio in ascending order (worst items first).
    std::sort(selectedItems.begin(), selectedItems.end());

    // Remove the worst items until the weight is within capacity.
    for (const auto &[ratio, idx] : selectedItems) {
        // If the individual is now valid, stop removing items.
        if (currentWeight <= KNAPSACK_CAPACITY) {
            break;
        }
        individual.bits[idx] = 0;           // Remove item by setting its bit to 0.
        currentWeight -= ITEM_WEIGHTS[idx]; // Update the current weight.
    }

    // Invalidate the fitness cache as the individual has been modified.
    individual.fitnessValid = false;
}


// Generates the initial population of random individuals.
std::vector<Individual> generateInitialPopulation() {
    // Init population vector and distribution for random bits.
    std::vector<Individual> population;
    std::uniform_int_distribution<int> bitDist(0, 1);   // Distribution for generating 0 or 1.

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
// Four individuals are chosen randomly, and the fittest from each pair of two becomes a parent.
std::pair<Individual, Individual> selection(const std::vector<Individual> &population) {
    // Init distribution for random indices.
    std::uniform_int_distribution<int> indexDist(0, POPULATION_SIZE - 1);

    // Select four random individuals from the population.
    int idx1 = indexDist(rng);
    int idx2 = indexDist(rng);
    int idx3 = indexDist(rng);
    int idx4 = indexDist(rng);

    // First tournament: select the fitter individual between idx1 and idx2.
    Individual parent1 = (population[idx1].fitness() > population[idx2].fitness())
        ? population[idx1] : population[idx2];

    // Second tournament: select the fitter individual between idx3 and idx4.
    Individual parent2 = (population[idx3].fitness() > population[idx4].fitness())
        ? population[idx3] : population[idx4];

    return { parent1, parent2 };
}


// Performs single-point crossover on two parent individuals to create two children.
// The bit strings of the parents are split at the midpoint and swapped.
std::pair<Individual, Individual> crossover(const Individual &parent1, const Individual &parent2) {
    // Crossover point is the midpoint of the bit string.
    int midpoint = NUM_ITEMS / 2;

    // Child 1 gets the first half from parent 1 and the second half from parent 2.
    Individual child1(NUM_ITEMS);
    for (int i = 0; i < midpoint; ++i) {
        child1.bits[i] = parent1.bits[i];
    }
    for (int i = midpoint; i < NUM_ITEMS; ++i) {
        child1.bits[i] = parent2.bits[i];
    }

    // Child 2 gets the first half from parent 2 and the second half from parent 1.
    Individual child2(NUM_ITEMS);
    for (int i = 0; i < midpoint; ++i) {
        child2.bits[i] = parent2.bits[i];
    }
    for (int i = midpoint; i < NUM_ITEMS; ++i) {
        child2.bits[i] = parent1.bits[i];
    }

    return { child1, child2 };
}


// Applies mutation to a group of individuals.
// Each bit in an individual's bit string has a chance to be flipped.
void mutate(std::vector<Individual> &individuals) {
    // Init distribution for mutation probability.
    std::uniform_real_distribution<double> probDist(0.0, 1.0);

    // Iterate over each individual in the group.
    for (auto &individual : individuals) {
        bool mutated = false;
        for (int i = 0; i < NUM_ITEMS; ++i) {
            // If a random number is less than the mutation rate, flip the bit.
            if (probDist(rng) < MUTATION_RATE) {
                individual.bits[i] = 1 - individual.bits[i];
                mutated = true;
            }
        }
        // If the individual was mutated, invalidate its fitness cache.
        if (mutated) {
            individual.fitnessValid = false;
        }
    }
}


// Generates the next generation of the population.
// Uses elitism, selection, crossover, mutation, and reproduction.
std::vector<Individual> nextGeneration(const std::vector<Individual> &population) {
    // Init next generation vector and probability distribution.
    std::vector<Individual> nextGen;
    std::uniform_real_distribution<double> probDist(0.0, 1.0);

    // Elitism: The best individual from the current population is carried over to the next generation.
    auto bestIt = max_element(
        population.begin(),
        population.end(),
        [](const Individual &a, const Individual &b) { return a.fitness() < b.fitness(); });
    nextGen.push_back(*bestIt);

    // Fill the rest of the new population.
    while (nextGen.size() < static_cast<size_t>(POPULATION_SIZE)) {
        // Select two parents for reproduction.
        auto [parent1, parent2] = selection(population);
        std::vector<Individual> children;

        // Reproduction: a small chance to pass parents directly to the next generation.
        if (probDist(rng) < REPRODUCTION_RATE) {
            children.push_back(parent1);
            children.push_back(parent2);
        }

        // Crossover and Mutation.
        else {
            // Crossover: a chance to create children by combining parents.
            if (probDist(rng) < CROSSOVER_RATE) {
                auto [child1, child2] = crossover(parent1, parent2);
                children.push_back(child1);
                children.push_back(child2);
            }
            // If no crossover, the parents are just copied.
            else {
                children.push_back(parent1);
                children.push_back(parent2);
            }

            // Mutate the children.
            mutate(children);
        }

        // Repair any children that are overweight.
        for (auto &child : children) {
            repairIndividual(child);
        }

        // Add the new children to the next generation until it's full.
        for (const auto &child : children) {
            if (nextGen.size() < static_cast<size_t>(POPULATION_SIZE)) {
                nextGen.push_back(child);
            }
        }
    }

    return nextGen;
}


// Main Genetic Algorithm function to solve the knapsack problem.
Result solveKnapsackGenetic() {
    // Init result struct.
    Result result;

    // Start the timer.
    auto start = std::chrono::high_resolution_clock::now();

    // Create the initial population.
    std::vector<Individual> population = generateInitialPopulation();

    // Evolve the population over a number of generations.
    for (int gen = 0; gen < MAX_GENERATIONS; ++gen) {
        population = nextGeneration(population);
    }

    // Find the best individual in the final population.
    Individual bestIndividual = *max_element(population.begin(), population.end(),
        [](const Individual &a, const Individual &b) {
            return a.fitness() < b.fitness();
        });

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
    size_t populationMemory = POPULATION_SIZE * NUM_ITEMS * sizeof(int);
    size_t vectorMemory = (sizeof(int) * (ITEM_WEIGHTS.size() + ITEM_VALUES.size())) +
        (sizeof(int) * result.selectedItems.size());
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
        else if (arg == "--help" || arg == "-h") {
            std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cerr << "Options:" << std::endl;
            std::cerr << "  --population_size <int>      Population size" << std::endl;
            std::cerr << "  --max_generations <int>      Max generations" << std::endl;
            std::cerr << "  --crossover_rate <float>     Crossover rate" << std::endl;
            std::cerr << "  --mutation_rate <float>      Mutation rate" << std::endl;
            std::cerr << "  --reproduction_rate <float>  Reproduction rate" << std::endl;
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

    // Read problem instance from standard input.
    int n, capacity;
    std::cin >> n >> capacity;

    NUM_ITEMS = n;
    KNAPSACK_CAPACITY = capacity;

    // If population size is not provided, compute it using a heuristic.
    if (POPULATION_SIZE == -1) {
        if (n < 100) POPULATION_SIZE = 20;
        else if (n < 1000) POPULATION_SIZE = 50;
        else if (n < 10000) POPULATION_SIZE = 100;
        else POPULATION_SIZE = 150;
    }

    // If max generations is not provided, compute it using a heuristic.
    if (MAX_GENERATIONS == -1) {
        if (n < 100) MAX_GENERATIONS = 200;
        else if (n < 1000) MAX_GENERATIONS = 100;
        else if (n < 10000) MAX_GENERATIONS = 50;
        else MAX_GENERATIONS = 30;
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
