#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <numeric>
#include <cstring>

using namespace std;

// Hyperparameters for the Genetic Algorithm (defaults from Python code)
int POPULATION_SIZE = 6;  // Python default: count=6
int MAX_GENERATIONS = 500;
double CROSSOVER_RATE = 0.53;
double MUTATION_RATE = 0.013;
double REPRODUCTION_RATE = 0.15;

// Global variables for problem instance
int KNAPSACK_CAPACITY;
vector<int> ITEM_WEIGHTS;
vector<int> ITEM_VALUES;
int NUM_ITEMS;

// Random number generator
random_device rd;
mt19937 rng(rd());

// A result struct to hold all output
struct Result {
    int maxValue;
    vector<int> selectedItems;
    long long executionTime; // in microseconds
    size_t memoryUsed; // in bytes (approximate)
};

// Individual representation: a bit string where each bit represents item selection
class Individual {
public:
    vector<int> bits; // 0 or 1 for each item

    Individual(int n) : bits(n, 0) {}

    Individual(const vector<int> &b) : bits(b) {}

    // Calculate fitness: total value if weight constraint is satisfied, 0 otherwise
    int fitness() const {
        int totalValue = 0;
        int totalWeight = 0;

        for (int i = 0; i < NUM_ITEMS; ++i) {
            if (bits[i] == 1) {
                totalValue += ITEM_VALUES[i];
                totalWeight += ITEM_WEIGHTS[i];
            }
        }

        // If weight exceeds capacity, fitness is 0
        if (totalWeight > KNAPSACK_CAPACITY) {
            return 0;
        }

        return totalValue;
    }

    // Get total weight of selected items
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

// Generate initial population with random individuals
vector<Individual> generateInitialPopulation() {
    vector<Individual> population;
    uniform_int_distribution<int> bitDist(0, 1);

    for (int p = 0; p < POPULATION_SIZE; ++p) {
        Individual ind(NUM_ITEMS);
        for (int i = 0; i < NUM_ITEMS; ++i) {
            ind.bits[i] = bitDist(rng);
        }
        population.push_back(ind);
    }

    return population;
}

// Tournament selection: shuffle population, pick first 4, run 2 tournaments
pair<Individual, Individual> selection(vector<Individual> &population) {
    // Shuffle the population
    shuffle(population.begin(), population.end(), rng);

    // Tournament 1: between first and second
    Individual parent1 = (population[0].fitness() > population[1].fitness())
        ? population[0] : population[1];

    // Tournament 2: between third and fourth
    Individual parent2 = (population[2].fitness() > population[3].fitness())
        ? population[2] : population[3];

    return { parent1, parent2 };
}

// Single-point crossover: split at midpoint and create two children
pair<Individual, Individual> crossover(const Individual &parent1, const Individual &parent2) {
    int midpoint = NUM_ITEMS / 2;

    Individual child1(NUM_ITEMS);
    Individual child2(NUM_ITEMS);

    // Child1: first half from parent1, second half from parent2
    for (int i = 0; i < midpoint; ++i) {
        child1.bits[i] = parent1.bits[i];
    }
    for (int i = midpoint; i < NUM_ITEMS; ++i) {
        child1.bits[i] = parent2.bits[i];
    }

    // Child2: first half from parent2, second half from parent1
    for (int i = 0; i < midpoint; ++i) {
        child2.bits[i] = parent2.bits[i];
    }
    for (int i = midpoint; i < NUM_ITEMS; ++i) {
        child2.bits[i] = parent1.bits[i];
    }

    return { child1, child2 };
}

// Mutation: randomly flip bits based on mutation rate
void mutate(vector<Individual> &individuals) {
    uniform_real_distribution<double> probDist(0.0, 1.0);

    for (auto &individual : individuals) {
        for (int i = 0; i < NUM_ITEMS; ++i) {
            if (probDist(rng) < MUTATION_RATE) {
                individual.bits[i] = 1 - individual.bits[i]; // Flip bit
            }
        }
    }
}

// Generate next generation using selection, crossover, mutation, and reproduction
vector<Individual> nextGeneration(vector<Individual> population) { // Pass by value to allow modification
    vector<Individual> nextGen;
    uniform_real_distribution<double> probDist(0.0, 1.0);

    while (nextGen.size() < static_cast<size_t>(POPULATION_SIZE)) {
        // Selection: get two parents
        auto [parent1, parent2] = selection(population);

        vector<Individual> children;

        // Reproduction: directly pass parents to next generation
        if (probDist(rng) < REPRODUCTION_RATE) {
            children.push_back(parent1);
            children.push_back(parent2);
        }
        // Crossover and/or Mutation
        else {
            // Crossover
            if (probDist(rng) < CROSSOVER_RATE) {
                auto [child1, child2] = crossover(parent1, parent2);
                children.push_back(child1);
                children.push_back(child2);
            }
            else {
                // No crossover, just copy parents
                children.push_back(parent1);
                children.push_back(parent2);
            }

            // Mutation is applied to the children (either from crossover or parents)
            mutate(children);
        }

        // Add children to next generation
        for (const auto &child : children) {
            if (nextGen.size() < static_cast<size_t>(POPULATION_SIZE)) {
                nextGen.push_back(child);
            }
        }
    }

    return nextGen;
}

// Solve knapsack using genetic algorithm
Result solveKnapsackGenetic() {
    Result result;

    auto start = chrono::high_resolution_clock::now();

    // Generate initial population
    vector<Individual> population = generateInitialPopulation();

    // Evolve through generations
    for (int gen = 0; gen < MAX_GENERATIONS; ++gen) {
        population = nextGeneration(population);
    }

    // Find the best individual in the final population
    Individual bestIndividual = *max_element(population.begin(), population.end(),
        [](const Individual &a, const Individual &b) {
            return a.fitness() < b.fitness();
        });

    auto end = chrono::high_resolution_clock::now();

    // Extract results
    result.maxValue = bestIndividual.fitness();

    for (int i = 0; i < NUM_ITEMS; ++i) {
        if (bestIndividual.bits[i] == 1) {
            result.selectedItems.push_back(i);
        }
    }

    result.executionTime = chrono::duration_cast<chrono::microseconds>(end - start).count();

    // Approximate memory: population + individuals
    size_t populationMemory = POPULATION_SIZE * NUM_ITEMS * sizeof(int);
    size_t vectorMemory = (sizeof(int) * (ITEM_WEIGHTS.size() + ITEM_VALUES.size())) +
        (sizeof(int) * result.selectedItems.size());
    result.memoryUsed = populationMemory + vectorMemory;

    return result;
}

// Parse command line arguments
void parseArguments(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

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
            cerr << "Usage: " << argv[0] << " [options]" << endl;
            cerr << "Options:" << endl;
            cerr << "  --population_size <int>      Population size (default: 6)" << endl;
            cerr << "  --max_generations <int>      Max generations (default: 500)" << endl;
            cerr << "  --crossover_rate <float>     Crossover rate (default: 0.53)" << endl;
            cerr << "  --mutation_rate <float>      Mutation rate (default: 0.013)" << endl;
            cerr << "  --reproduction_rate <float>  Reproduction rate (default: 0.15)" << endl;
            exit(0);
        }
    }
}

int main(int argc, char *argv[]) {
    // Fast I/O
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Parse command line arguments
    parseArguments(argc, argv);

    // Read input from stdin
    int n, capacity;
    cin >> n >> capacity;

    NUM_ITEMS = n;
    KNAPSACK_CAPACITY = capacity;

    ITEM_WEIGHTS.resize(n);
    ITEM_VALUES.resize(n);

    for (int i = 0; i < n; i++) {
        cin >> ITEM_WEIGHTS[i];
    }

    for (int i = 0; i < NUM_ITEMS; ++i) {
        cin >> ITEM_VALUES[i];
    }

    // Solve the knapsack problem using genetic algorithm
    Result result = solveKnapsackGenetic();

    // Output results in the required format
    cout << result.maxValue << endl;
    cout << result.selectedItems.size() << endl;
    if (!result.selectedItems.empty()) {
        for (size_t i = 0; i < result.selectedItems.size(); ++i) {
            cout << result.selectedItems[i] << (i == result.selectedItems.size() - 1 ? "" : " ");
        }
        cout << endl;
    }
    cout << result.executionTime << endl;
    cout << result.memoryUsed << endl;

    return 0;
}
