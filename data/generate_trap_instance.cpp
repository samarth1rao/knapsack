/**
 * Knapsack Problem Instance Generator (C++ Worker)
 *
 * This program generates a single, hard-to-solve 0/1 knapsack problem
 * instance and appends it as a single CSV row to a specified file.
 *
 * It creates an instance with a known optimal solution that will "trap"
 * a standard greedy (price/weight ratio) algorithm.
 *
 * COMPILE (from Python or manually):
 * g++ -O3 -std=c++17 -o generate_trap_instance generate_trap_instance.cpp
 *
 * USAGE (called by Python):
 * ./generate_trap_instance <filepath> <category> <n> <capacity> <seed>
 *
 * EXAMPLE:
 * ./generate_trap_instance data.csv Small 1000 100000 12345
 *
 * CSV ROW FORMAT:
 * category, n, "[weights,...]", "[profits,...]", capacity, "[picks,...]", best_price, seed
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>       // For std::mt19937, std::uniform_int_distribution
#include <numeric>      // For std::iota
#include <algorithm>    // For std::shuffle, std::sort
#include <sstream>      // For std::stringstream

 // Use 64-bit integers for weights, profits, and capacity
using int64 = long long;

/**
 * @brief Helper function to JSON-serialize a vector of numbers.
 * @tparam T Numeric type (int, long long)
 * @param vec Vector to serialize
 * @return String representation, e.g., "[1, 2, 3]"
 */
template<typename T>
std::string vec_to_json(const std::vector<T> &vec) {
    if (vec.empty()) {
        return "[]";
    }
    std::stringstream ss;
    ss << "[" << vec[0];
    for (size_t i = 1; i < vec.size(); ++i) {
        ss << ", " << vec[i];
    }
    ss << "]";
    return ss.str();
}

/**
 * @brief Helper function to quote a string for a CSV field.
 * @param s String to quote
 * @return CSV-safe, quoted string, e.g., "mystring" -> "\"mystring\""
 */
std::string quote(const std::string &s) {
    std::string escaped = s;
    size_t pos = 0;
    while ((pos = escaped.find('"', pos)) != std::string::npos) {
        escaped.insert(pos, 1, '"');
        pos += 2;
    }
    return "\"" + escaped + "\"";
}

int main(int argc, char *argv[]) {
    // --- 1. Parse Command-Line Arguments ---
    if (argc != 6) {
        std::cerr << "Error: Expected 5 arguments: <filepath> <category> <n> <capacity> <seed>\n";
        return 1;
    }

    std::string filepath = argv[1];
    std::string category = argv[2];
    int64 n = std::stoll(argv[3]);
    int64 capacity = std::stoll(argv[4]);
    int seed = std::stoi(argv[5]);

    if (n < 3) {
        std::cerr << "Error: n must be at least 3 to create the trap.\n";
        return 1;
    }

    // --- 2. Initialize RNG ---
    std::mt19937 rng(seed);

    // --- 3. Define the "Greedy Trap" Structure ---
    // We will create 3 special items and (n-3) filler items.
    //
    // SET B (True Optimum): 2 items that *perfectly* fill the knapsack.
    //   - p/w ratio = 1.0
    //   - Total profit = capacity
    int64 w_opt = capacity / 2;
    int64 p_opt = capacity / 2;
    int64 best_price = capacity; // We know this by construction!

    // SET A (The "Greedy Trap"): 1 item with a *better* p/w ratio.
    //   - Its weight is chosen so it *blocks* the optimal set.
    //   - A greedy algo will pick this + some filler, resulting in a
    //     sub-optimal profit.
    //   - p/w ratio = 1.0125
    int64 w_trap = (capacity * 8) / 10; // 80% of capacity
    int64 p_trap = (capacity * 81) / 100; // 81% of capacity (p/w = 81/80)

    // Check: Greedy solution profit
    // Picks trap: Profit=p_trap, Weight=w_trap.
    // Remaining capacity = capacity - w_trap = capacity * 0.2
    // Fillers (see below) have p/w < 1.0.
    // Best filler profit < 0.2 * capacity.
    // Total greedy profit < p_trap + 0.2 * capacity = 0.81*c + 0.2*c = 1.01*c
    // Wait, the filler logic needs to be tight. Let's make fillers small.

    // SET C (Filler Items): (n-3) items with p/w < 1.0
    // We'll make their weights small so they can fill remaining space.
    std::uniform_int_distribution<int64> w_fill_dist(1, capacity / 20);
    // p/w ratio will be ~0.9

    // --- 4. Generate and Place Items ---
    std::vector<int64> weights;
    weights.reserve(n);
    weights.resize(n);
    std::vector<int64> profits;
    profits.reserve(n);
    profits.resize(n);
    std::vector<int> best_picks;
    best_picks.reserve(2);

    // Get n random, unique indices to place our items
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., n-1
    std::shuffle(indices.begin(), indices.end(), rng);

    // Place Set B (True Optimum)
    int idx_opt1 = indices[0];
    int idx_opt2 = indices[1];
    weights[idx_opt1] = w_opt;
    profits[idx_opt1] = p_opt;
    weights[idx_opt2] = w_opt;
    profits[idx_opt2] = p_opt;

    best_picks.push_back(idx_opt1);
    best_picks.push_back(idx_opt2);
    std::sort(best_picks.begin(), best_picks.end()); // Store sorted

    // Place Set A (The Trap)
    int idx_trap = indices[2];
    weights[idx_trap] = w_trap;
    profits[idx_trap] = p_trap;

    // Place Set C (Fillers)
    std::uniform_int_distribution<int64> noise_dist(-2, 2);
    for (int i = 3; i < n; ++i) {
        int idx_fill = indices[i];
        int64 w = w_fill_dist(rng);

        // p = w * 0.9, with a little noise
        int64 p_noise = noise_dist(rng);
        int64 p = (w * 9) / 10 - p_noise;  // Use integer arithmetic
        p = std::max(1LL, p); // Ensure positive profit

        weights[idx_fill] = w;
        profits[idx_fill] = p;
    }

    // --- 5. Format Output Row ---
    // Note: We open in "append" mode. The file *must* exist.
    std::ofstream outfile;
    outfile.open(filepath, std::ios_base::app);

    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file: " << filepath << "\n";
        return 2;
    }

    // "category",n,"[w1,...]","[p1,...]",capacity,"[p_idx1,...]",best_price,seed
    outfile << std::nounitbuf << quote(category) << ","
        << n << ","
        << quote(vec_to_json(weights)) << ","
        << quote(vec_to_json(profits)) << ","
        << capacity << ","
        << quote(vec_to_json(best_picks)) << ","
        << best_price << ","
        << seed << "\n";

    outfile.close();
    return 0;
}
