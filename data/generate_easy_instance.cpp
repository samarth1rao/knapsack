/**
 * Knapsack "Easy" Problem Instance Generator (C++ Worker)
 *
 * This program generates a single, "easy-to-solve" 0/1 knapsack problem
 * instance and appends it as a single CSV row to a specified file.
 *
 * An "easy" instance has a pre-selected optimal solution where the chosen
 * items have a significantly better price-to-weight ratio than other items.
 *
 * COMPILE (from Python or manually):
 * g++ -O3 -std=c++17 -o generate_easy_instance generate_easy_instance.cpp
 *
 * USAGE (called by Python):
 * ./generate_easy_instance <filepath> <category> <n> <seed>
 *
 * EXAMPLE:
 * ./generate_easy_instance data.csv Small 1000 12345
 *
 * CSV ROW FORMAT:
 * category, n, "[weights,...]", "[profits,...]", capacity, "[picks,...]", best_price, seed
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <set>

 // Use 64-bit integers for weights, profits, and capacity
using int64 = long long;

/**
 * @brief Helper function to JSON-serialize a vector of numbers.
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
    if (argc != 5) {
        std::cerr << "Error: Expected 4 arguments: <filepath> <category> <n> <seed>\n";
        return 1;
    }

    std::string filepath = argv[1];
    std::string category = "E" + std::string(argv[2]);
    int64 n = std::stoll(argv[3]);
    int seed = std::stoi(argv[4]);

    // --- 2. Initialize RNGs ---
    std::mt19937 rng(seed);

    // --- 3. Determine k (number of selected items) ---
    int64 k;
    if (n <= 50) {
        int64 upper_bound = std::min((int64)n, std::max(1LL, (int64)(n * 0.5)));
        std::uniform_int_distribution<int64> k_dist(1, upper_bound);
        k = k_dist(rng);
    }
    else {
        int64 k_max = std::max(1LL, std::min((int64)n, (int64)(n * 0.1)));
        int64 upper_bound = std::max(1LL, std::min(k_max, (int64)(n * 0.02)));
        std::uniform_int_distribution<int64> k_dist(1, upper_bound);
        k = k_dist(rng);
    }
    k = std::max(1LL, k);

    // --- 4. Select Optimal Items ---
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Partial Fisher-Yates shuffle: only shuffle first k elements
    for (int64 i = 0; i < k; ++i) {
        std::uniform_int_distribution<int64> dist(i, n - 1);
        std::swap(indices[i], indices[dist(rng)]);
    }

    std::set<int> selected_indices;
    for (int64 i = 0; i < k; ++i) {
        selected_indices.insert(indices[i]);
    }

    // --- 5. Generate Weights and Calculate Capacity ---
    std::vector<int64> weights;
    weights.reserve(n);
    weights.resize(n);
    int64 capacity = 0;
    std::mt19937 r_weights(seed ^ 0xA5A5A5A5);
    std::uniform_int_distribution<int64> w_dist(1, 100);

    for (int i = 0; i < n; ++i) {
        weights[i] = w_dist(r_weights);
        if (selected_indices.count(i)) {
            capacity += weights[i];
        }
    }

    // --- 6. Generate Correlated Prices ---
    std::vector<int64> profits;
    profits.reserve(n);
    profits.resize(n);
    int64 best_price = 0;
    const int M_LOW = 10;
    const int M_HIGH = 15;
    const int NOISE_RANGE = std::max(1, (int)(M_LOW * 0.1));

    std::mt19937 r_prices_noise(seed ^ 0x5A5A5A5A);
    std::uniform_int_distribution<int> noise_dist(-NOISE_RANGE, NOISE_RANGE);

    for (int i = 0; i < n; ++i) {
        int64 w = weights[i];
        int noise = noise_dist(r_prices_noise);
        int64 price;

        if (selected_indices.count(i)) {
            price = (w * M_HIGH) + noise;
            best_price += price;
        }
        else {
            price = (w * M_LOW) + noise;
        }
        profits[i] = std::max(1LL, price);
    }

    // --- 7. Format Output Row ---
    std::ofstream outfile;
    outfile.open(filepath, std::ios_base::app);

    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file: " << filepath << "\n";
        return 2;
    }

    std::vector<int> best_picks;
    best_picks.reserve(selected_indices.size());
    best_picks.assign(selected_indices.begin(), selected_indices.end());
    std::sort(best_picks.begin(), best_picks.end());

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
