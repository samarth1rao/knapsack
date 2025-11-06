#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <numeric>

using namespace std;

// Struct to hold item information for sorting
struct Item {
    int weight;
    int value;
    int index;
    double density; // value/weight ratio
};

// Result struct to hold all output
struct Result {
    int maxValue;
    vector<int> selectedItems;
    long long executionTime; // in microseconds
    size_t memoryUsed; // in bytes (approximate)
};

/**
 * @brief Solves the 0/1 Knapsack problem using Greedy by Density heuristic.
 * Sorts items by value/weight ratio and picks greedily until capacity is exceeded.
 * @param capacity The total capacity of the knapsack.
 * @param weights Const reference to the item weights vector.
 * @param values Const reference to the item values vector.
 * @return A Result struct containing the solution, time, and memory.
 */
Result solveKnapsackGreedy(int capacity, const vector<int>& weights, const vector<int>& values) {
    Result result;
    int n = weights.size();

    auto start = chrono::high_resolution_clock::now();

    // Create items with density calculation
    vector<Item> items(n);
    for (int i = 0; i < n; ++i) {
        items[i].weight = weights[i];
        items[i].value = values[i];
        items[i].index = i;
        // Handle zero weight case (though unlikely in valid inputs)
        items[i].density = (weights[i] > 0) ? static_cast<double>(values[i]) / weights[i] : 0.0;
    }

    // Sort items by density in descending order (highest density first)
    sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        return a.density > b.density;
    });

    // Greedily select items
    int currentWeight = 0;
    int currentValue = 0;

    for (const auto& item : items) {
        // If adding this item doesn't exceed capacity, add it
        if (currentWeight + item.weight <= capacity) {
            currentWeight += item.weight;
            currentValue += item.value;
            result.selectedItems.push_back(item.index);
        }
        // Note: In 0/1 knapsack, we can't take fractions, so we skip items that don't fit
    }

    result.maxValue = currentValue;

    // Sort selected items by index for consistent output
    sort(result.selectedItems.begin(), result.selectedItems.end());

    auto end = chrono::high_resolution_clock::now();
    result.executionTime = chrono::duration_cast<chrono::microseconds>(end - start).count();

    // Approximate memory used
    size_t itemVectorMemory = sizeof(Item) * n;
    size_t resultVectorMemory = sizeof(int) * result.selectedItems.size();
    size_t inputVectorsMemory = (sizeof(int) * weights.size()) + (sizeof(int) * values.size());
    result.memoryUsed = itemVectorMemory + resultVectorMemory + inputVectorsMemory;

    return result;
}

int main(int argc, char* argv[]) {
    // Read input from stdin
    int n, capacity;
    cin >> n >> capacity;

    vector<int> weights(n);
    vector<int> values(n);

    for (int i = 0; i < n; i++) {
        cin >> weights[i];
    }

    for (int i = 0; i < n; i++) {
        cin >> values[i];
    }

    // Solve the knapsack problem
    Result result = solveKnapsackGreedy(capacity, weights, values);

    // Output results in required format
    cout << result.maxValue << endl;
    cout << result.selectedItems.size() << endl;
    for (size_t i = 0; i < result.selectedItems.size(); ++i) {
        cout << result.selectedItems[i];
        if (i + 1 < result.selectedItems.size()) cout << " ";
    }
    cout << endl;
    cout << result.executionTime << endl;
    cout << result.memoryUsed << endl;

    return 0;
}