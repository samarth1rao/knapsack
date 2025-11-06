#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <numeric> // For std::iota

using namespace std;

// A result struct to hold all output, matching your desired format
struct Result {
    int maxValue;
    vector<int> selectedItems;
    long long executionTime; // in microseconds
    size_t memoryUsed; // in bytes (approximate)
};

/**
 * @brief Solves the 0/1 Knapsack problem using bottom-up Dynamic Programming (Tabulation).
 * @param capacity The total capacity of the knapsack.
 * @param weights Const reference to the item weights vector.
 * @param values Const reference to the item values vector.
 * @return A Result struct containing the solution, time, and memory.
 */
Result solveKnapsackDP(int capacity, const vector<int> &weights, const vector<int> &values) {
    Result result;
    int n = weights.size();

    auto start = chrono::high_resolution_clock::now();

    // Create the DP table. dp[i][w] will store the max value using
    // the first 'i' items with a capacity of 'w'.
    // We use (n+1) and (capacity+1) to handle 0-based indexing easily.
    vector<vector<int>> dp(n + 1, vector<int>(capacity + 1, 0));

    // Fill the table
    for (int i = 1; i <= n; ++i) {
        // Get the weight and value of the *current* item (i-1 for 0-based index)
        int currentWeight = weights[i - 1];
        int currentValue = values[i - 1];

        for (int w = 0; w <= capacity; ++w) {
            // Case 1: The current item is too heavy to fit (w < currentWeight)
            // So, we can't include it. The max value is the same as without it.
            if (currentWeight > w) {
                dp[i][w] = dp[i - 1][w];
            }
            // Case 2: The current item fits.
            // We must decide: is it better to include it or exclude it?
            else {
                // Value if we *exclude* the item
                int excludeValue = dp[i - 1][w];

                // Value if we *include* the item
                // This is the item's value + the max value from the remaining capacity
                int includeValue = currentValue + dp[i - 1][w - currentWeight];

                // Take the maximum of the two choices
                dp[i][w] = max(includeValue, excludeValue);
            }
        }
    }

    // The final answer is in the bottom-right corner
    result.maxValue = dp[n][capacity];

    // --- Backtrack to find the selected items ---
    int w = capacity; // Start at the final capacity
    for (int i = n; i > 0; --i) {
        // Compare the current cell with the one directly above it
        // If they are the same, it means item 'i' was *not* included
        if (dp[i][w] == dp[i - 1][w]) {
            // Item (i-1) was not included, just move up
        }
        // If they are different, it means item 'i' *was* included
        else {
            result.selectedItems.push_back(i - 1); // Add the 0-based index
            w -= weights[i - 1]; // Reduce the capacity we're looking for
        }
    }

    // The items are found in reverse order, so we reverse to get 0..n order
    reverse(result.selectedItems.begin(), result.selectedItems.end());

    auto end = chrono::high_resolution_clock::now();
    result.executionTime = chrono::duration_cast<chrono::microseconds>(end - start).count();

    // Approximate memory used by the DP table + input/output vectors
    size_t dpTableMemory = sizeof(int) * (n + 1) * (capacity + 1);
    size_t vectorMemory = (sizeof(int) * (weights.size() + values.size())) +
        (sizeof(int) * result.selectedItems.size());
    result.memoryUsed = dpTableMemory + vectorMemory;

    return result;
}

int main(int argc, char *argv[]) {
    // Use fast I/O.
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

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
    Result result = solveKnapsackDP(capacity, weights, values);

    // Output results
    cout << result.maxValue << endl;
    cout << result.selectedItems.size() << endl;
    for (int idx : result.selectedItems) {
        cout << idx << " ";
    }
    if (!result.selectedItems.empty()) {
        cout << endl;
    }
    cout << result.executionTime << endl;
    cout << result.memoryUsed << endl;

    return 0;
}
