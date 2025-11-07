#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <queue>
#include <limits>

using namespace std;

// Struct to represent a node in the branch and bound tree
struct Node {
    int level;      // Current item index being considered
    int value;      // Current total value
    int weight;     // Current total weight
    double bound;   // Upper bound on maximum value
    vector<int> selectedItems; // Items selected so far

    // Constructor
    Node(int l, int v, int w, double b, const vector<int>& sel)
        : level(l), value(v), weight(w), bound(b), selectedItems(sel) {}

    // For priority queue (max-heap based on bound)
    bool operator<(const Node& other) const {
        return bound < other.bound;
    }
};

// Result struct to hold all output
struct Result {
    int maxValue;
    vector<int> selectedItems;
    long long executionTime; // in microseconds
    size_t memoryUsed; // in bytes (approximate)
};

/**
 * @brief Calculate upper bound using fractional knapsack relaxation
 * @param node Current node
 * @param n Total number of items
 * @param capacity Knapsack capacity
 * @param weights Item weights
 * @param values Item values
 * @return Upper bound on maximum value from this node
 */
double calculateBound(const Node& node, int n, int capacity,
                     const vector<int>& weights, const vector<int>& values) {
    if (node.weight >= capacity) {
        return 0; // Invalid node
    }

    double bound = node.value;
    int remainingWeight = capacity - node.weight;
    int j = node.level + 1;

    // Add items greedily (fractional knapsack)
    while (j < n && remainingWeight > 0) {
        if (weights[j] <= remainingWeight) {
            // Take the whole item
            bound += values[j];
            remainingWeight -= weights[j];
        } else {
            // Take fraction of the item
            bound += values[j] * (static_cast<double>(remainingWeight) / weights[j]);
            remainingWeight = 0;
        }
        j++;
    }

    return bound;
}

/**
 * @brief Solves the 0/1 Knapsack problem using Branch and Bound
 * @param capacity The total capacity of the knapsack
 * @param weights Const reference to the item weights vector
 * @param values Const reference to the item values vector
 * @return A Result struct containing the solution, time, and memory
 */
Result solveKnapsackBranchAndBound(int capacity, const vector<int>& weights, const vector<int>& values) {
    Result result;
    int n = weights.size();

    auto start = chrono::high_resolution_clock::now();

    // Initialize result
    result.maxValue = 0;
    result.selectedItems.clear();

    // Priority queue for branch and bound (max-heap)
    priority_queue<Node> pq;

    // Create root node
    Node root(-1, 0, 0, 0.0, vector<int>());
    root.bound = calculateBound(root, n, capacity, weights, values);
    pq.push(root);

    while (!pq.empty()) {
        // Get the node with highest bound
        Node current = pq.top();
        pq.pop();

        // If bound is worse than current best, prune
        if (current.bound <= result.maxValue) {
            continue;
        }

        // If we've considered all items
        if (current.level == n - 1) {
            // Check if this is better than current best
            if (current.value > result.maxValue && current.weight <= capacity) {
                result.maxValue = current.value;
                result.selectedItems = current.selectedItems;
            }
            continue;
        }

        // Consider next item (level + 1)
        int nextLevel = current.level + 1;

        // Branch 1: Don't take the item
        {
            Node child(nextLevel, current.value, current.weight,
                      0.0, current.selectedItems);
            child.bound = calculateBound(child, n, capacity, weights, values);

            if (child.bound > result.maxValue) {
                pq.push(child);
            }
        }

        // Branch 2: Take the item (if it fits)
        if (current.weight + weights[nextLevel] <= capacity) {
            vector<int> newSelected = current.selectedItems;
            newSelected.push_back(nextLevel);

            Node child(nextLevel,
                      current.value + values[nextLevel],
                      current.weight + weights[nextLevel],
                      0.0,
                      newSelected);
            child.bound = calculateBound(child, n, capacity, weights, values);

            if (child.bound > result.maxValue) {
                pq.push(child);
            }
        }
    }

    // Sort selected items by index for consistent output
    sort(result.selectedItems.begin(), result.selectedItems.end());

    auto end = chrono::high_resolution_clock::now();
    result.executionTime = chrono::duration_cast<chrono::microseconds>(end - start).count();

    // Approximate memory used
    // Priority queue memory is hard to estimate, but we can approximate
    // based on input size and typical node count
    size_t inputMemory = (sizeof(int) * weights.size()) + (sizeof(int) * values.size());
    size_t resultMemory = sizeof(int) * result.selectedItems.size();
    // Rough estimate: assume O(n) nodes in priority queue on average
    size_t pqMemoryEstimate = sizeof(Node) * n;
    result.memoryUsed = inputMemory + resultMemory + pqMemoryEstimate;

    return result;
}

int main(int argc, char* argv[]) {
    // Use fast I/O
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

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
    Result result = solveKnapsackBranchAndBound(capacity, weights, values);

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