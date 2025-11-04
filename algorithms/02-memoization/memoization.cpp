#include <bits/stdc++.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

vector<int> weights, values;
vector<vector<int>> memo;
int n, capacity;

// Recursive function with memoization
int knapsack_memoization(int i, int remaining_capacity) {
    if (i == 0 || remaining_capacity == 0)
        return 0;

    if (memo[i][remaining_capacity] != -1)
        return memo[i][remaining_capacity];

    if (weights[i - 1] > remaining_capacity)
        return memo[i][remaining_capacity] = knapsack_memoization(i - 1, remaining_capacity);
    else {
        int include_item = values[i - 1] + knapsack_memoization(i - 1, remaining_capacity - weights[i - 1]);
        int exclude_item = knapsack_memoization(i - 1, remaining_capacity);
        return memo[i][remaining_capacity] = max(include_item, exclude_item);
    }
}

// Function to reconstruct the chosen items
vector<int> reconstruct_solution() {
    vector<int> selected_items;
    int i = n, w = capacity;

    while (i > 0 && w > 0) {
        if (memo[i][w] != memo[i - 1][w]) {
            selected_items.push_back(i - 1);
            w -= weights[i - 1];
        }
        i--;
    }
    reverse(selected_items.begin(), selected_items.end());
    return selected_items;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> n >> capacity;
    weights.resize(n);
    values.resize(n);

    for (int i = 0; i < n; i++)
        cin >> weights[i];
    for (int i = 0; i < n; i++)
        cin >> values[i];

    memo.assign(n + 1, vector<int>(capacity + 1, -1));

    auto start = high_resolution_clock::now();
    int max_value = knapsack_memoization(n, capacity);
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start).count();

    vector<int> selected_items = reconstruct_solution();

    // Approximate memory used (arrays + memo table)
    size_t memory_used = sizeof(int) * ((n + 1) * (capacity + 1) + n * 2);

    cout << max_value << "\n";
    cout << selected_items.size() << "\n";
    for (size_t i = 0; i < selected_items.size(); i++) {
        cout << selected_items[i];
        if (i + 1 < selected_items.size()) cout << " ";
    }
    cout << "\n";
    cout << duration << "\n";
    cout << memory_used << "\n";

    return 0;
}
