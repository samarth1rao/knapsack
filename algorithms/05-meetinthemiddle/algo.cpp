
#include <bits/stdc++.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

struct Item {
    int weight, value, index;
};

// Store (weight, value, mask)
struct Subset {
    int weight, value, mask;
};

int main() {
    int n, capacity;
    cin >> n >> capacity;
    vector<int> weights(n), values(n);
    for (int i = 0; i < n; ++i) cin >> weights[i];
    for (int i = 0; i < n; ++i) cin >> values[i];

    auto start = high_resolution_clock::now();

    int n1 = n / 2, n2 = n - n1;
    vector<Subset> left, right;

    // Generate all subsets for left half
    for (int mask = 0; mask < (1 << n1); ++mask) {
        int w = 0, v = 0;
        for (int i = 0; i < n1; ++i) {
            if (mask & (1 << i)) {
                w += weights[i];
                v += values[i];
            }
        }
        if (w <= capacity)
            left.push_back({w, v, mask});
    }

    // Generate all subsets for right half
    for (int mask = 0; mask < (1 << n2); ++mask) {
        int w = 0, v = 0;
        for (int i = 0; i < n2; ++i) {
            if (mask & (1 << i)) {
                w += weights[n1 + i];
                v += values[n1 + i];
            }
        }
        if (w <= capacity)
            right.push_back({w, v, mask});
    }

    // Sort right by weight, keep only best value for each weight
    sort(right.begin(), right.end(), [](const Subset &a, const Subset &b) {
        return a.weight < b.weight;
    });
    vector<Subset> filtered;
    int maxv = -1;
    for (auto &s : right) {
        if (s.value > maxv) {
            filtered.push_back(s);
            maxv = s.value;
        }
    }
    right = filtered;

    // Meet in the middle
    int best_value = 0, best_left = 0, best_right = 0;
    for (auto &l : left) {
        int rem = capacity - l.weight;
        // Binary search for best right subset
        int lo = 0, hi = right.size() - 1, idx = -1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (right[mid].weight <= rem) {
                idx = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        if (idx != -1) {
            int total_value = l.value + right[idx].value;
            if (total_value > best_value) {
                best_value = total_value;
                best_left = l.mask;
                best_right = right[idx].mask;
            }
        }
    }

    // Collect selected items
    vector<int> selected;
    for (int i = 0; i < n1; ++i) {
        if (best_left & (1 << i)) selected.push_back(i);
    }
    for (int i = 0; i < n2; ++i) {
        if (best_right & (1 << i)) selected.push_back(n1 + i);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();

    // Estimate memory usage: vectors + subset structs
    size_t mem_used = sizeof(int) * (weights.size() + values.size() + selected.size());
    mem_used += left.size() * sizeof(Subset) + right.size() * sizeof(Subset);

    // Output as per required format
    cout << best_value << endl;
    cout << selected.size() << endl;
    for (size_t i = 0; i < selected.size(); ++i) {
        cout << selected[i];
        if (i + 1 < selected.size()) cout << " ";
    }
    cout << endl;
    cout << duration << endl;
    cout << mem_used << endl;
}