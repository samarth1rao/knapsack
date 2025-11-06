

#include <bits/stdc++.h>
#include <chrono>
using namespace std;
using namespace std::chrono;


struct SubsetL {
    int weight, value, mask;
};
struct SubsetR {
    int weight, value;
};



int main() {
    int n, capacity;
    cin >> n >> capacity;
    vector<int> weights(n), values(n);
    for (int i = 0; i < n; ++i) cin >> weights[i];
    for (int i = 0; i < n; ++i) cin >> values[i];

    // Hard limit for n1/n2 to avoid time/memory blowup
    int n1 = n / 2, n2 = n - n1;
    if (n1 > 24 || n2 > 24) {
        // Output error code for simulation harness
        cout << -1 << endl;
        cout << 0 << endl;
        cout << endl;
        cout << 0 << endl;
        cout << 0 << endl;
        return 69;
    }

    auto start = high_resolution_clock::now();

    vector<SubsetL> left;
    vector<SubsetR> right;

    // Generate all subsets for left half (store mask)
    for (int mask = 0; mask < (1 << n1); ++mask) {
        int w = 0, v = 0;
        for (int i = 0; i < n1; ++i) {
            if (mask & (1 << i)) {
                w += weights[i];
                v += values[i];
            }
        }
        left.push_back({ w, v, mask });
    }

    // Generate all subsets for right half (no mask)
    for (int mask = 0; mask < (1 << n2); ++mask) {
        int w = 0, v = 0;
        for (int i = 0; i < n2; ++i) {
            if (mask & (1 << i)) {
                w += weights[n1 + i];
                v += values[n1 + i];
            }
        }
        right.push_back({ w, v });
    }

    // Sort right by weight
    sort(right.begin(), right.end(), [](const SubsetR &a, const SubsetR &b) {
        return a.weight < b.weight;
        });

    // Filter dominated pairs in right
    vector<SubsetR> filtered;
    int maxv = INT_MIN;
    for (auto &s : right) {
        if (s.value > maxv) {
            filtered.push_back(s);
            maxv = s.value;
        }
    }
    right = filtered;

    // Meet in the middle
    int best_value = 0, best_left = 0, best_right_weight = 0, best_right_value = 0;
    int best_right_mask = 0;
    for (auto &l : left) {
        if (l.weight > capacity) continue;
        int rem = capacity - l.weight;
        // Binary search for best right subset
        int lo = 0, hi = (int)right.size() - 1, idx = -1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (right[mid].weight <= rem) {
                idx = mid;
                lo = mid + 1;
            }
            else {
                hi = mid - 1;
            }
        }
        if (idx != -1) {
            int total_value = l.value + right[idx].value;
            if (total_value > best_value) {
                best_value = total_value;
                best_left = l.mask;
                best_right_weight = right[idx].weight;
                best_right_value = right[idx].value;
                best_right_mask = idx; // store index for reconstruction
            }
        }
    }

    // Reconstruct right half mask by brute force (since we only stored index)
    int n2_masks = 1 << n2;
    int right_mask = 0;
    for (int mask = 0; mask < n2_masks; ++mask) {
        int w = 0, v = 0;
        for (int i = 0; i < n2; ++i) {
            if (mask & (1 << i)) {
                w += weights[n1 + i];
                v += values[n1 + i];
            }
        }
        if (w == best_right_weight && v == best_right_value) {
            right_mask = mask;
            break;
        }
    }

    // Collect selected items
    vector<int> selected;
    for (int i = 0; i < n1; ++i) {
        if (best_left & (1 << i)) selected.push_back(i);
    }
    for (int i = 0; i < n2; ++i) {
        if (right_mask & (1 << i)) selected.push_back(n1 + i);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();

    // Estimate memory usage: vectors + subset structs
    size_t mem_used = sizeof(int) * (weights.size() + values.size()) + selected.size() * sizeof(int);
    mem_used += left.size() * sizeof(SubsetL) + right.size() * sizeof(SubsetR);

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
