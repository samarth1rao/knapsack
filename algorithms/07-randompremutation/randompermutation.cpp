#include <bits/stdc++.h>
using namespace std;
using namespace std::chrono;

struct Item {
    int weight;
    int value;
    int index;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // --- Input parsing ---
    int n, W;
    if (!(cin >> n >> W)) return 0;

    vector<int> weights(n), values(n);
    for (int i = 0; i < n; ++i) cin >> weights[i];
    for (int i = 0; i < n; ++i) cin >> values[i];

    vector<Item> items(n);
    int wmax = 0;
    for (int i = 0; i < n; ++i) {
        items[i] = {weights[i], values[i], i};
        wmax = max(wmax, weights[i]);
    }

    // --- Random permutation ---
    random_device rd;
    mt19937_64 rng(rd());
    shuffle(items.begin(), items.end(), rng);

    // --- Start timer ---
    auto start = high_resolution_clock::now();

    // --- DP computation ---
    const int NEG_INF = -1e9;
    vector<int> prev(W + 1, NEG_INF), curr(W + 1, NEG_INF);
    prev[0] = 0;

    // Parent tracking for reconstruction
    vector<vector<int>> parent(n + 1, vector<int>(W + 1, -1));

    for (int i = 1; i <= n; ++i) {
        double mu = (1.0 * i / n) * W;
        double delta = sqrt(i * log(max(2, n))) * wmax;
        int low = max(0, (int)floor(mu - delta));
        int high = min(W, (int)ceil(mu + delta));

        fill(curr.begin() + low, curr.begin() + high + 1, NEG_INF);

        for (int j = low; j <= high; ++j) {
            // Option 1: don't take
            curr[j] = prev[j];
            parent[i][j] = j;

            // Option 2: take item i-1
            int w = items[i - 1].weight;
            int v = items[i - 1].value;
            if (j >= w && prev[j - w] != NEG_INF) {
                int candidate = prev[j - w] + v;
                if (candidate > curr[j]) {
                    curr[j] = candidate;
                    parent[i][j] = j - w;
                }
            }
        }
        swap(prev, curr);
    }

    // --- Get best value and reconstruct solution ---
    int best_val = 0, best_w = 0;
    for (int j = 0; j <= W; ++j) {
        if (prev[j] > best_val) {
            best_val = prev[j];
            best_w = j;
        }
    }

    // Backtrack selected items
    vector<int> chosen;
    int j = best_w;
    for (int i = n; i >= 1; --i) {
        int pj = parent[i][j];
        if (pj == -1) continue;
        if (pj != j) { // item was taken
            chosen.push_back(items[i - 1].index);
        }
        j = pj;
    }

    reverse(chosen.begin(), chosen.end());

    // --- Stop timer ---
    auto end = high_resolution_clock::now();
    auto exec_time = duration_cast<microseconds>(end - start).count();

    // --- Rough memory usage estimation ---
    size_t mem_bytes = sizeof(int) * (2 * (W + 1) + (size_t)(n + 1) * (W + 1));

    // --- Output (strict format) ---
    cout << best_val << "\n";
    cout << chosen.size() << "\n";
    for (size_t i = 0; i < chosen.size(); ++i) {
        if (i) cout << " ";
        cout << chosen[i];
    }
    cout << "\n";
    cout << exec_time << "\n";
    cout << mem_bytes << "\n";

    return 0;
}
