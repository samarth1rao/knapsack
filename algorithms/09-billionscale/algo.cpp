// dual_knapsack.cpp
// Dual-descent 0/1 knapsack simulator compatible with simulations.py
// Outputs:
// 1) total_value (int64)
// 2) num_selected (int)
// 3) indices (0-based) space separated (may be empty line)
// 4) execution_time_microseconds (int64)
// 5) memory_bytes (int64)
//
// Compile:
//   g++ -O3 -std=c++17 -fopenmp -o dual_knapsack dual_knapsack.cpp
//
// Notes:
// - Reads from stdin: "n W" then a line of n weights then a line of n profits.
// - Uses multiplicative dual update with optional OpenMP parallelization.
// - Performs a postprocessing pass to ensure final solution is feasible.

#include <bits/stdc++.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using int64 = long long;

static int64 get_peak_memory_bytes() {
    // Memory measurement not available on Windows
    return 0;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    int64 W;
    if (!(cin >> n >> W)) {
        cerr << "Input error: expected n W\n";
        return 2;
    }
    if (n <= 0) {
        // print empty solution
        cout << 0 << "\n" << 0 << "\n\n" << 0 << "\n" << 0 << "\n";
        return 0;
    }

    vector<int64> weights(n), profits(n);
    for (int i = 0; i < n; ++i) {
        cin >> weights[i];
    }
    for (int i = 0; i < n; ++i) {
        cin >> profits[i];
    }

    // Hyperparameters (tuneable inside code or by editing)
    double alpha = 0.05;        // learning rate for multiplicative update
    double tol = 1e-4;          // relative tolerance for capacity
    int max_iters = 1000;       // safety cap
    double lambda = 0.0;        // initial lambda; will set below heuristically

    // Heuristic initial lambda: average profit/weight (avoid division by zero)
    double sum_pw = 0.0;
    double sum_w = 0.0;
    for (int i = 0; i < n; ++i) {
        if (weights[i] > 0) {
            sum_pw += static_cast<double>(profits[i]) / static_cast<double>(weights[i]);
            sum_w += 1.0;
        }
    }
    if (sum_w > 0.0) lambda = sum_pw / sum_w;
    else lambda = 0.0;

    vector<char> x(n, 0); // selected flags (0/1)

    // Timing start
    auto t0 = chrono::high_resolution_clock::now();

    int iter = 0;
    double rel_err = 1e9;
    int64 total_weight = 0;
    int64 total_value = 0;

    // Main dual-descent loop
    for (iter = 0; iter < max_iters; ++iter) {
        // compute provisional x_i at current lambda
        int64 local_weight = 0;
        int64 local_value = 0;

        #ifdef _OPENMP
        // parallel local arrays for reduction
        int64 weight_acc = 0;
        int64 value_acc = 0;
        #pragma omp parallel
        {
            int64 wsum = 0;
            int64 vsum = 0;
            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                double thr = lambda * static_cast<double>(weights[i]);
                if (static_cast<double>(profits[i]) > thr) {
                    x[i] = 1;
                    wsum += weights[i];
                    vsum += profits[i];
                } else {
                    x[i] = 0;
                }
            }
            #pragma omp atomic
            weight_acc += wsum;
            #pragma omp atomic
            value_acc += vsum;
        }
        local_weight = weight_acc;
        local_value = value_acc;
        #else
        local_weight = 0;
        local_value = 0;
        for (int i = 0; i < n; ++i) {
            double thr = lambda * static_cast<double>(weights[i]);
            if (static_cast<double>(profits[i]) > thr) {
                x[i] = 1;
                local_weight += weights[i];
                local_value += profits[i];
            } else {
                x[i] = 0;
            }
        }
        #endif

        total_weight = local_weight;
        total_value = local_value;

        // Relative error on weight vs capacity
        double denom = (W == 0) ? 1.0 : static_cast<double>(W);
        rel_err = fabs(static_cast<double>(total_weight) - static_cast<double>(W)) / denom;
        if (rel_err <= tol) break;

        // Update lambda multiplicatively; ensure lambda >= 0
        double frac = (static_cast<double>(total_weight) - static_cast<double>(W)) / denom;
        double mult = 1.0 + alpha * frac;
        if (mult < 0.0) mult = 0.0; // safeguard
        lambda = lambda * mult;
        if (lambda < 0.0) lambda = 0.0;
    }

    // Post-processing: if overweight, drop selected items with smallest profit/weight ratio
    total_weight = 0;
    total_value = 0;
    vector<int> selected_indices;
    selected_indices.reserve(n);
    for (int i = 0; i < n; ++i) if (x[i]) {
        total_weight += weights[i];
        total_value += profits[i];
        selected_indices.push_back(i);
    }

    if (total_weight > W) {
        // create vector of (ratio, index, weight, profit) for selected items
        vector<tuple<double,int,int64,int64>> sel;
        sel.reserve(selected_indices.size());
        for (int idx : selected_indices) {
            double ratio;
            if (weights[idx] > 0) ratio = static_cast<double>(profits[idx]) / static_cast<double>(weights[idx]);
            else ratio = numeric_limits<double>::infinity();
            sel.emplace_back(ratio, idx, weights[idx], profits[idx]);
        }
        // sort ascending by ratio (worst items first), break ties by smallest profit
        sort(sel.begin(), sel.end(), [](const auto &a, const auto &b){
            if (get<0>(a) != get<0>(b)) return get<0>(a) < get<0>(b);
            return get<3>(a) < get<3>(b);
        });
        // remove items until feasible
        for (auto &t : sel) {
            if (total_weight <= W) break;
            int idx = get<1>(t);
            total_weight -= get<2>(t);
            total_value -= get<3>(t);
            x[idx] = 0;
        }
    }

    // finalize selected list
    selected_indices.clear();
    for (int i = 0; i < n; ++i) if (x[i]) selected_indices.push_back(i);

    // Timing end
    auto t1 = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();

    // Peak memory in bytes
    int64 peak_mem = get_peak_memory_bytes();

    // Output as expected by simulations.py
    cout << total_value << "\n";
    cout << (int)selected_indices.size() << "\n";
    if (!selected_indices.empty()) {
        for (size_t i = 0; i < selected_indices.size(); ++i) {
            if (i) cout << " ";
            cout << selected_indices[i];
        }
        cout << "\n";
    } else {
        cout << "\n";
    }
    cout << elapsed << "\n";
    cout << peak_mem << "\n";

    return 0;
}
