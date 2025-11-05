"""Generate a knapsack problems dataset.

Command-line arguments (all provided as --arg value):
	--out     : output CSV path (default: data/knapsack_dataset.csv)
	--total   : total number of problems to generate (default: 100)
	--level   : maximum difficulty level to include (default: 2)
							0 => Tiny only
							1 => Tiny + Small
							2 => Tiny + Small + Medium  (default)
							3 => + Large
							4 => + Massive (all categories)
	--seed    : optional integer seed for reproducible sampling of (category,n)

What this script emits:
	A CSV where each row is a single knapsack instance. Fields are:
		category, n, weights, prices, capacity, best_picks, best_price, seed

Notes on semantics:
	- `weights` and `prices` are JSON arrays (quoted) of length n. The generator
		uses a correlated price scheme so that a pre-selected small set S is the
		provably optimal solution while prices appear noisy and non-trivial.
	- `best_picks` is a compact JSON array of selected indices (e.g. "[1,45,102]")
		— we do NOT emit a binary 0/1 mask of length n to avoid huge CSV fields.

How optimality is enforced (brief):
	- Pre-select a small set S of k items. Generate weights w_i randomly.
	- Set capacity = sum(w_i for i in S).
	- For items in S set price approximately = w_i * M_HIGH + noise.
		For non-selected items set price approximately = w_i * M_LOW + noise
		with M_HIGH > M_LOW. This ensures items in S have strictly better
		price-to-weight ratios and S is the unique optimal solution.

Memory note:
	The script streams arrays into the CSV and replays the RNGs to avoid
	holding large arrays in memory. It also deletes per-instance objects and
	calls the garbage collector after each row.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import gc
from typing import Dict, List, Optional


def log_uniform_int(low: int, high: int, rng: random.Random) -> int:
	"""Sample integer between low..high (inclusive) on a log-uniform scale."""
	if low >= high:
		return low
	log_low = math.log10(low)
	log_high = math.log10(high)
	u = rng.random() * (log_high - log_low) + log_low
	return max(low, min(high, int(round(10 ** u))))



def build_n_list(total: int, rng: random.Random, level: Optional[int] = None) -> List[Dict]:
	"""Build a list of (category, n) entries totaling `total` problems.

	Categories and sampling ranges:
	  Tiny   : 20 - 40
	  Small  : 10^2 - 10^3
	  Medium : 10^4 - 10^5
	  Large  : 10^6 - 10^7
	  Massive: 10^8 - 10^9

	We distribute `total` roughly evenly across the included categories and
	sample n log-uniformly (Tiny uniformly).
	"""
	categories = [
		("Tiny", 20, 40),
		("Small", 10 ** 2, 10 ** 3),
		("Medium", 10 ** 4, 10 ** 5),
		("Large", 10 ** 6, 10 ** 7),
		("Massive", 10 ** 8, 10 ** 9),
	]
	# level controls how many categories to include: 0..4
	if level is not None:
		if level < 0 or level > 4:
			raise ValueError("level must be 0,1,2,3,4 or omitted")
		categories = categories[: level + 1]
	# Distribute `total` evenly across the number of included categories.
	num_cats = len(categories)
	per_cat = [total // num_cats] * num_cats
	for i in range(total % num_cats):
		per_cat[i] += 1

	result = []
	# Use per-category counters to avoid rescanning `result` on every loop
	per_cat_counts = {c[0]: 0 for c in categories}
	for (cat, low, high), count in zip(categories, per_cat):
		# For very small ranges (like Tiny) it's possible the user requests more
		# entries than there are unique n values. In that case allow duplicates
		# instead of spinning forever trying to deduplicate.
		unique_possible = (high - low + 1) if cat == "Tiny" else None
		seen = set()
		attempts = 0
		while per_cat_counts[cat] < count and attempts < max(1000, count * 10):
			attempts += 1
			if cat == "Tiny":
				n = rng.randint(low, high)
			else:
				n = log_uniform_int(low, high, rng)
			# If duplicates are possible but the requested count exceeds the
			# unique possibilities, don't deduplicate.
			if unique_possible is not None and count <= unique_possible:
				if (cat, n) in seen:
					continue
				seen.add((cat, n))
			result.append({"category": cat, "n": int(n)})
			per_cat_counts[cat] += 1

	# Shuffle to avoid grouped categories
	rng.shuffle(result)
	return result


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "knapsack_dataset.csv"), help="Output CSV path")
	parser.add_argument("--total", type=int, default=100, help="Total number of problems to generate (default 100)")
	parser.add_argument("--level", type=int, choices=[0, 1, 2, 3, 4], default=2, help="Which difficulty levels to include: 0=Tiny, 1=Tiny+Small, 2=Tiny+Small+Medium (default), 3=+Large, 4=+Massive (all)")
	parser.add_argument("--seed", type=int, default=None)
	args = parser.parse_args()

	rng = random.Random(args.seed)
	specs = build_n_list(args.total, rng, level=args.level)

	# CSV fields: no compact/notes fields; always emit full arrays
	fields = ["category", "n", "weights", "prices", "capacity", "best_picks", "best_price", "seed"]

	def _csv_quote(s: str) -> str:
		# simple CSV quoting: double any double-quotes and wrap in quotes
		return '"' + s.replace('"', '""') + '"'

	with open(args.out, "w", newline="", encoding="utf-8") as f:
		# write header
		f.write(",".join(fields) + "\n")

		for i, spec in enumerate(specs):
			cat = spec["category"]
			n = spec["n"]
			seed = rng.randint(0, 2 ** 31 - 1)

			# Per-instance RNG for reproducibility of sampling choices
			r = random.Random(seed)

			# choose k (number selected) using same heuristic as before
			k_max = max(1, min(n, int(max(1, n * 0.1))))
			if n <= 50:
				k = r.randint(1, min(n, max(1, int(n * 0.5))))
			else:
				k = r.randint(1, max(1, min(k_max, max(1, int(n * 0.02)))))

			selected = set(sorted(r.sample(range(n), k)))

			# Stream-write arrays directly into CSV to avoid building large lists.
			# 1) Weights: generate deterministic sequence with a dedicated RNG
			r_weights = random.Random(seed ^ 0xA5A5A5A5)

			# Start building the row: category and n
			# We'll write fields manually separated by commas. JSON arrays are quoted.
			f.write(_csv_quote(cat) + "," + str(n) + ",")

			# Stream weights JSON array and compute capacity
			f.write('"[')
			capacity = 0
			for idx in range(n):
				w = r_weights.randint(1, 100)
				if idx in selected:
					capacity += w
				# write number
				if idx != 0:
					f.write(", ")
				f.write(str(w))
			f.write(']"')

			f.write(",")

			# 2) PRICES: generate correlated prices so selected items have
			#    higher price-to-weight ratios (harder to spot) while
			#    preserving a provably optimal selected set S.
			# Parameters for correlation
			M_LOW = 10
			M_HIGH = 15  # must be > M_LOW
			NOISE_RANGE = max(1, int(M_LOW * 0.1))

			# We can't store all weights for huge n. Instead generate the
			# weights a second time (deterministically) so we can compute
			# correlated prices in a second pass without holding the list.
			r_weights_pass2 = random.Random(seed ^ 0xA5A5A5A5)
			r_prices_noise = random.Random(seed ^ 0x5A5A5A5)

			f.write('"[')
			best_price = 0
			for idx in range(n):
				w = r_weights_pass2.randint(1, 100)  # same sequence as weights
				noise = r_prices_noise.randint(-NOISE_RANGE, NOISE_RANGE)

				if idx in selected:
					price = (w * M_HIGH) + noise
					best_price += price
				else:
					price = (w * M_LOW) + noise

				# Ensure price is a positive integer
				price = max(1, int(price))

				if idx != 0:
					f.write(", ")
				f.write(str(price))
			f.write(']"')

			# capacity and then best_picks (we write indices, not a huge binary array)
			f.write(",")
			f.write(str(capacity))
			f.write(",")

			# Write the (small) list of selected indices as a compact JSON array
			import json as _json
			selected_list = sorted(list(selected))
			f.write(_csv_quote(_json.dumps(selected_list)))

			# finish the row with best_price and seed
			f.write(",")
			f.write(str(best_price))
			f.write(",")
			f.write(str(seed))
			f.write("\n")

			# Free per-instance objects immediately to reduce memory pressure.
			try:
				del selected
				del r
				del r_weights
				del r_weights_pass2
				del r_prices_noise
			except Exception:
				pass
			# Hint to the GC to reclaim memory promptly (helps on long runs)
			gc.collect()

	print(f"Wrote {args.out} with {len(specs)} problems (seed={args.seed}, level={args.level})")


if __name__ == "__main__":
	main()
