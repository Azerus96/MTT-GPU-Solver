import numpy as np
from scipy.optimize import nnls
import struct

N_FLOPS = 1755
TOTAL_FEATURES = 4 * 2 * 169

def generate_1755_flops_meta():
    flops = []
    mc = lambda r, s: r * 4 + s
    for r in range(12, -1, -1):
        flops.append((mc(r, 0), mc(r, 1), mc(r, 2), 4))
    for rp in range(12, -1, -1):
        for rk in range(12, -1, -1):
            if rp == rk: continue
            flops.append((mc(rp, 0), mc(rp, 1), mc(rk, 0), 12))
            flops.append((mc(rp, 0), mc(rp, 1), mc(rk, 2), 12))
    for r1 in range(12, 1, -1):
        for r2 in range(r1 - 1, 0, -1):
            for r3 in range(r2 - 1, -1, -1):
                flops.append((mc(r1, 0), mc(r2, 0), mc(r3, 0), 4))
                flops.append((mc(r1, 0), mc(r2, 0), mc(r3, 1), 12))
                flops.append((mc(r1, 0), mc(r2, 1), mc(r3, 0), 12))
                flops.append((mc(r1, 1), mc(r2, 0), mc(r3, 0), 12))
                flops.append((mc(r1, 0), mc(r2, 1), mc(r3, 2), 24))
    return flops

if __name__ == "__main__":
    A = np.fromfile("mtt_ev_dump_1755.bin", dtype=np.float32).reshape((N_FLOPS, TOTAL_FEATURES)).T
    flops_meta = generate_1755_flops_meta()
    orbit_weights = np.array([f[3] for f in flops_meta], dtype=np.float64)
    b = (A @ orbit_weights) / np.sum(orbit_weights)

    selected_indices = []
    residual = b.copy()
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    A_normed = A / col_norms

    for step in range(100):
        correlations = A_normed.T @ residual
        for idx in selected_indices:
            correlations[idx] = -float("inf")
        best_candidate = np.argmax(correlations)
        selected_indices.append(best_candidate)
        w_sub, _ = nnls(A[:, selected_indices], b)
        residual = b - A[:, selected_indices] @ w_sub

    weights, _ = nnls(A[:, selected_indices], b)
    normalized_weights = weights * (N_FLOPS / np.sum(weights))

    with open("mtt_subset.bin", "wb") as f:
        f.write(struct.pack("I", len(selected_indices)))
        for idx, w in zip(selected_indices, normalized_weights):
            c1, c2, c3, _ = flops_meta[idx]
            f.write(struct.pack("BBBBf", c1, c2, c3, 0, float(w)))

    with open("mtt_subset_weights.txt", "w") as f:
        RANKS = "23456789TJQKA"
        SUITS = "cdhs"
        card_str = lambda c: RANKS[c // 4] + SUITS[c % 4]
        for idx, w in zip(selected_indices, normalized_weights):
            c1, c2, c3, _ = flops_meta[idx]
            name = card_str(c1) + card_str(c2) + card_str(c3)
            f.write(f"{name}:{w:.4f}\n")

    print(f"Subset generated: 100 flops saved to mtt_subset.bin")
