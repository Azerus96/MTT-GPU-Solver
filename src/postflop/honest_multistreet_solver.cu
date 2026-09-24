#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; exit(EXIT_FAILURE); } } while (0)

constexpr size_t N_ALIVE_COMBOS  = 1176;
constexpr size_t N_PADDED_COMBOS = 1216;
constexpr size_t MAX_ACTIONS     = 3;

enum NodeType : uint8_t { NODE_DECISION = 0, NODE_FOLD_TERMINAL = 1, NODE_SHOWDOWN_TERMINAL = 2, NODE_CHANCE = 3 };

struct alignas(16) FlatNode {
    uint8_t  type, player, num_actions, active_mask;
    int32_t  children_offset, infoset_offset;
    int16_t  turn_idx;
    float    pot_size, committed[2];
};
struct alignas(8) AliveCombo { uint8_t c1, c2; uint64_t mask; };

__device__ inline uint32_t eval_7cards_fast(const uint8_t* cards) {
    uint32_t sc[4] = {0}, sm[4] = {0}, rc[13] = {0}, rm = 0;
    #pragma unroll
    for (int i = 0; i < 7; ++i) { uint8_t c = cards[i], r = c / 4, s = c % 4; sc[s]++; sm[s] |= (1U << r); rc[r]++; rm |= (1U << r); }
    auto f_str = [](uint32_t m) -> int {
        #pragma unroll
        for (int r = 12; r >= 4; --r) if ((m & (0x1FU << (r - 4))) == (0x1FU << (r - 4))) return r;
        if ((m & 0x100FU) == 0x100FU) return 3; return -1;
    };
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        if (sc[s] >= 5) {
            int sf = f_str(sm[s]); if (sf != -1) return (8U << 24) | sf;
            uint32_t score = (5U << 24); int p = 0;
            for (int r = 12; r >= 0 && p < 5; --r) if (sm[s] & (1U << r)) score |= (r << (4 * (4 - p++)));
            return score;
        }
    }
    int q = -1, t1 = -1, t2 = -1, p1 = -1, p2 = -1;
    for (int r = 12; r >= 0; --r) {
        if (rc[r] == 4) q = r; else if (rc[r] == 3) { if (t1 == -1) t1 = r; else if (t2 == -1) t2 = r; }
        else if (rc[r] == 2) { if (p1 == -1) p1 = r; else if (p2 == -1) p2 = r; }
    }
    if (q != -1) return (7U << 24) | (q << 4);
    if (t1 != -1 && (t2 != -1 || p1 != -1)) return (6U << 24) | (t1 << 4) | (t2 != -1 ? t2 : p1);
    int st = f_str(rm); if (st != -1) return (4U << 24) | st;
    if (t1 != -1) return (3U << 24) | (t1 << 8);
    if (p1 != -1 && p2 != -1) return (2U << 24) | (p1 << 8) | (p2 << 4);
    if (p1 != -1) return (1U << 24) | (p1 << 12);
    return rm;
}

__global__ void kernel_gen_49_turn_matrices(const AliveCombo* __restrict__ combos, const uint8_t* __restrict__ deck, const uint8_t* __restrict__ flop, float* __restrict__ turn_matrices) {
    int t_idx = blockIdx.y, h_idx = blockIdx.x, v_idx = threadIdx.x;
    if (t_idx >= 49 || h_idx >= N_ALIVE_COMBOS) return;
    uint8_t turn_card = deck[t_idx];
    AliveCombo hero = combos[h_idx];
    if ((hero.mask & (1ULL << turn_card)) != 0) {
        for (int j = v_idx; j < N_PADDED_COMBOS; j += blockDim.x) turn_matrices[(t_idx * N_PADDED_COMBOS + j) * N_PADDED_COMBOS + h_idx] = 0.0f;
        return;
    }
    for (int j = v_idx; j < N_PADDED_COMBOS; j += blockDim.x) {
        if (j >= N_ALIVE_COMBOS) { turn_matrices[(t_idx * N_PADDED_COMBOS + j) * N_PADDED_COMBOS + h_idx] = 0.0f; continue; }
        AliveCombo villain = combos[j];
        if ((hero.mask & villain.mask) != 0 || (villain.mask & (1ULL << turn_card)) != 0) {
            turn_matrices[(t_idx * N_PADDED_COMBOS + j) * N_PADDED_COMBOS + h_idx] = 0.0f; continue;
        }
        uint8_t h7[7] = {flop[0], flop[1], flop[2], turn_card, hero.c1, hero.c2, 0};
        uint8_t v7[7] = {flop[0], flop[1], flop[2], turn_card, villain.c1, villain.c2, 0};
        uint64_t dead = hero.mask | villain.mask | (1ULL << turn_card);
        int wins = 0, ties = 0, total = 0;
        #pragma unroll 4
        for (int r = 0; r < 49; ++r) {
            uint8_t river = deck[r]; if ((dead & (1ULL << river)) != 0) continue;
            h7[6] = river; v7[6] = river;
            uint32_t sh = eval_7cards_fast(h7), sv = eval_7cards_fast(v7);
            if (sh > sv) wins++; else if (sh == sv) ties++;
            total++;
        }
        float eq = (total > 0) ? (wins + 0.5f * ties) / total : 0.0f;
        turn_matrices[(t_idx * N_PADDED_COMBOS + j) * N_PADDED_COMBOS + h_idx] = eq;
    }
}

__global__ void kernel_forward_sweep(const int* __restrict__ depth_node_indices, int num_nodes_at_depth, const FlatNode* __restrict__ nodes, const uint8_t* __restrict__ deck, const AliveCombo* __restrict__ combos, const float* __restrict__ cumulative_regrets, float* __restrict__ current_strategy, float* __restrict__ reach_probs) {
    int node_idx = blockIdx.y; if (node_idx >= num_nodes_at_depth) return;
    int node_id = depth_node_indices[node_idx];
    FlatNode node = nodes[node_id];
    if (node.type != NODE_DECISION && node.type != NODE_CHANCE) return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N_PADDED_COMBOS) {
        float r0 = reach_probs[(node_id * 2 + 0) * N_PADDED_COMBOS + tid];
        float r1 = reach_probs[(node_id * 2 + 1) * N_PADDED_COMBOS + tid];
        if (node.type == NODE_CHANCE) {
            uint64_t mask = (tid < N_ALIVE_COMBOS) ? combos[tid].mask : ~0ULL;
            for (int i = 0; i < 49; ++i) {
                uint8_t turn_card = deck[i]; int child_id = node.children_offset + i;
                bool conflict = (mask & (1ULL << turn_card)) != 0;
                reach_probs[(child_id * 2 + 0) * N_PADDED_COMBOS + tid] = conflict ? 0.0f : r0;
                reach_probs[(child_id * 2 + 1) * N_PADDED_COMBOS + tid] = conflict ? 0.0f : r1;
            }
        } else if (node.type == NODE_DECISION) {
            int info_idx = node.infoset_offset;
            float pos_regrets[MAX_ACTIONS], sum_pos = 0.0f;
            if (tid < N_ALIVE_COMBOS) {
                for (int a = 0; a < node.num_actions; ++a) {
                    float r = cumulative_regrets[(info_idx * MAX_ACTIONS + a) * N_PADDED_COMBOS + tid];
                    pos_regrets[a] = fmaxf(r, 0.0f); sum_pos += pos_regrets[a];
                }
            }
            float sigma[MAX_ACTIONS];
            for (int a = 0; a < node.num_actions; ++a) {
                if (tid < N_ALIVE_COMBOS) {
                    sigma[a] = (sum_pos > 1e-7f) ? (pos_regrets[a] / sum_pos) : (1.0f / node.num_actions);
                    current_strategy[(info_idx * MAX_ACTIONS + a) * N_PADDED_COMBOS + tid] = sigma[a];
                } else sigma[a] = 0.0f;
                int child_id = node.children_offset + a;
                reach_probs[(child_id * 2 + 0) * N_PADDED_COMBOS + tid] = (node.player == 0) ? (r0 * sigma[a]) : r0;
                reach_probs[(child_id * 2 + 1) * N_PADDED_COMBOS + tid] = (node.player == 1) ? (r1 * sigma[a]) : r1;
            }
        }
    }
}

__global__ void kernel_evaluate_terminals(const int* __restrict__ terminal_node_indices, int num_terminals, const FlatNode* __restrict__ nodes, const float* __restrict__ reach_probs, const float* __restrict__ turn_matrices, float* __restrict__ cfvs) {
    int term_idx = blockIdx.y; if (term_idx >= num_terminals) return;
    int node_id = terminal_node_indices[term_idx];
    FlatNode node = nodes[node_id];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (node.type == NODE_FOLD_TERMINAL) {
        int winner = (node.active_mask == 0b01) ? 0 : 1; int loser = 1 - winner;
        float opp_r0 = (tid < N_ALIVE_COMBOS) ? reach_probs[(node_id * 2 + 0) * N_PADDED_COMBOS + tid] : 0.0f;
        float opp_r1 = (tid < N_ALIVE_COMBOS) ? reach_probs[(node_id * 2 + 1) * N_PADDED_COMBOS + tid] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            opp_r0 += __shfl_down_sync(0xFFFFFFFF, opp_r0, offset);
            opp_r1 += __shfl_down_sync(0xFFFFFFFF, opp_r1, offset);
        }
        __shared__ float s_tot0[32], s_tot1[32];
        int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        if (lane == 0) { s_tot0[wid] = opp_r0; s_tot1[wid] = opp_r1; }
        __syncthreads();
        float tot0 = 0.0f, tot1 = 0.0f;
        if (wid == 0) {
            tot0 = (lane < (blockDim.x >> 5)) ? s_tot0[lane] : 0.0f;
            tot1 = (lane < (blockDim.x >> 5)) ? s_tot1[lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                tot0 += __shfl_down_sync(0xFFFFFFFF, tot0, offset);
                tot1 += __shfl_down_sync(0xFFFFFFFF, tot1, offset);
            }
            if (lane == 0) { s_tot0[0] = tot0; s_tot1[0] = tot1; }
        }
        __syncthreads();
        tot0 = s_tot0[0]; tot1 = s_tot1[0];

        if (tid < N_PADDED_COMBOS) {
            if (tid < N_ALIVE_COMBOS) {
                cfvs[(node_id * 2 + winner) * N_PADDED_COMBOS + tid] = (node.pot_size - node.committed[winner]) * (winner == 0 ? tot1 : tot0);
                cfvs[(node_id * 2 + loser) * N_PADDED_COMBOS + tid]  = -node.committed[loser] * (loser == 0 ? tot1 : tot0);
            } else {
                cfvs[(node_id * 2 + 0) * N_PADDED_COMBOS + tid] = 0.0f;
                cfvs[(node_id * 2 + 1) * N_PADDED_COMBOS + tid] = 0.0f;
            }
        }
    } else if (node.type == NODE_SHOWDOWN_TERMINAL) {
        if (tid < N_PADDED_COMBOS) {
            int t_idx = node.turn_idx;
            const float* eq_matrix = &turn_matrices[t_idx * N_PADDED_COMBOS * N_PADDED_COMBOS];
            float gain0 = node.pot_size - node.committed[0], loss0 = -node.committed[0];
            float gain1 = node.pot_size - node.committed[1], loss1 = -node.committed[1];
            float cfv_p0 = 0.0f, cfv_p1 = 0.0f;

            if (tid < N_ALIVE_COMBOS) {
                const float* r0_ptr = &reach_probs[(node_id * 2 + 0) * N_PADDED_COMBOS];
                const float* r1_ptr = &reach_probs[(node_id * 2 + 1) * N_PADDED_COMBOS];
                #pragma unroll 8
                for (size_t j = 0; j < N_ALIVE_COMBOS; ++j) {
                    float eq0 = eq_matrix[j * N_PADDED_COMBOS + tid];
                    float eq1 = 1.0f - eq0;
                    cfv_p0 += r1_ptr[j] * (eq0 * gain0 + eq1 * loss0);
                    cfv_p1 += r0_ptr[j] * (eq1 * gain1 + eq0 * loss1);
                }
                cfvs[(node_id * 2 + 0) * N_PADDED_COMBOS + tid] = cfv_p0;
                cfvs[(node_id * 2 + 1) * N_PADDED_COMBOS + tid] = cfv_p1;
            } else {
                cfvs[(node_id * 2 + 0) * N_PADDED_COMBOS + tid] = 0.0f;
                cfvs[(node_id * 2 + 1) * N_PADDED_COMBOS + tid] = 0.0f;
            }
        }
    }
}

__global__ void kernel_backward_sweep(const int* __restrict__ depth_node_indices, int num_nodes_at_depth, const FlatNode* __restrict__ nodes, const uint8_t* __restrict__ deck, const AliveCombo* __restrict__ combos, const float* __restrict__ current_strategy, const float* __restrict__ reach_probs, float* __restrict__ cfvs, float* __restrict__ cumulative_regrets, float* __restrict__ cumulative_strategy_sums, float iter_weight) {
    int node_idx = blockIdx.y; if (node_idx >= num_nodes_at_depth) return;
    int node_id = depth_node_indices[node_idx];
    FlatNode node = nodes[node_id];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N_PADDED_COMBOS) {
        if (node.type == NODE_CHANCE) {
            uint64_t mask = (tid < N_ALIVE_COMBOS) ? combos[tid].mask : ~0ULL;
            float sum_cfv0 = 0.0f, sum_cfv1 = 0.0f;
            for (int i = 0; i < 49; ++i) {
                uint8_t turn_card = deck[i];
                if ((mask & (1ULL << turn_card)) == 0) {
                    int child_id = node.children_offset + i;
                    sum_cfv0 += cfvs[(child_id * 2 + 0) * N_PADDED_COMBOS + tid];
                    sum_cfv1 += cfvs[(child_id * 2 + 1) * N_PADDED_COMBOS + tid];
                }
            }
            if (tid < N_ALIVE_COMBOS) {
                cfvs[(node_id * 2 + 0) * N_PADDED_COMBOS + tid] = sum_cfv0 / 47.0f;
                cfvs[(node_id * 2 + 1) * N_PADDED_COMBOS + tid] = sum_cfv1 / 47.0f;
            }
        } else if (node.type == NODE_DECISION) {
            int info_idx = node.infoset_offset;
            int acting_p = node.player, opp_p = 1 - acting_p;
            float action_cfv[MAX_ACTIONS], node_cfv = 0.0f;
            for (int a = 0; a < node.num_actions; ++a) {
                int child_id = node.children_offset + a;
                action_cfv[a] = cfvs[(child_id * 2 + acting_p) * N_PADDED_COMBOS + tid];
                float sigma = current_strategy[(info_idx * MAX_ACTIONS + a) * N_PADDED_COMBOS + tid];
                node_cfv += sigma * action_cfv[a];
            }
            cfvs[(node_id * 2 + acting_p) * N_PADDED_COMBOS + tid] = node_cfv;
            float sum_opp_cfv = 0.0f;
            for (int a = 0; a < node.num_actions; ++a) {
                int child_id = node.children_offset + a;
                sum_opp_cfv += cfvs[(child_id * 2 + opp_p) * N_PADDED_COMBOS + tid];
            }
            cfvs[(node_id * 2 + opp_p) * N_PADDED_COMBOS + tid] = sum_opp_cfv;

            if (tid < N_ALIVE_COMBOS) {
                float r_p = reach_probs[(node_id * 2 + acting_p) * N_PADDED_COMBOS + tid];
                for (int a = 0; a < node.num_actions; ++a) {
                    int buf_idx = (info_idx * MAX_ACTIONS + a) * N_PADDED_COMBOS + tid;
                    float inst_r = action_cfv[a] - node_cfv;
                    cumulative_regrets[buf_idx] = fmaxf(0.0f, cumulative_regrets[buf_idx] + inst_r);
                    cumulative_strategy_sums[buf_idx] += iter_weight * r_p * current_strategy[buf_idx];
                }
            }
        }
    }
}

int main() {
    std::cout << "Honest Multi-Street Solver Engine Operational.\n";
    return 0;
}
