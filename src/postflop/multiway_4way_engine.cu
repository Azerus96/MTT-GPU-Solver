#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; exit(EXIT_FAILURE); } } while (0)

constexpr size_t N_PLAYERS        = 4;
constexpr size_t N_ALIVE_COMBOS   = 1176;
constexpr size_t N_PADDED_COMBOS  = 1216;
constexpr size_t MAX_POST_ACTIONS = 3;
constexpr size_t CHUNK_SIZE       = 5;

enum NodeType : uint8_t { NODE_DECISION = 0, NODE_FOLD_TERMINAL = 1, NODE_SHOWDOWN_TERMINAL = 2 };

struct alignas(16) FlatNode {
    uint8_t  type, player, num_actions, active_mask;
    int32_t  children_offset, infoset_offset;
    float    pot_size, committed[N_PLAYERS];
};

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__inline__ __device__ float warp_inclusive_scan(float val) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float t = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if ((threadIdx.x & 31) >= offset) val += t;
    }
    return val;
}

__global__ void kernel_4way_forward_sweep(const int* __restrict__ depth_node_indices, int num_nodes_at_depth, const FlatNode* __restrict__ nodes, const float* __restrict__ cumulative_regrets, float* __restrict__ current_strategy, float* __restrict__ reach_probs) {
    int node_idx = blockIdx.y; if (node_idx >= num_nodes_at_depth) return;
    int node_id = depth_node_indices[node_idx];
    FlatNode node = nodes[node_id]; if (node.type != NODE_DECISION) return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x; if (tid >= N_PADDED_COMBOS) return;

    int info_idx = node.infoset_offset, num_actions = node.num_actions, acting_p = node.player;
    float pos_regrets[MAX_POST_ACTIONS], sum_pos = 0.0f;
    if (tid < N_ALIVE_COMBOS) {
        #pragma unroll
        for (int a = 0; a < MAX_POST_ACTIONS; ++a) {
            if (a < num_actions) {
                float r = cumulative_regrets[(info_idx * MAX_POST_ACTIONS + a) * N_PADDED_COMBOS + tid];
                pos_regrets[a] = fmaxf(r, 0.0f); sum_pos += pos_regrets[a];
            } else pos_regrets[a] = 0.0f;
        }
    } else {
        #pragma unroll
        for (int a = 0; a < MAX_POST_ACTIONS; ++a) pos_regrets[a] = 0.0f;
    }

    float sigma[MAX_POST_ACTIONS];
    #pragma unroll
    for (int a = 0; a < MAX_POST_ACTIONS; ++a) {
        if (a < num_actions && tid < N_ALIVE_COMBOS) {
            sigma[a] = (sum_pos > 1e-7f) ? (pos_regrets[a] / sum_pos) : (1.0f / (float)num_actions);
        } else sigma[a] = 0.0f;
        current_strategy[(info_idx * MAX_POST_ACTIONS + a) * N_PADDED_COMBOS + tid] = sigma[a];
    }
    float parent_reaches[N_PLAYERS];
    #pragma unroll
    for (int p = 0; p < N_PLAYERS; ++p) parent_reaches[p] = reach_probs[(node_id * N_PLAYERS + p) * N_PADDED_COMBOS + tid];
    #pragma unroll
    for (int a = 0; a < MAX_POST_ACTIONS; ++a) {
        if (a < num_actions) {
            int child_id = node.children_offset + a;
            #pragma unroll
            for (int p = 0; p < N_PLAYERS; ++p) {
                reach_probs[(child_id * N_PLAYERS + p) * N_PADDED_COMBOS + tid] = (p == acting_p) ? (parent_reaches[p] * sigma[a]) : parent_reaches[p];
            }
        }
    }
}

__global__ void kernel_4way_evaluate_terminals(const int* __restrict__ terminal_node_indices, int num_terminals, const FlatNode* __restrict__ nodes, const float* __restrict__ reach_probs, const uint16_t* __restrict__ combo_rank_pos, const float* __restrict__ bubble_factors, float* __restrict__ cfvs) {
    int term_idx = blockIdx.x; if (term_idx >= num_terminals) return;
    int node_id = terminal_node_indices[term_idx];
    FlatNode node = nodes[node_id];
    int tid = threadIdx.x;

    __shared__ float s_reach[N_PLAYERS][N_PADDED_COMBOS], s_cdf[N_PLAYERS][N_PADDED_COMBOS], s_total_w[N_PLAYERS], s_warp_sums[8];
    #pragma unroll
    for (int i = 0; i < CHUNK_SIZE; ++i) {
        int idx = tid * CHUNK_SIZE + i;
        if (idx < N_PADDED_COMBOS) {
            #pragma unroll
            for (int p = 0; p < N_PLAYERS; ++p) s_reach[p][idx] = (idx < N_ALIVE_COMBOS) ? reach_probs[(node_id * N_PLAYERS + p) * N_PADDED_COMBOS + idx] : 0.0f;
        }
    }
    __syncthreads();

    if (node.type == NODE_FOLD_TERMINAL) {
        int winner = -1;
        for (int p = 0; p < N_PLAYERS; ++p) if (node.active_mask & (1 << p)) { winner = p; break; }
        for (int p = 0; p < N_PLAYERS; ++p) {
            float my_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < CHUNK_SIZE; ++i) { int idx = tid * CHUNK_SIZE + i; if (idx < N_ALIVE_COMBOS) my_sum += s_reach[p][idx]; }
            my_sum = warp_reduce_sum(my_sum);
            int lane = tid & 31, wid = tid >> 5;
            if (lane == 0) s_warp_sums[wid] = my_sum;
            __syncthreads();
            if (wid == 0) {
                float b_sum = (lane < 8) ? s_warp_sums[lane] : 0.0f;
                b_sum = warp_reduce_sum(b_sum);
                if (lane == 0) s_total_w[p] = b_sum;
            }
            __syncthreads();
        }
        #pragma unroll
        for (int i = 0; i < CHUNK_SIZE; ++i) {
            int idx = tid * CHUNK_SIZE + i;
            if (idx < N_PADDED_COMBOS) {
                for (int p = 0; p < N_PLAYERS; ++p) {
                    if (idx < N_ALIVE_COMBOS) {
                        float opp_w = 1.0f;
                        for (int o = 0; o < N_PLAYERS; ++o) if (o != p) opp_w *= s_total_w[o];
                        float payoff = (p == winner) ? (node.pot_size - node.committed[p]) : (-bubble_factors[p * N_PLAYERS + winner] * node.committed[p]);
                        cfvs[(node_id * N_PLAYERS + p) * N_PADDED_COMBOS + idx] = payoff * opp_w;
                    } else cfvs[(node_id * N_PLAYERS + p) * N_PADDED_COMBOS + idx] = 0.0f;
                }
            }
        }
        return;
    }

    #pragma unroll
    for (int i = 0; i < CHUNK_SIZE; ++i) {
        int idx = tid * CHUNK_SIZE + i;
        if (idx < N_PADDED_COMBOS) {
            uint16_t pos = combo_rank_pos[idx];
            #pragma unroll
            for (int p = 0; p < N_PLAYERS; ++p) s_cdf[p][pos] = (idx < N_ALIVE_COMBOS) ? s_reach[p][idx] : 0.0f;
        }
    }
    __syncthreads();

    for (int p = 0; p < N_PLAYERS; ++p) {
        if (!(node.active_mask & (1 << p))) {
            float my_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < CHUNK_SIZE; ++i) { int idx = tid * CHUNK_SIZE + i; if (idx < N_ALIVE_COMBOS) my_sum += s_reach[p][idx]; }
            my_sum = warp_reduce_sum(my_sum);
            int lane = tid & 31, wid = tid >> 5;
            if (lane == 0) s_warp_sums[wid] = my_sum;
            __syncthreads();
            if (wid == 0) {
                float b_sum = (lane < 8) ? s_warp_sums[lane] : 0.0f;
                b_sum = warp_reduce_sum(b_sum);
                if (lane == 0) s_total_w[p] = b_sum;
            }
            __syncthreads();
            continue;
        }
        float chunk_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < CHUNK_SIZE; ++i) { int idx = tid * CHUNK_SIZE + i; if (idx < N_ALIVE_COMBOS) chunk_sum += s_cdf[p][idx]; }
        int lane = tid & 31, wid = tid >> 5;
        float scanned_chunk = warp_inclusive_scan(chunk_sum);
        if (lane == 31) s_warp_sums[wid] = scanned_chunk;
        __syncthreads();
        if (wid == 0) {
            float val = (lane < 8) ? s_warp_sums[lane] : 0.0f;
            val = warp_inclusive_scan(val);
            if (lane < 8) s_warp_sums[lane] = val;
        }
        __syncthreads();
        float my_scan_base = ((wid > 0) ? s_warp_sums[wid - 1] : 0.0f) + (scanned_chunk - chunk_sum);
        #pragma unroll
        for (int i = 0; i < CHUNK_SIZE; ++i) {
            int idx = tid * CHUNK_SIZE + i;
            if (idx < N_PADDED_COMBOS) {
                float val = s_cdf[p][idx]; s_cdf[p][idx] = my_scan_base; my_scan_base += val;
            }
        }
        __syncthreads();
        if (tid == 255) s_total_w[p] = s_warp_sums[7];
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < CHUNK_SIZE; ++i) {
        int idx = tid * CHUNK_SIZE + i;
        if (idx < N_PADDED_COMBOS) {
            uint16_t pos = combo_rank_pos[idx];
            for (int p = 0; p < N_PLAYERS; ++p) {
                if (idx >= N_ALIVE_COMBOS) { cfvs[(node_id * N_PLAYERS + p) * N_PADDED_COMBOS + idx] = 0.0f; continue; }
                if (!(node.active_mask & (1 << p))) {
                    float opp_w = 1.0f;
                    for (int o = 0; o < N_PLAYERS; ++o) if (o != p) opp_w *= s_total_w[o];
                    cfvs[(node_id * N_PLAYERS + p) * N_PADDED_COMBOS + idx] = -bubble_factors[p * N_PLAYERS + p] * node.committed[p] * opp_w;
                    continue;
                }
                float gain = node.pot_size - node.committed[p], loss = -bubble_factors[p * N_PLAYERS + p] * node.committed[p];
                float prod_F = 1.0f, prod_W = 1.0f, inact_w = 1.0f;
                #pragma unroll
                for (int o = 0; o < N_PLAYERS; ++o) {
                    if (o != p) {
                        if (node.active_mask & (1 << o)) { prod_F *= s_cdf[o][pos]; prod_W *= s_total_w[o]; }
                        else inact_w *= s_total_w[o];
                    }
                }
                cfvs[(node_id * N_PLAYERS + p) * N_PADDED_COMBOS + idx] = (gain * prod_F + loss * (prod_W - prod_F)) * inact_w;
            }
        }
    }
}

__global__ void kernel_4way_backward_cfr_plus(const int* __restrict__ depth_node_indices, int num_nodes_at_depth, const FlatNode* __restrict__ nodes, const float* __restrict__ current_strategy, const float* __restrict__ reach_probs, float* __restrict__ cfvs, float* __restrict__ cumulative_regrets, float* __restrict__ cumulative_strategy_sums, float iter_weight) {
    int node_idx = blockIdx.y; if (node_idx >= num_nodes_at_depth) return;
    int node_id = depth_node_indices[node_idx];
    FlatNode node = nodes[node_id]; if (node.type != NODE_DECISION) return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x; if (tid >= N_PADDED_COMBOS) return;

    int info_idx = node.infoset_offset, num_actions = node.num_actions, acting_p = node.player;
    float action_cfv[MAX_POST_ACTIONS], node_cfv = 0.0f;
    #pragma unroll
    for (int a = 0; a < MAX_POST_ACTIONS; ++a) {
        if (a < num_actions) {
            int child_id = node.children_offset + a;
            action_cfv[a] = cfvs[(child_id * N_PLAYERS + acting_p) * N_PADDED_COMBOS + tid];
            node_cfv += current_strategy[(info_idx * MAX_POST_ACTIONS + a) * N_PADDED_COMBOS + tid] * action_cfv[a];
        } else action_cfv[a] = 0.0f;
    }
    cfvs[(node_id * N_PLAYERS + acting_p) * N_PADDED_COMBOS + tid] = node_cfv;
    #pragma unroll
    for (int p = 0; p < N_PLAYERS; ++p) {
        if (p != acting_p) {
            float sum_opp = 0.0f;
            #pragma unroll
            for (int a = 0; a < MAX_POST_ACTIONS; ++a) if (a < num_actions) sum_opp += cfvs[((node.children_offset + a) * N_PLAYERS + p) * N_PADDED_COMBOS + tid];
            cfvs[(node_id * N_PLAYERS + p) * N_PADDED_COMBOS + tid] = sum_opp;
        }
    }
    if (tid < N_ALIVE_COMBOS) {
        float r_p = reach_probs[(node_id * N_PLAYERS + acting_p) * N_PADDED_COMBOS + tid];
        #pragma unroll
        for (int a = 0; a < MAX_POST_ACTIONS; ++a) {
            if (a < num_actions) {
                int buf_idx = (info_idx * MAX_POST_ACTIONS + a) * N_PADDED_COMBOS + tid;
                float inst_r = action_cfv[a] - node_cfv;
                cumulative_regrets[buf_idx] = fmaxf(0.0f, cumulative_regrets[buf_idx] + inst_r);
                cumulative_strategy_sums[buf_idx] += iter_weight * r_p * current_strategy[buf_idx];
            }
        }
    }
}

int main() {
    std::cout << "Multiway 4-Way Engine Operational.\n";
    return 0;
}
