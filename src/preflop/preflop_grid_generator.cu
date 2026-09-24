#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; exit(EXIT_FAILURE); } } while (0)

constexpr size_t N_CLASSES         = 169;
constexpr size_t N_CLASSES_PADDED  = 192;
constexpr size_t MAX_PRE_ACTIONS   = 3;

struct SubsetFlop { uint8_t c1, c2, c3, pad; float weight; };
__constant__ float d_combo_weights[N_CLASSES_PADDED];

__global__ void kernel_evaluate_subset_flops(
    int num_subset_flops, const SubsetFlop* __restrict__ subset_flops,
    const float* __restrict__ reach_p0, const float* __restrict__ reach_p1,
    const float* __restrict__ allin_eq, float* __restrict__ out_ev_p0, float* __restrict__ out_ev_p1, float pot_flop)
{
    int cls = blockIdx.x * blockDim.x + threadIdx.x;
    if (cls >= N_CLASSES) return;
    float acc_p0 = 0.0f, acc_p1 = 0.0f, total_w = 0.0f;
    float sum_opp_p1 = 0.0f, eq_vs_p1 = 0.0f;
    for (int opp = 0; opp < N_CLASSES; ++opp) {
        float r = reach_p1[opp]; sum_opp_p1 += r; eq_vs_p1 += r * allin_eq[cls * N_CLASSES + opp];
    }
    if (sum_opp_p1 > 1e-7f) eq_vs_p1 /= sum_opp_p1;

    float sum_opp_p0 = 0.0f, eq_vs_p0 = 0.0f;
    for (int opp = 0; opp < N_CLASSES; ++opp) {
        float r = reach_p0[opp]; sum_opp_p0 += r; eq_vs_p0 += r * (1.0f - allin_eq[opp * N_CLASSES + cls]);
    }
    if (sum_opp_p0 > 1e-7f) eq_vs_p0 /= sum_opp_p0;

    float r_ip  = 1.15f + 0.05f * (cls >= 13 && cls < 91 ? 1.0f : 0.0f);
    float r_oop = 0.85f - 0.05f * (cls >= 91 ? 1.0f : 0.0f);

    for (int f = 0; f < num_subset_flops; ++f) {
        float w = subset_flops[f].weight; total_w += w;
        acc_p0 += w * (eq_vs_p1 * r_ip * pot_flop - 2.1f);
        acc_p1 += w * (eq_vs_p0 * r_oop * pot_flop - 2.1f);
    }
    if (total_w > 0.0f) { out_ev_p0[cls] = acc_p0 / total_w; out_ev_p1[cls] = acc_p1 / total_w; }
}

__global__ void kernel_build_closed_tree_cfvs(
    const float* __restrict__ flop_ev_btn, const float* __restrict__ flop_ev_bb,
    const float* __restrict__ sig_btn_root, const float* __restrict__ sig_bb_def,
    const float* __restrict__ sig_btn_vs_3b, const float* __restrict__ sig_bb_vs_jam,
    const float* __restrict__ reach_btn, const float* __restrict__ reach_bb,
    const float* __restrict__ allin_eq, float* __restrict__ cfv_btn_root,
    float* __restrict__ cfv_bb_def, float* __restrict__ cfv_btn_vs_3b, float* __restrict__ cfv_bb_vs_jam,
    float stack_bb, float bubble_factor)
{
    int cls = blockIdx.x * blockDim.x + threadIdx.x;
    if (cls >= N_CLASSES) return;

    cfv_btn_vs_3b[0 * N_CLASSES_PADDED + cls] = -2.1f;
    float jam_mass_bb = 0.0f, eq_btn_call = 0.0f;
    for (int opp = 0; opp < N_CLASSES; ++opp) {
        float r_3b = reach_bb[opp] * sig_bb_def[2 * N_CLASSES_PADDED + opp];
        jam_mass_bb += r_3b; eq_btn_call += r_3b * allin_eq[cls * N_CLASSES + opp];
    }
    if (jam_mass_bb > 1e-7f) eq_btn_call /= jam_mass_bb;
    cfv_btn_vs_3b[1 * N_CLASSES_PADDED + cls] = eq_btn_call * stack_bb + (1.0f - eq_btn_call) * (-bubble_factor * stack_bb);
    float v_btn_3b = sig_btn_vs_3b[0 * N_CLASSES_PADDED + cls] * cfv_btn_vs_3b[0 * N_CLASSES_PADDED + cls] +
                     sig_btn_vs_3b[1 * N_CLASSES_PADDED + cls] * cfv_btn_vs_3b[1 * N_CLASSES_PADDED + cls];

    cfv_bb_def[0 * N_CLASSES_PADDED + cls] = -1.0f;
    cfv_bb_def[1 * N_CLASSES_PADDED + cls] = flop_ev_bb[cls];

    float btn_f_3b = 0.0f, btn_c_3b = 0.0f, eq_bb_jam = 0.0f;
    for (int opp = 0; opp < N_CLASSES; ++opp) {
        float r_open = reach_btn[opp] * sig_btn_root[1 * N_CLASSES_PADDED + opp];
        btn_f_3b += r_open * sig_btn_vs_3b[0 * N_CLASSES_PADDED + opp];
        btn_c_3b += r_open * sig_btn_vs_3b[1 * N_CLASSES_PADDED + opp];
        eq_bb_jam += r_open * sig_btn_vs_3b[1 * N_CLASSES_PADDED + opp] * (1.0f - allin_eq[opp * N_CLASSES + cls]);
    }
    float tot_btn_3b = btn_f_3b + btn_c_3b + 1e-9f;
    if (btn_c_3b > 1e-7f) eq_bb_jam /= btn_c_3b;
    float ev_sh_bb = eq_bb_jam * stack_bb + (1.0f - eq_bb_jam) * (-bubble_factor * stack_bb);
    cfv_bb_def[2 * N_CLASSES_PADDED + cls] = (btn_f_3b * 3.6f + btn_c_3b * ev_sh_bb) / tot_btn_3b;

    cfv_bb_vs_jam[0 * N_CLASSES_PADDED + cls] = -1.0f;
    float btn_j_mass = 0.0f, eq_vs_openjam = 0.0f;
    for (int opp = 0; opp < N_CLASSES; ++opp) {
        float r_j = reach_btn[opp] * sig_btn_root[2 * N_CLASSES_PADDED + opp];
        btn_j_mass += r_j; eq_vs_openjam += r_j * (1.0f - allin_eq[opp * N_CLASSES + cls]);
    }
    if (btn_j_mass > 1e-7f) eq_vs_openjam /= btn_j_mass;
    cfv_bb_vs_jam[1 * N_CLASSES_PADDED + cls] = eq_vs_openjam * stack_bb + (1.0f - eq_vs_openjam) * (-bubble_factor * stack_bb);

    cfv_btn_root[0 * N_CLASSES_PADDED + cls] = 0.0f;
    float bb_f = 0.0f, bb_c = 0.0f, bb_j = 0.0f;
    for (int opp = 0; opp < N_CLASSES; ++opp) {
        float r = reach_bb[opp];
        bb_f += r * sig_bb_def[0 * N_CLASSES_PADDED + opp];
        bb_c += r * sig_bb_def[1 * N_CLASSES_PADDED + opp];
        bb_j += r * sig_bb_def[2 * N_CLASSES_PADDED + opp];
    }
    float tot_bb = bb_f + bb_c + bb_j + 1e-9f;
    cfv_btn_root[1 * N_CLASSES_PADDED + cls] = (bb_f * 1.5f + bb_c * flop_ev_btn[cls] + bb_j * v_btn_3b) / tot_bb;

    float bb_fj = 0.0f, bb_cj = 0.0f, eq_btn_oj = 0.0f;
    for (int opp = 0; opp < N_CLASSES; ++opp) {
        float r = reach_bb[opp];
        bb_fj += r * sig_bb_vs_jam[0 * N_CLASSES_PADDED + opp];
        bb_cj += r * sig_bb_vs_jam[1 * N_CLASSES_PADDED + opp];
        eq_btn_oj += r * sig_bb_vs_jam[1 * N_CLASSES_PADDED + opp] * allin_eq[cls * N_CLASSES + opp];
    }
    float tot_bb_j = bb_fj + bb_cj + 1e-9f;
    if (bb_cj > 1e-7f) eq_btn_oj /= bb_cj;
    cfv_btn_root[2 * N_CLASSES_PADDED + cls] = (bb_fj * 1.5f + bb_cj * (eq_btn_oj * stack_bb + (1.0f - eq_btn_oj) * (-bubble_factor * stack_bb))) / tot_bb_j;
}

__global__ void kernel_dcfr_update(
    int num_actions, float* __restrict__ regrets, float* __restrict__ current_strat,
    float* __restrict__ strat_sums, const float* __restrict__ action_cfvs, const float* __restrict__ reach,
    float w_pos, float w_strat)
{
    int cls = blockIdx.x * blockDim.x + threadIdx.x;
    if (cls >= N_CLASSES) return;
    float node_v = 0.0f;
    for (int a = 0; a < num_actions; ++a) node_v += current_strat[a * N_CLASSES_PADDED + cls] * action_cfvs[a * N_CLASSES_PADDED + cls];

    float pos_sum = 0.0f, next_sig[MAX_PRE_ACTIONS] = {0.0f};
    for (int a = 0; a < num_actions; ++a) {
        int idx = a * N_CLASSES_PADDED + cls;
        float inst_r = action_cfvs[idx] - node_v;
        float new_r = w_pos * fmaxf(0.0f, regrets[idx] + inst_r);
        regrets[idx] = new_r; pos_sum += new_r; next_sig[a] = new_r;
        strat_sums[idx] += w_strat * reach[cls] * current_strat[idx];
    }
    for (int a = 0; a < num_actions; ++a) {
        int idx = a * N_CLASSES_PADDED + cls;
        current_strat[idx] = (pos_sum > 1e-7f) ? (next_sig[a] / pos_sum) : (1.0f / (float)num_actions);
    }
}

int main(int argc, char** argv) {
    int iters = 300; float stack = 20.0f, bf = 1.35f;
    std::ifstream in("mtt_subset.bin", std::ios::binary);
    if (!in) { std::cerr << "mtt_subset.bin missing!\n"; return 1; }
    uint32_t count = 0; in.read(reinterpret_cast<char*>(&count), 4);
    std::vector<SubsetFlop> subset_flops(count);
    for (uint32_t i = 0; i < count; ++i) {
        in.read(reinterpret_cast<char*>(&subset_flops[i].c1), 1);
        in.read(reinterpret_cast<char*>(&subset_flops[i].c2), 1);
        in.read(reinterpret_cast<char*>(&subset_flops[i].c3), 1);
        in.read(reinterpret_cast<char*>(&subset_flops[i].pad), 1);
        in.read(reinterpret_cast<char*>(&subset_flops[i].weight), sizeof(float));
    }

    SubsetFlop* d_subset; CUDA_CHECK(cudaMalloc(&d_subset, count * sizeof(SubsetFlop)));
    CUDA_CHECK(cudaMemcpy(d_subset, subset_flops.data(), count * sizeof(SubsetFlop), cudaMemcpyHostToDevice));

    std::vector<float> h_combos(N_CLASSES_PADDED, 0.0f);
    for (size_t i = 0; i < 13; ++i) h_combos[i] = 6.0f;
    for (size_t i = 13; i < 91; ++i) h_combos[i] = 4.0f;
    for (size_t i = 91; i < 169; ++i) h_combos[i] = 12.0f;
    CUDA_CHECK(cudaMemcpyToSymbol(d_combo_weights, h_combos.data(), N_CLASSES_PADDED * 4));

    std::vector<float> h_eq(N_CLASSES * N_CLASSES, 0.5f);
    for (size_t i = 0; i < N_CLASSES; ++i) {
        for (size_t j = 0; j < N_CLASSES; ++j) {
            if (i < 13 && j >= 13) h_eq[i * N_CLASSES + j] = 0.65f;
            else if (i >= 13 && j < 13) h_eq[i * N_CLASSES + j] = 0.35f;
            else if (i < 13 && j < 13) h_eq[i * N_CLASSES + j] = (i < j) ? 0.82f : (i > j ? 0.18f : 0.50f);
            else h_eq[i * N_CLASSES + j] = (i < j) ? 0.60f : (i > j ? 0.40f : 0.50f);
        }
    }
    float* d_eq; CUDA_CHECK(cudaMalloc(&d_eq, N_CLASSES * N_CLASSES * 4));
    CUDA_CHECK(cudaMemcpy(d_eq, h_eq.data(), N_CLASSES * N_CLASSES * 4, cudaMemcpyHostToDevice));

    float *d_r_btn, *d_sig_btn, *d_sums_btn, *d_cfv_btn;
    float *d_r_bb, *d_sig_bb, *d_sums_bb, *d_cfv_bb;
    float *d_r_btn_3b, *d_sig_btn_3b, *d_sums_btn_3b, *d_cfv_btn_3b;
    float *d_r_bb_j, *d_sig_bb_j, *d_sums_bb_j, *d_cfv_bb_j;
    float *d_reach_btn, *d_reach_bb, *d_ev_btn, *d_ev_bb;

    size_t a3 = 3 * N_CLASSES_PADDED * 4, a2 = 2 * N_CLASSES_PADDED * 4, a1 = N_CLASSES_PADDED * 4;
    CUDA_CHECK(cudaMalloc(&d_r_btn, a3)); CUDA_CHECK(cudaMalloc(&d_sig_btn, a3)); CUDA_CHECK(cudaMalloc(&d_sums_btn, a3)); CUDA_CHECK(cudaMalloc(&d_cfv_btn, a3));
    CUDA_CHECK(cudaMalloc(&d_r_bb, a3)); CUDA_CHECK(cudaMalloc(&d_sig_bb, a3)); CUDA_CHECK(cudaMalloc(&d_sums_bb, a3)); CUDA_CHECK(cudaMalloc(&d_cfv_bb, a3));
    CUDA_CHECK(cudaMalloc(&d_r_btn_3b, a2)); CUDA_CHECK(cudaMalloc(&d_sig_btn_3b, a2)); CUDA_CHECK(cudaMalloc(&d_sums_btn_3b, a2)); CUDA_CHECK(cudaMalloc(&d_cfv_btn_3b, a2));
    CUDA_CHECK(cudaMalloc(&d_r_bb_j, a2)); CUDA_CHECK(cudaMalloc(&d_sig_bb_j, a2)); CUDA_CHECK(cudaMalloc(&d_sums_bb_j, a2)); CUDA_CHECK(cudaMalloc(&d_cfv_bb_j, a2));
    CUDA_CHECK(cudaMalloc(&d_reach_btn, a1)); CUDA_CHECK(cudaMalloc(&d_reach_bb, a1));
    CUDA_CHECK(cudaMalloc(&d_ev_btn, a1)); CUDA_CHECK(cudaMalloc(&d_ev_bb, a1));

    CUDA_CHECK(cudaMemset(d_r_btn, 0, a3)); CUDA_CHECK(cudaMemset(d_sums_btn, 0, a3));
    CUDA_CHECK(cudaMemset(d_r_bb, 0, a3)); CUDA_CHECK(cudaMemset(d_sums_bb, 0, a3));
    CUDA_CHECK(cudaMemset(d_r_btn_3b, 0, a2)); CUDA_CHECK(cudaMemset(d_sums_btn_3b, 0, a2));
    CUDA_CHECK(cudaMemset(d_r_bb_j, 0, a2)); CUDA_CHECK(cudaMemset(d_sums_bb_j, 0, a2));

    std::vector<float> init_3(3 * N_CLASSES_PADDED, 1.0f / 3.0f), init_2(2 * N_CLASSES_PADDED, 0.5f);
    CUDA_CHECK(cudaMemcpy(d_sig_btn, init_3.data(), a3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sig_bb, init_3.data(), a3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sig_btn_3b, init_2.data(), a2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sig_bb_j, init_2.data(), a2, cudaMemcpyHostToDevice));

    std::vector<float> init_r(N_CLASSES_PADDED, 0.0f);
    for (size_t i = 0; i < N_CLASSES; ++i) init_r[i] = h_combos[i] / 1326.0f;
    CUDA_CHECK(cudaMemcpy(d_reach_btn, init_r.data(), a1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reach_bb, init_r.data(), a1, cudaMemcpyHostToDevice));

    dim3 block(128);
    dim3 grid((N_CLASSES + block.x - 1) / block.x);

    for (int t = 1; t <= iters; ++t) {
        float tf = static_cast<float>(t);
        float t15 = powf(tf, 1.5f);
        float wp = t15 / (t15 + 1.0f);
        float ws = tf * tf;

        kernel_evaluate_subset_flops<<<grid, block>>>(count, d_subset, d_reach_btn, d_reach_bb, d_eq, d_ev_btn, d_ev_bb, 4.7f);
        kernel_build_closed_tree_cfvs<<<grid, block>>>(d_ev_btn, d_ev_bb, d_sig_btn, d_sig_bb, d_sig_btn_3b, d_sig_bb_j, d_reach_btn, d_reach_bb, d_eq, d_cfv_btn, d_cfv_bb, d_cfv_btn_3b, d_cfv_bb_j, stack, bf);
        kernel_dcfr_update<<<grid, block>>>(3, d_r_btn, d_sig_btn, d_sums_btn, d_cfv_btn, d_reach_btn, wp, ws);
        kernel_dcfr_update<<<grid, block>>>(3, d_r_bb, d_sig_bb, d_sums_bb, d_cfv_bb, d_reach_bb, wp, ws);
        kernel_dcfr_update<<<grid, block>>>(2, d_r_btn_3b, d_sig_btn_3b, d_sums_btn_3b, d_cfv_btn_3b, d_reach_btn, wp, ws);
        kernel_dcfr_update<<<grid, block>>>(2, d_r_bb_j, d_sig_bb_j, d_sums_bb_j, d_cfv_bb_j, d_reach_bb, wp, ws);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_sums(3 * N_CLASSES_PADDED);
    CUDA_CHECK(cudaMemcpy(h_sums.data(), d_sums_btn, a3, cudaMemcpyDeviceToHost));
    std::ofstream out("anchor_20bb_bf135.bin", std::ios::binary);
    for (int a = 0; a < 3; ++a) {
        for (size_t c = 0; c < N_CLASSES; ++c) {
            float s = h_sums[a * N_CLASSES_PADDED + c];
            out.write(reinterpret_cast<const char*>(&s), 4);
        }
    }
    std::cout << "Successfully solved and saved anchor_20bb_bf135.bin\n";
    return 0;
}
