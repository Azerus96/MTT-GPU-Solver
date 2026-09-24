#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; exit(EXIT_FAILURE); } } while (0)

constexpr size_t N_CANONICAL_FLOPS = 1755;
constexpr size_t N_PREFLOP_CLASSES = 169;
constexpr size_t N_SCENARIOS       = 4;
constexpr size_t N_ALIVE_COMBOS    = 1176;
constexpr size_t N_PADDED_COMBOS   = 1216;

struct CanonicalFlop { uint8_t c1, c2, c3, orbit_weight; };
struct alignas(8) AliveCombo { uint8_t c1, c2; uint64_t mask; };

inline uint16_t get_canonical_class(uint8_t c1, uint8_t c2) {
    uint8_t r1 = c1 / 4, s1 = c1 % 4, r2 = c2 / 4, s2 = c2 % 4;
    uint8_t hi_r = (r1 > r2) ? r1 : r2, lo_r = (r1 < r2) ? r1 : r2;
    if (r1 == r2) return static_cast<uint16_t>(12 - hi_r);
    else if (s1 == s2) { uint16_t b = 13; for (int r = 12; r > hi_r; --r) b += r; return b + (hi_r - 1 - lo_r); }
    else { uint16_t b = 91; for (int r = 12; r > hi_r; --r) b += r; return b + (hi_r - 1 - lo_r); }
}

std::vector<CanonicalFlop> generate_all_1755_flops() {
    std::vector<CanonicalFlop> flops; flops.reserve(N_CANONICAL_FLOPS);
    auto mc = [](int r, int s) -> uint8_t { return r * 4 + s; };
    for (int r = 12; r >= 0; --r) flops.push_back({mc(r, 0), mc(r, 1), mc(r, 2), 4});
    for (int rp = 12; rp >= 0; --rp) for (int rk = 12; rk >= 0; --rk) {
        if (rp == rk) continue;
        flops.push_back({mc(rp, 0), mc(rp, 1), mc(rk, 0), 12});
        flops.push_back({mc(rp, 0), mc(rp, 1), mc(rk, 2), 12});
    }
    for (int r1 = 12; r1 >= 2; --r1) for (int r2 = r1 - 1; r2 >= 1; --r2) for (int r3 = r2 - 1; r3 >= 0; --r3) {
        flops.push_back({mc(r1, 0), mc(r2, 0), mc(r3, 0), 4});
        flops.push_back({mc(r1, 0), mc(r2, 0), mc(r3, 1), 12});
        flops.push_back({mc(r1, 0), mc(r2, 1), mc(r3, 0), 12});
        flops.push_back({mc(r1, 1), mc(r2, 0), mc(r3, 0), 12});
        flops.push_back({mc(r1, 0), mc(r2, 1), mc(r3, 2), 24});
    }
    return flops;
}

__device__ inline uint32_t fast_7card_score(const uint8_t* cards) {
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

__global__ void kernel_evaluate_flop_ev(const AliveCombo* combos, const uint8_t* deck, const uint8_t* flop, const float* reach_p0, const float* reach_p1, float* out_ev_p0, float* out_ev_p1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_ALIVE_COMBOS) return;
    AliveCombo hero = combos[i];
    uint8_t h7[7] = {flop[0], flop[1], flop[2], hero.c1, hero.c2, 0, 0};
    float total_ev_p0 = 0.0f, total_ev_p1 = 0.0f;
    for (int j = 0; j < N_ALIVE_COMBOS; j += 4) {
        AliveCombo villain = combos[j];
        if ((hero.mask & villain.mask) != 0) continue;
        uint8_t v7[7] = {flop[0], flop[1], flop[2], villain.c1, villain.c2, 0, 0};
        uint64_t dead = hero.mask | villain.mask;
        int wins = 0, ties = 0, total = 0;
        #pragma unroll 4
        for (int r1 = 0; r1 < 47; r1 += 2) {
            uint8_t turn = deck[r1]; if (dead & (1ULL << turn)) continue;
            h7[5] = turn; v7[5] = turn;
            for (int r2 = r1 + 1; r2 < 47; r2 += 3) {
                uint8_t river = deck[r2]; if (dead & (1ULL << river)) continue;
                h7[6] = river; v7[6] = river;
                uint32_t s_hero = fast_7card_score(h7), s_vill = fast_7card_score(v7);
                if (s_hero > s_vill) wins += 2; else if (s_hero == s_vill) ties += 1;
                total += 2;
            }
        }
        if (total > 0) {
            float eq = static_cast<float>(wins + ties) / static_cast<float>(total);
            total_ev_p0 += reach_p1[j] * (eq * 2.0f - 1.0f);
            total_ev_p1 += reach_p0[j] * ((1.0f - eq) * 2.0f - 1.0f);
        }
    }
    out_ev_p0[i] = total_ev_p0; out_ev_p1[i] = total_ev_p1;
}

int main(int argc, char** argv) {
    int limit_flops = N_CANONICAL_FLOPS;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--limit") == 0 && i + 1 < argc) limit_flops = std::atoi(argv[++i]);
    }
    auto flops = generate_all_1755_flops();
    std::vector<float> global_ev_tensor(limit_flops * N_SCENARIOS * 2 * N_PREFLOP_CLASSES, 0.0f);
    int num_gpus = 1; cudaGetDeviceCount(&num_gpus);

    #pragma omp parallel num_threads(num_gpus)
    {
        int gpu_id = omp_get_thread_num();
        CUDA_CHECK(cudaSetDevice(gpu_id));
        AliveCombo* d_combos; uint8_t *d_deck, *d_flop; float *d_r0, *d_r1, *d_ev0, *d_ev1;
        CUDA_CHECK(cudaMalloc(&d_combos, N_PADDED_COMBOS * sizeof(AliveCombo)));
        CUDA_CHECK(cudaMalloc(&d_deck, 52)); CUDA_CHECK(cudaMalloc(&d_flop, 3));
        CUDA_CHECK(cudaMalloc(&d_r0, N_PADDED_COMBOS * 4)); CUDA_CHECK(cudaMalloc(&d_r1, N_PADDED_COMBOS * 4));
        CUDA_CHECK(cudaMalloc(&d_ev0, N_PADDED_COMBOS * 4)); CUDA_CHECK(cudaMalloc(&d_ev1, N_PADDED_COMBOS * 4));
        std::vector<float> h_r(N_PADDED_COMBOS, 1.0f / N_ALIVE_COMBOS);
        CUDA_CHECK(cudaMemcpy(d_r0, h_r.data(), N_PADDED_COMBOS * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_r1, h_r.data(), N_PADDED_COMBOS * 4, cudaMemcpyHostToDevice));
        std::vector<float> h_ev0(N_PADDED_COMBOS), h_ev1(N_PADDED_COMBOS);

        #pragma omp for schedule(dynamic, 4)
        for (int f_idx = 0; f_idx < limit_flops; ++f_idx) {
            CanonicalFlop f = flops[f_idx];
            uint8_t flop_cards[3] = {f.c1, f.c2, f.c3};
            uint64_t flop_mask = (1ULL << f.c1) | (1ULL << f.c2) | (1ULL << f.c3);
            std::vector<uint8_t> deck; for (uint8_t c = 0; c < 52; ++c) if (!(flop_mask & (1ULL << c))) deck.push_back(c);
            std::vector<AliveCombo> alive(N_PADDED_COMBOS); std::vector<uint16_t> combo_to_class(N_ALIVE_COMBOS);
            int n_al = 0;
            for (uint8_t c1 = 0; c1 < 52; ++c1) for (uint8_t c2 = c1 + 1; c2 < 52; ++c2) {
                uint64_t m = (1ULL << c1) | (1ULL << c2);
                if (!(m & flop_mask)) { alive[n_al] = {c1, c2, m}; combo_to_class[n_al] = get_canonical_class(c1, c2); n_al++; }
            }
            CUDA_CHECK(cudaMemcpy(d_flop, flop_cards, 3, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_deck, deck.data(), deck.size(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_combos, alive.data(), N_PADDED_COMBOS * sizeof(AliveCombo), cudaMemcpyHostToDevice));
            kernel_evaluate_flop_ev<<<(N_ALIVE_COMBOS + 255) / 256, 256>>>(d_combos, d_deck, d_flop, d_r0, d_r1, d_ev0, d_ev1);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_ev0.data(), d_ev0, N_ALIVE_COMBOS * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_ev1.data(), d_ev1, N_ALIVE_COMBOS * 4, cudaMemcpyDeviceToHost));
            for (size_t scen = 0; scen < N_SCENARIOS; ++scen) {
                size_t base_idx = ((f_idx * N_SCENARIOS + scen) * 2) * N_PREFLOP_CLASSES;
                float scen_mult = 1.0f + 0.1f * scen;
                for (int c = 0; c < N_ALIVE_COMBOS; ++c) {
                    uint16_t cls = combo_to_class[c];
                    global_ev_tensor[base_idx + 0 * N_PREFLOP_CLASSES + cls] += h_ev0[c] * scen_mult;
                    global_ev_tensor[base_idx + 1 * N_PREFLOP_CLASSES + cls] += h_ev1[c] * scen_mult;
                }
            }
        }
        cudaFree(d_combos); cudaFree(d_deck); cudaFree(d_flop); cudaFree(d_r0); cudaFree(d_r1); cudaFree(d_ev0); cudaFree(d_ev1);
    }
    std::ofstream out("mtt_ev_dump_1755.bin", std::ios::binary);
    out.write(reinterpret_cast<const char*>(global_ev_tensor.data()), global_ev_tensor.size() * 4);
    std::cout << "Successfully exported mtt_ev_dump_1755.bin\n";
    return 0;
}
