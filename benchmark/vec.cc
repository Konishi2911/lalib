#include "lalib/vec/sized_vec.hpp"
#include "lalib/vec/dyn_vec.hpp"
#include "lalib/ops/vec_ops.hpp"
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>

template<size_t N>
auto generate_rand_sized_vec(std::mt19937& rand) -> lalib::SizedVec<double, N> {
    auto dist = std::uniform_real_distribution<double>(-1e+5, 1e+5);
    auto v = lalib::SizedVec<double, N>::uninit();
    for (auto i = 0u; i < N; ++i) {
        v[i] = dist(rand);
    }
    return v;
}

auto generate_rand_dyn_vec(size_t n, std::mt19937& rand) -> lalib::DynVec<double> {
    auto dist = std::uniform_real_distribution<double>(-1e+5, 1e+5);
    auto v = lalib::DynVec<double>::uninit(n);
    for (auto i = 0u; i < n; ++i) {
        v[i] = dist(rand);
    }
    return v;
}

template<typename F>
auto measure_consumption_time(int64_t min_time, F func) -> double {
    auto start = std::chrono::system_clock::now();
    auto iter = 0;
    int64_t elapsed = 0;
    for (iter = 0; elapsed < min_time; ++iter) {
        func();
        
        auto end = std::chrono::system_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    return elapsed / static_cast<double>(iter);
}


void sized_vec_bench();
void dyn_vec_bench();

int main(int argc, char* argv[]) {
    auto backend = 
    #ifdef LALIB_BLAS_BACKEND
        "BLAS";
    #else
        "Internal";
    #endif

    auto use_accelerator = 
    #if LALIB_USE_ACCELERATOR
        true;
    #else
        false;
    #endif

    std::cout << "Benchmarks for linear algebric vector implementations." << std::endl;
    std::cout << std::setw(20) << std::left << " Backend" << ": " << backend << std::endl;
    std::cout << std::setw(20) << std::left << " Use accelerator" << ": " << use_accelerator << std::endl;

    sized_vec_bench();
    dyn_vec_bench();
}

void sized_vec_bench() {
    std::cout << std::endl;
    std::cout << "=== SizedVec ===" << std::endl;

    std::cout << " Creating datasets ... " << std::flush;
    auto start = std::chrono::system_clock::now();

    auto rand = std::mt19937(std::random_device()());
    auto v1_1 = generate_rand_sized_vec<1>(rand);
    auto v1_2 = generate_rand_sized_vec<1>(rand);
    auto v2_1 = generate_rand_sized_vec<2>(rand);
    auto v2_2 = generate_rand_sized_vec<2>(rand);
    auto v10_1 = generate_rand_sized_vec<10>(rand);
    auto v10_2 = generate_rand_sized_vec<10>(rand);
    auto v100_1 = generate_rand_sized_vec<100>(rand);
    auto v100_2 = generate_rand_sized_vec<100>(rand);
    auto v1000_1 = generate_rand_sized_vec<1000>(rand);
    auto v1000_2 = generate_rand_sized_vec<1000>(rand);
    auto v10000_1 = generate_rand_sized_vec<10000>(rand);
    auto v10000_2 = generate_rand_sized_vec<10000>(rand);
    auto v100000_1 = generate_rand_sized_vec<100000>(rand);
    auto v100000_2 = generate_rand_sized_vec<100000>(rand);

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Done. (Elapsed: " << elapsed << " ms)" << std::endl;

    std::cout << std::endl;
    std::cout << " # Addition" << std::endl;
    std::cout << " # of elems | Elapsed " << std::endl;
    std::cout << " -----------|--------------" << std::endl;
    {
        auto v2_r = lalib::SizedVec<double, 2>::uninit();
        auto elapsed_v2 = measure_consumption_time(100, [&](){
            lalib::add(v2_1, v2_2, v2_r);
        });
        std::cout << "  " << std::setw(10) << "2" << "| " << elapsed_v2 << " ms" << std::endl;
    }
    {
        auto v10_r = lalib::SizedVec<double, 10>::uninit();
        auto elapsed_v10 = measure_consumption_time(100, [&](){
            lalib::add(v10_1, v10_2, v10_r);
        });
        std::cout << "  " << std::setw(10) << "10" << "| " << elapsed_v10 << " ms" << std::endl;
    }
    {
        auto v100_r = lalib::SizedVec<double, 100>::uninit();
        auto elapsed_v100 = measure_consumption_time(100, [&](){
            lalib::add(v100_1, v100_2, v100_r);
        });
        std::cout << "  " << std::setw(10) << "100" << "| " << elapsed_v100 << " ms" << std::endl;
    }
    {
        auto v1000_r = lalib::SizedVec<double, 1000>::uninit();
        auto elapsed_v1000 = measure_consumption_time(100, [&](){
            lalib::add(v1000_1, v1000_2, v1000_r);
        });
        std::cout << "  " << std::setw(10) << "1000" << "| " << elapsed_v1000 << " ms" << std::endl;
    }
    {
        auto v10000_r = lalib::SizedVec<double, 10000>::uninit();
        auto elapsed_v10000 = measure_consumption_time(100, [&](){
            lalib::add(v10000_1, v10000_2, v10000_r);
        });
        std::cout << "  " << std::setw(10) << "10000" << "| " << elapsed_v10000 << " ms" << std::endl;
    }
    {
        auto v100000_r = lalib::SizedVec<double, 100000>::uninit();
        auto elapsed_v100000 = measure_consumption_time(100, [&](){
            lalib::add(v100000_1, v100000_2, v100000_r);
        });
        std::cout << "  " << std::setw(10) << "100000" << "| " << elapsed_v100000 << " ms" << std::endl;
    }
}

void dyn_vec_bench() {
    std::cout << std::endl;
    std::cout << "=== DynVec ===" << std::endl;

    std::cout << std::endl;
    std::cout << " # Addition" << std::endl;
    std::cout << " # of elems | Elapsed " << std::endl;
    std::cout << " -----------|--------------" << std::endl;

    constexpr auto order = 7;
    auto rand = std::mt19937(std::random_device()());
    for (auto i = 0; i <= order; ++i) {
        auto n = static_cast<size_t>(std::pow(10, i));
        auto v1 = generate_rand_dyn_vec(n, rand);
        auto v2 = generate_rand_dyn_vec(n, rand);
        auto vr = lalib::DynVec<double>::uninit(n);

        double elapsed = measure_consumption_time(100, [&](){
            lalib::add(v1, v2, vr);
        });
        std::cout << "  " << std::setw(10) << n << "| " << elapsed << " ms" << std::endl;
    }
}