#include "lalib/solver/gmres.hpp"
#include <cstddef>
#include <fstream>
#include <gtest/gtest.h>
#include "assets.hpp"

auto load_mxt(std::string filename) -> std::tuple<size_t, lalib::SpCooMat<double>> {
    auto ifs = std::ifstream(filename);
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    size_t n;

    std::string line;
    while (std::getline(ifs, line)) {
        if (line[0] == '%') {
            continue;
        } else {
            // skip matrix size
            std::string dummy = "";
            auto ss = std::stringstream(line);
            ss >> n >> dummy >> dummy;
            break;
        }
    }

    std::vector<size_t> row;
    std::vector<size_t> col;
    std::vector<double> val;

    while (std::getline(ifs, line)) {
        auto ss = std::stringstream(line);
        size_t i, j;
        double v;
        ss >> i >> j >> v;
        row.push_back(i - 1);
        col.push_back(j - 1);
        val.push_back(v);
    }

    auto mat = lalib::SpCooMat<double>(std::move(val), std::move(row), std::move(col));
    return std::make_tuple(n, mat);
}

TEST(GmresTests, GmresTest) {
    auto mat = lalib::DynMat<double>(3, 3, {
        4.0, 2.0, 6.0,
        2.0, 5.0, 5.0,
        6.0, 5.0, 14.0
    });
    auto b = lalib::DynVec<double>({2.0, 5.0, 1.0});
    auto gmres = lalib::solver::Gmres<double, lalib::DynMat<double>>(std::move(mat), 1e-6);
    auto sol = gmres.solve(b);

    ASSERT_NEAR(1.25, sol[0], 1e-6);
    ASSERT_NEAR(1.5, sol[1], 1e-6);
    ASSERT_NEAR(-1.0, sol[2], 1e-6);
}

TEST(GmresTests, SpGmresTest) {
    auto mat = lalib::SpMat<double>(
        {4.0, 2.0, 6.0, 2.0, 5.0, 5.0, 6.0, 5.0, 14.0}, 
        {0, 3, 6, 9}, 
        {0, 1, 2, 0, 1, 2, 0, 1, 2}
    );
    auto b = lalib::DynVec<double>({2.0, 5.0, 1.0});
    auto gmres = lalib::solver::Gmres(std::move(mat), 1e-6);
    auto sol = gmres.solve(b);

    ASSERT_NEAR(1.25, sol[0], 1e-6);
    ASSERT_NEAR(1.5, sol[1], 1e-6);
    ASSERT_NEAR(-1.0, sol[2], 1e-6);
}

TEST(GmresTests, SpLargeGmresTest) {
    auto [n, mat_tmp] = load_mxt(ASSETS_DIR"/pores_1.mtx");
    auto mat = lalib::SpMat<double>(std::move(mat_tmp));
    ASSERT_EQ(mat.shape().first, n);
    ASSERT_EQ(mat.shape().second, n);

    auto x = lalib::DynVec<double>::filled(mat.shape().first, 1.0);
    auto b = mat * x;
    auto gmres = lalib::solver::Gmres(std::move(mat), 1e-6);
    auto sol = gmres.solve(b);

    for (auto i = 0u; i < sol.size(); ++i) {
        ASSERT_NEAR(1.0, sol[i], 1e-6);
    }
}