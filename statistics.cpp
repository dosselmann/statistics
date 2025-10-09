// Copyright 2024 Richard Dosselmann
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

// statistics.cpp
//

#include <array>
#include <iostream>
#include <list>
#include <ranges>

#include "statistics.hpp"

/* main */

int
main()
{
    // example 1
    {
        struct PRODUCT {
            float price;
            int   quantity;
        };

    std::array<PRODUCT, 5> A = {
        {{5.0f, 1}, {1.7f, 2}, {9.2f, 5}, {4.4f, 7}, {1.7f, 3}}
    };
    auto A_ = A
        | std::views::transform([](const auto& product)
            { return product.price; })
        | std::ranges::to<std::vector<float>>();
    std::array<float, 5> W = { { 2.0f, 2.0f, 1.0f, 3.0f, 5.0f } };

    std::cout << "mean = "
        << std::mean(std::execution::par, A_);
    std::cout << "\nweighted mean = "
        << std::mean(std::execution::par, A_, W);
    std::cout << "\ngeometric mean = "          << std::geometric_mean(A_);
    std::cout << "\nweighted geometric mean = " << std::geometric_mean(A_, W);
    std::cout << "\nharmonic mean = "           << std::harmonic_mean(A_);
    std::cout << "\nweighted harmonic mean = "  << std::harmonic_mean(A_, W);
    std::cout << "\nvariance = "                << std::variance(A_);
    std::cout << "\nstandard deviation = "      << std::standard_deviation(A_);
    std::cout << "\nskewness = "                << std::skewness(A_);
    std::cout << "\nkurtosis = "                << std::kurtosis(A_);
    }

    std::cout << "\n";

    // example 2
    {
    std::list<float> L = { 8.0f, 6.0f, 12.0f, 3.0f, 5.0f };

    auto [mean, variance] = std::mean_variance(L);
    std::cout << "mean = "       << mean;
    std::cout << "\nvariance = " << variance;
    }

    std::cout << "\n";

    // example 3
    {
    std::vector<double> v = { 2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0 };

    std::cout << "skewness = " << std::skewness(v, false);
    
    std::cout << "\nkurtosis = " << std::kurtosis(
        v, { .sample=false, .excess=true });
    }

    std::cout << "\n";

    // example 4
    {
    std::vector<double> v1 = {
         2.0,  3.0,  5.0,  7.0,  11.0,  13.0,  17.0,  19.0 };
    std::vector<double> v2 = {
        -2.0, -3.0, -5.0, -7.0, -11.0, -13.0, -17.0, -19.0 };

    std::cout << "covariance = " << std::covariance(v1, v2);
    }
}