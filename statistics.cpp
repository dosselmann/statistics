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

#include <iostream>
#include <list>

#include "statistics.hpp"

// custom unweighted accumulator

class custom_unweighted_accumulator :
    public std::unweighted_accumulator<double>
{
public:
    constexpr custom_unweighted_accumulator() noexcept
    { first_ = true; sum_sq_=0; }
  
    constexpr void operator()(const double& x)
    {
        unweighted_accumulator::operator()(x);
    
        if (first_)
        {
            min_   = max_ = x;
            first_ = false;
        }
        else
        {
            min_ = std::min(min_, x);
            max_ = std::max(max_, x);
        }
    
        sum_sq_ += x*x;
    }
  
    constexpr double min()    const noexcept { return min_; }
    constexpr double max()    const noexcept { return max_; }
    constexpr double sum_sq() const noexcept { return sum_sq_; }

private:
    bool   first_;
    double min_, max_;
    double sum_sq_;
};

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

    std::cout << "mean = " << std::mean(std::execution::par, A_, W);
    std::cout << "\nvariance = " << std::variance(A_);
    std::cout << "\nstandard deviation = " << std::standard_deviation(A_);
    }

    std::cout << "\n";

    // example 2
    {
    std::list<int> L = { 8, 6, 12, 3, 5 };

    auto [mean, variance] = std::mean_variance<float>(L);
    std::cout << "mean = "       << mean;
    std::cout << "\nvariance = " << variance;
    }

    std::cout << "\n";

    // example 3
    {
    std::vector<double> v = { 2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0 };

    std::cout << "kurtosis = " << std::kurtosis(v);
    }

    std::cout << "\n";

    // example 4
    {
    std::list<double> L = { 1., 2., 2., 2., 3., 3., 3. };

    std::unweighted_accumulator<double> acc;

    for (const auto& x : L)
        acc(x);

    std::cout << "mean = "                 << acc.mean();
    std::cout << "\nvariance = "           << acc.variance(0);
    std::cout << "\nstandard deviation = " << acc.standard_deviation();
    std::cout << "\nskewness = "           << acc.skewness(false);
    std::cout << "\nkurtosis = "           << acc.kurtosis();
    }

    std::cout << "\n";
    
    // example 5
    {
    std::vector<double> L = { 17.2, -14.27, 19.22, 13.56, -0.01, 2.6 };

    custom_unweighted_accumulator acc;

    for (const double& x: L)
        acc(x);

    std::cout << "mean = "             << acc.mean();
    std::cout << "\nvariance = "       << acc.variance();
    std::cout << "\nmax = "            << acc.max();
    std::cout << "\nmin = "            << acc.min();
    std::cout << "\nsum of squares = " << acc.sum_sq();
    }
}