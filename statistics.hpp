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

// statistics.hpp : P1708
//

// REFERENCE:
//
// Philippe Pébay, Timothy B. Terriberry, Hemanth Kolla and Janine Bennett,
// Numerically stable, scalable formulas for parallel and online computation
// of higher-order multivariate central moments with arbitrary weights,
// Computational Statistics, 31(4), p. 1305-1325, 2016.

#pragma once

#include <execution>

// implementation specific
#include <cmath>

/* ======================================================================== */

namespace std {

// implementation specific

template<class T>
inline constexpr T sqrt_(T x)
{ return (static_cast<T>(x) < T()) ? T() : static_cast<T>(std::sqrt(x)); }

/* functions */

// mean functions

// (1)
template<ranges::input_range R>
constexpr auto mean(R&& r) -> std::ranges::range_value_t<R>
{
	using T = std::iter_value_t<R>;

	T      m1 = T();
	size_t n  = 0;

	for (auto& x : r)
	{
		++n;
		m1 += (x-m1) / n;
	}

	return m1;
}

// (2)
template<ranges::input_range R, ranges::input_range Weights>
constexpr auto mean(R&& r, Weights&& w) -> std::ranges::range_value_t<R>
{
	using T1 = std::iter_value_t<R>;
	using T2 = std::iter_value_t<Weights>;

	T1   m1  = T1();
	T2   w1  = T2(),
	     w2  = T2();
	auto it1 = r.cbegin();
	auto it2 = w.cbegin();

	for (; it1 != r.cend(); ++it1, ++it2)
	{
		w2  = *it2;
		w1 += w2;
		m1 += w2/w1 * (*it1 - m1);
	}

	return m1;
}

// (3)
template<class ExecutionPolicy, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto mean(ExecutionPolicy&& policy, R&& r) -> std::ranges::range_value_t<R>
{ return std::mean<R>(r); }

// (4)
template<class ExecutionPolicy,
	ranges::input_range R,
	ranges::input_range Weights>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto mean(ExecutionPolicy&& policy, R&& r, Weights&& w) ->
	std::ranges::range_value_t<R>
{ return std::mean<R, Weights>(r, w); }

// (5)
template<ranges::input_range R>
constexpr auto geometric_mean(R&& r) -> std::ranges::range_value_t<R>
{
	using T = std::iter_value_t<R>;

	T      geometric_mean_ = T();
	size_t n               = 0;

	for (const auto& x : r)
	{
		++n;
		geometric_mean_ += (
			static_cast<T>(std::log(x)) - geometric_mean_) / n;
	}

	return static_cast<T>(std::exp(geometric_mean_));
}

// (6)
template<ranges::input_range R, ranges::input_range Weights>
constexpr auto geometric_mean(R&& r, Weights&& w) ->
	std::ranges::range_value_t<R>
{
	using T1 = std::iter_value_t<R>;
	using T2 = std::iter_value_t<Weights>;

	T1   geometric_mean_ = T1();
	T2   w1              = T2(),
	     w2              = T2();
	auto it1             = r.cbegin();
	auto it2             = w.cbegin();

	for (; it1 != r.cend(); ++it1, ++it2)
	{
		w2  = *it2;
		w1 += w2;
		geometric_mean_ += w2/w1 * (
			static_cast<T1>(std::log(*it1)) - geometric_mean_);
	}

	return static_cast<T1>(std::exp(geometric_mean_));
}

// (7)
template<class ExecutionPolicy, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto geometric_mean(ExecutionPolicy&& policy, R&& r) ->
	std::ranges::range_value_t<R>
{ return std::geometric_mean<R>(r); }

// (8)
template<class ExecutionPolicy,
	ranges::input_range R,
	ranges::input_range Weights>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto geometric_mean(ExecutionPolicy&& policy, R&& r, Weights&& w) ->
	std::ranges::range_value_t<R>
{ return geometric_mean<R, Weights>(r, w); }

// (9)
template<ranges::input_range R>
constexpr auto harmonic_mean(R&& r) -> std::ranges::range_value_t<R>
{
	using T = std::iter_value_t<R>;

	T      harmonic_mean_ = T();
	size_t n              = 0;

	for (const auto& x : r)
	{
		++n;
		harmonic_mean_ += (T(1)/x - harmonic_mean_) / n;
	}

	return T(1)/harmonic_mean_;
}

// (10)
template<ranges::input_range R, ranges::input_range Weights>
constexpr auto harmonic_mean(R&& r, Weights&& w) ->
	std::ranges::range_value_t<R>
{
	using T1 = std::iter_value_t<R>;
	using T2 = std::iter_value_t<Weights>;

	T1   harmonic_mean_ = T1();
	T2   w1             = T2(),
	     w2             = T2();
	auto it1            = r.cbegin();
	auto it2            = w.cbegin();

	for (; it1 != r.cend(); ++it1, ++it2)
	{
		w2              = *it2;
		w1             += w2;
		harmonic_mean_ += w2/w1 * (T1(1)/(*it1) - harmonic_mean_);
	}

	return w1/harmonic_mean_;
}

// (11)
template<class ExecutionPolicy, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto harmonic_mean(ExecutionPolicy&& policy, R&& r) ->
	std::ranges::range_value_t<R>
{ return std::harmonic_mean<R>(r); }

// (12)
template<class ExecutionPolicy,
    ranges::input_range R,
    ranges::input_range Weights>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto harmonic_mean(ExecutionPolicy&& policy, R&& r, Weights&& w) ->
std::ranges::range_value_t<R>
{ return std::harmonic_mean<R, Weights>(r, w); }

// variance functions

// (1)
template<ranges::input_range R>
constexpr auto variance(
	R&& r,
	std::ranges::range_value_t<R> ddof = std::ranges::range_value_t<R>(1)) ->
		std::ranges::range_value_t<R>
{
	using T = std::iter_value_t<R>;
	
	T      d,
	       m1 = T(),
	       m2 = T();
	size_t n  = 0;

	for (const auto& x : r)
	{
		++n;
		d   = x-m1;
		m2 += d*d*(n-1) / n;
		m1 += d/n;
	}

	return m2 / (n-ddof);
}

// (2)
template<class ExecutionPolicy, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto variance(
	ExecutionPolicy&& policy,
	R&& r,
	std::ranges::range_value_t<R> ddof = std::ranges::range_value_t<R>(1)) ->
		std::ranges::range_value_t<R>
{ return std::variance<R>(r, ddof); }

// standard deviation functions

// (1)
template<ranges::input_range R>
constexpr auto standard_deviation(
	R&& r,
	std::ranges::range_value_t<R> ddof = std::ranges::range_value_t<R>(1)) ->
		std::ranges::range_value_t<R>
{
	return static_cast<std::ranges::range_value_t<R>>(
		sqrt_<std::ranges::range_value_t<R>>(std::variance<R>(r, ddof)));
}

// (2)
template<class ExecutionPolicy, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto standard_deviation(
	ExecutionPolicy&& policy,
	R&& r,
	std::ranges::range_value_t<R> ddof = std::ranges::range_value_t<R>(1)) ->
		std::ranges::range_value_t<R>
{ return std::standard_deviation<R>(r, ddof); }

// mean, variance, standard deviation convenience functions

template<class T>
struct mean_variance_result { T mean, variance; };

template<class T>
struct mean_standard_deviation_result { T mean, standard_deviation; };

// (1)
template<ranges::input_range R>
constexpr auto mean_variance(
	R&& r,
	std::ranges::range_value_t<R> ddof = std::ranges::range_value_t<R>(1)) ->
		mean_variance_result<std::ranges::range_value_t<R>>
{
	using T = std::iter_value_t<R>;
	
	T      d,
	       m1 = T(),
	       m2 = T();
	size_t n  = 0;

	for (const auto& x : r)
	{
		++n;
		d   = x-m1;
		m2 += d*d*(n-1) / n;
		m1 += d/n;
	}

	return mean_variance_result<T>{.mean = m1, .variance = m2 / (n-ddof) };
}

// (2)
template<class ExecutionPolicy, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto mean_variance(
	ExecutionPolicy&& policy,
	R&& r,
	std::ranges::range_value_t<R> ddof = std::ranges::range_value_t<R>(1)) ->
		mean_variance_result<std::ranges::range_value_t<R>>
{ return std::mean_variance<R>(r, ddof); }

// (3)
template<ranges::input_range R>
constexpr auto mean_standard_deviation(
	R&& r,
	std::ranges::range_value_t<R> ddof = std::ranges::range_value_t<R>(1)) ->
		mean_standard_deviation_result<std::ranges::range_value_t<R>>
{
	using T = std::iter_value_t<R>;

	auto [mean, variance] = std::mean_variance<R>(r, ddof);
	return mean_standard_deviation_result<T>
	{
		.mean               = mean,
		.standard_deviation = static_cast<T>(sqrt_<T>(std::variance(r, ddof)))
	};
}

// (4)
template<class ExecutionPolicy, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto mean_standard_deviation(
	ExecutionPolicy&& policy,
	R&& r,
	std::ranges::range_value_t<R> ddof = std::ranges::range_value_t<R>(1)) ->
		mean_standard_deviation_result<std::ranges::range_value_t<R>>
{ return std::mean_standard_deviation<R>(policy, r, ddof); }

// skewness functions

// (1)
template<ranges::input_range R>
constexpr auto skewness(R&& r, bool sample=true) ->
	std::ranges::range_value_t<R>
{
	using T = std::iter_value_t<R>;

	T d, y1, y2,
	       m1 = T(),
	       m2 = T(),
	       m3 = T();
	size_t n  = 0,
	       nA;

	for (const auto& x : r)
	{
		nA = n;
		++n;
		d    = x-m1;
		y1   = -T(1)/n * d;
		y2   = static_cast<T>(nA)/n * d;
		m3  += nA*y1*y1*y1 + y2*y2*y2 + 3*m2*y1;
		m2  += d*d*(n-1) / n;
		m1  += d/n;
	}

	m3 /= n*static_cast<T>(std::pow(m2/n, 1.5));
	
	if (sample)
		m3 *= static_cast<T>(sqrt_<T>(static_cast<T>(n)*(n-1)) / (n-2));
	
	return m3;
}

// (2)
template<class ExecutionPolicy, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
constexpr auto skewness(ExecutionPolicy&& policy, R&& r, bool sample=true) ->
	std::ranges::range_value_t<R>
{ return std::skewness<R>(r, sample); }

// kurtosis functions

struct kurtosis_parameters { bool sample = true; bool excess = true; };

// (1)
template<ranges::input_range R>
constexpr auto kurtosis(R&& r, kurtosis_parameters params = {}) ->
	std::ranges::range_value_t<R>
{
	using T = std::iter_value_t<R>;

	T d, y1, y12, y2, y22,
	       m1     = T(),
	       m2     = T(),
	       m3     = T(),
	       m4     = T(),
	       sigma2 = T();
	size_t n      = 0,
		   nA;

	for (const auto& x : r)
	{
		nA = n;
		++n;
		d     = x-m1;
		y1    = -T(1)/n * d;
		y12   = y1*y1;
		y2    = static_cast<T>(nA)/n * d;
		y22   = y2*y2;
		m4   += nA*y12*y12 + y22*y22 + 4*m3*y1 + 6*m2*y12;
		m3   += nA*y12*y1 + y22*y2 + 3*m2*y1;
		m2   += d*d*(n-1) / n;
		m1   += d/n;
	}

	if (params.sample)
	{
		sigma2 = m2/(n-1);
		m4    *= static_cast<T>(n)*(n+1) / ((n-1)*(n-2)*(n-3)*sigma2*sigma2);
		
		if (params.excess)
			m4 -= 3*(static_cast<T>(n)-1)*(n-1) / ((n-2)*(n-3));
	}
	else
	{
		sigma2 = m2/n;
		m4    /= (n*sigma2*sigma2);
		
		if (params.excess)
			m4 -= 3;
	}

	return m4;
}

// (2)
template<class ExecutionPolicy, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
constexpr auto kurtosis(
	ExecutionPolicy&& policy, R&& r, kurtosis_parameters params = {}) ->
		std::ranges::range_value_t<R>
{ return std::kurtosis<R>(r, params); }

// covariance functions

template<ranges::input_range R1, ranges::input_range R2>
constexpr auto covariance(
	R1&& r1, R2&& r2,
	std::common_type_t<std::iter_value_t<R1>, std::iter_value_t<R2>> ddof = 1) ->
	std::common_type_t<std::iter_value_t<R1>, std::iter_value_t<R2>>
{
	using T = std::common_type_t<std::iter_value_t<R1>, std::iter_value_t<R2>>;

	T dx, dy,
	       m1x  = T(),
		   m1y  = T(),
		   m2xy = T();
	size_t n    = 0;
	auto   it1  = r1.cbegin();
	auto   it2  = r2.cbegin();

	for (; it1 != r1.cend(); ++it1, ++it2)
	{
		++n;
		dx    = *it1 - m1x;
		dy    = *it2 - m1y;
		m2xy += dx*dy*(static_cast<T>(n)-1) / n;
		m1x  += dx/n;
		m1y  += dy/n;
	}

	return m2xy /= (n-ddof);
}

template<class ExecutionPolicy, ranges::input_range R1, ranges::input_range R2>
	requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
constexpr auto covariance(
	ExecutionPolicy&& policy,
	R1&& r1, R2&& r2,
	std::common_type_t<std::iter_value_t<R1>, std::iter_value_t<R2>> ddof = 1) ->
	std::common_type_t<std::iter_value_t<R1>, std::iter_value_t<R2>>
{ return std::covariance<R1, R2>(r1, r2, ddof); }

}; /* namespace std */