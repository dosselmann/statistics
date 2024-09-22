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

/* ======================================================================== */

namespace std {

// helper function (implementation specific)

template<class T>
inline constexpr T sqrt_(T x)
{ return (static_cast<T>(x) < T()) ? T() : static_cast<T>(std::sqrt(x)); }

/* functions */

// mean functions

// (1)
template<class T, ranges::input_range R>
constexpr auto mean(R&& r) -> T
{
	T      m1 = T();
	size_t n  = 0;

	for (auto& x : r)
	{
		++n;
		m1 += (static_cast<T>(x) - m1) / n;
	}

	return m1;
}

// (2)
template<ranges::input_range R>
constexpr auto mean(R&& r) -> std::ranges::range_value_t<R>
{ return std::mean<std::ranges::range_value_t<R>, R>(r); }

// (3)
template<class T, ranges::input_range R, ranges::input_range W>
constexpr auto mean(R&& r, W&& w) -> T
{
	T m1 = T(), w1 = T();
	auto  it1 = r.cbegin();
	auto  it2 = w.cbegin();
	
	for (; it1 != r.cend(); ++it1, ++it2)
	{
		T w2 = static_cast<T>(*it2);
		w1  += w2;
		m1  += w2/w1 * (static_cast<T>(*it1) - m1);
	}

	return m1;
}

// (4)
template<ranges::input_range R, ranges::input_range W>
constexpr auto mean(R&& r, W&& w) -> std::ranges::range_value_t<R>
{ return std::mean<std::ranges::range_value_t<R>, R, W>(r, w); }

// (5)
template<class ExecutionPolicy, class T, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto mean(ExecutionPolicy&& policy, R&& r) -> T
{ return std::mean<T, R>(r); }

// (6)
template<class ExecutionPolicy, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto mean(ExecutionPolicy&& policy, R&& r) -> std::ranges::range_value_t<R>
{
	return std::mean<ExecutionPolicy, std::ranges::range_value_t<R>, R>(
		policy, r);
}

// (7)
template<class ExecutionPolicy,
         class T,
         ranges::input_range R,
         ranges::input_range W>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto mean(ExecutionPolicy&& policy, R&& r, W&& w) -> T
{ return std::mean<T, R, W>(r, w); }

// (8)
template<class ExecutionPolicy, ranges::input_range R, ranges::input_range W>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto mean(ExecutionPolicy&& policy, R&& r, W&& w) ->
	std::ranges::range_value_t<R>
{
	return std::mean<ExecutionPolicy, std::ranges::range_value_t<R>, R, W>(
		policy, r, w);
}

// (9)
template<class T, ranges::input_range R>
constexpr auto geometric_mean(R&& r) -> T
{
	T      m1 = T();
	size_t n  = 0;

	for (const auto& x : r)
	{
		++n;
		m1 += (static_cast<T>(std::log(static_cast<T>(x))) - m1) / n;
	}

	return static_cast<T>(std::exp(m1));
}

// (10)
template<ranges::input_range R>
constexpr auto geometric_mean(R&& r) -> std::ranges::range_value_t<R>
{ return std::geometric_mean<std::ranges::range_value_t<R>, R>(r); }

// (11)
template<class T, ranges::input_range R, ranges::input_range W>
constexpr auto geometric_mean(R&& r, W&& w) -> T
{
	T m1 = T(), w1 = T();
	auto it1 = r.cbegin();
	auto it2 = w.cbegin();

	for (; it1 != r.cend(); ++it1, ++it2)
	{
		T w2 = static_cast<T>(*it2);
		w1  += w2;
		m1  += w2/w1 * (static_cast<T>(std::log(*it1)) - m1);
	}
	
	return static_cast<T>(std::exp(m1));
}

// (12)
template<ranges::input_range R, ranges::input_range W>
constexpr auto geometric_mean(R&& r, W&& w) -> std::ranges::range_value_t<R>
{ return std::geometric_mean<std::ranges::range_value_t<R>, R, W>(r, w); }

// (13)
template<class ExecutionPolicy, class T, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto geometric_mean(ExecutionPolicy&& policy, R&& r) -> T
{ return std::geometric_mean<T, R>(r); }

// (14)
template<class ExecutionPolicy, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto geometric_mean(ExecutionPolicy&& policy, R&& r) ->
	std::ranges::range_value_t<R>
{
	return std::geometric_mean<
		ExecutionPolicy, std::ranges::range_value_t<R>, R>(policy, r);
}

// (15)
template<class ExecutionPolicy,
         class T,
         ranges::input_range R,
         ranges::input_range W>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto geometric_mean(ExecutionPolicy&& policy, R&& r, W&& w) -> T
{ return geometric_mean<T, R, W>(r, w); }

// (16)
template<class ExecutionPolicy, ranges::input_range R, ranges::input_range W>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto geometric_mean(ExecutionPolicy&& policy, R&& r, W&& w) ->
std::ranges::range_value_t<R>
{
	return geometric_mean<
		ExecutionPolicy, std::ranges::range_value_t<R>, R, W>(policy, r, w);
}

// (17)
template<class T, ranges::input_range R>
constexpr auto harmonic_mean(R&& r) -> T
{
	T      m1 = T();
	size_t n  = 0;

	for (const auto& x : r)
	{
		++n;
		m1 += (T(1) / static_cast<T>(x) - m1) / n;
	}
	
	return T(1)/m1;
}

// (18)
template<ranges::input_range R>
constexpr auto harmonic_mean(R&& r) -> std::ranges::range_value_t<R>
{ return std::harmonic_mean<std::ranges::range_value_t<R>, R>(r); }

// (19)
template<class T, ranges::input_range R, ranges::input_range W>
constexpr auto harmonic_mean(R&& r, W&& w) -> T
{
	T m1 = T(), w1 = T();
	auto it1 = r.cbegin();
	auto it2 = w.cbegin();

	for (; it1 != r.cend(); ++it1, ++it2)
	{
		T w2 = static_cast<T>(*it2);
		w1  += w2;
		m1  += w2/w1 * (T(1) / static_cast<T>(*it1) - m1);
	}

	return w1/m1;
}

// (20)
template<ranges::input_range R, ranges::input_range W>
constexpr auto harmonic_mean(R&& r, W&& w) -> std::ranges::range_value_t<R>
{ return std::harmonic_mean<std::ranges::range_value_t<R>, R, W>(r, w); }

// (21)
template<class ExecutionPolicy, class T, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto harmonic_mean(ExecutionPolicy&& policy, R&& r) -> T
{ return std::harmonic_mean<T, R>(r); }

// (22)
template<class ExecutionPolicy, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto harmonic_mean(ExecutionPolicy&& policy, R&& r) ->
	std::ranges::range_value_t<R>
{
	return std::harmonic_mean<
		ExecutionPolicy, std::ranges::range_value_t<R>, R>(policy, r);
}

// (23)
template<class ExecutionPolicy,
         class T,
         ranges::input_range R,
         ranges::input_range W>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto harmonic_mean(ExecutionPolicy&& policy, R&& r, W&& w) -> T
{ return std::harmonic_mean<T, R, W>(r, w); }

// (24)
template<class ExecutionPolicy,
    ranges::input_range R,
    ranges::input_range W>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto harmonic_mean(ExecutionPolicy&& policy, R&& r, W&& w) ->
std::ranges::range_value_t<R>
{
	return std::harmonic_mean<
		ExecutionPolicy, std::ranges::range_value_t<R>, R, W>(policy, r, w);
}

// variance functions

// (1)
template<class T, ranges::input_range R>
constexpr auto variance(R&& r, T ddof = T(1)) -> T
{
	T      m1 = T(), m2 = T();
	size_t n = 0;

	for (const auto& x : r)
	{
		++n;
		T d = static_cast<T>(x) - m1;
		m2 += d*d*(n-1) / n;
		m1 += d/n;
	}

	return m2 / (n-ddof);
}

// (2)
template<ranges::input_range R>
constexpr auto variance(
	R&& r,
	std::ranges::range_value_t<R> ddof = std::ranges::range_value_t<R>(1)) ->
		std::ranges::range_value_t<R>
{ return variance<std::ranges::range_value_t<R>, R>(r, ddof); }

// (3)
template<class ExecutionPolicy, class T, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto variance(ExecutionPolicy&& policy, R&& r, T ddof = T(1)) -> T
{ return std::variance<T, R>(r, ddof); }

// (4)
template<class ExecutionPolicy, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto variance(
	ExecutionPolicy&& policy,
	R&& r,
	std::ranges::range_value_t<R> ddof = std::ranges::range_value_t<R>(1)) ->
		std::ranges::range_value_t<R>
{
	return std::variance<
		ExecutionPolicy, std::ranges::range_value_t<R>, R>(policy, r, ddof);
}

// standard deviation functions

// (1)
template<class T, ranges::input_range R>
constexpr auto standard_deviation(R&& r, T ddof = T(1)) -> T
{ return static_cast<T>(sqrt_<T>(std::variance<T, R>(r, ddof))); }

// (2)
template<ranges::input_range R>
constexpr auto standard_deviation(
	R&& r,
	std::ranges::range_value_t<R> ddof = std::ranges::range_value_t<R>(1)) ->
		std::ranges::range_value_t<R>
{ return std::standard_deviation<std::ranges::range_value_t<R>, R>(r, ddof); }

// (3)
template<class ExecutionPolicy, class T, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto standard_deviation(ExecutionPolicy&& policy, R&& r, T ddof = T(1)) -> T
{ return std::standard_deviation<T, R>(r, ddof); }

// (4)
template<class ExecutionPolicy, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto standard_deviation(
	ExecutionPolicy&& policy,
	R&& r,
	std::ranges::range_value_t<R> ddof = std::ranges::range_value_t<R>(1)) ->
		std::ranges::range_value_t<R>
{
	return std::standard_deviation<
		ExecutionPolicy, std::ranges::range_value_t<R>, R>(policy, r, ddof);
}

// mean, variance, standard deviation convenience functions

// (1)
template<class T, ranges::input_range R>
constexpr auto mean_variance(R&& r, T ddof = T(1)) -> std::pair<T,T>
{
	T m1 = T(), m2 = T();
	size_t n = 0;

	for (const auto& x : r)
	{
		++n;
		T d = static_cast<T>(x) - m1;
		m2 += d*d*(n-1) / n;
		m1 += d/n;
	}

	return std::make_pair(m1, m2 / (n-ddof));
}

// (2)
template<ranges::input_range R>
constexpr auto mean_variance(
	R&& r,
	std::ranges::range_value_t<R> ddof = std::ranges::range_value_t<R>(1)) ->
		std::pair<std::ranges::range_value_t<R>,
			std::ranges::range_value_t<R>>
{ return std::mean_variance<std::ranges::range_value_t<R>, R>(R, ddof); }

// (3)
template<class ExecutionPolicy, class T, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto mean_variance(ExecutionPolicy&& policy, R&& r, T ddof = T(1)) ->
	std::pair<T,T>
{ return std::mean_variance<T, R>(r, ddof); }

// (4)
template<class ExecutionPolicy, ranges::input_range R>
	requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto mean_variance(
	ExecutionPolicy&& policy,
	R&& r,
	std::ranges::range_value_t<R> ddof = std::ranges::range_value_t<R>(1)) ->
		std::pair<std::ranges::range_value_t<R>,
			std::ranges::range_value_t<R>>
{
	return std::mean_variance<
		ExecutionPolicy, std::ranges::range_value_t<R>, R>(policy, r, ddof);
}

// (5)
template<class T, ranges::input_range R>
constexpr auto mean_standard_deviation(R&& r, T ddof = T(1)) ->
	std::pair<T, T>
{
	auto [mean, variance] = std::mean_variance<T, R>(r, ddof);
	return std::make_pair(
		mean, static_cast<T>(sqrt_<T>(std::variance(r, ddof))));
}

// (6)
template<ranges::input_range R>
constexpr auto mean_standard_deviation(
	R&& r,
	std::ranges::range_value_t<R> ddof = std::ranges::range_value_t<R>(1)) ->
		std::pair<std::ranges::range_value_t<R>,
			std::ranges::range_value_t<R>>
{
	return std::mean_standard_deviation<std::ranges::range_value_t<R>, R>(
		r, ddof);
}

// (7)
template<class ExecutionPolicy, class T, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto mean_standard_deviation(
	ExecutionPolicy&& policy, R&& r, T ddof = T(1)) -> std::pair<T, T>
{ return std::mean_standard_deviation<T, R>(r, ddof); }

// (8)
template<class ExecutionPolicy, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
auto mean_standard_deviation(
	ExecutionPolicy&& policy,
	R&& r,
	std::ranges::range_value_t<R> ddof = std::ranges::range_value_t<R>(1)) ->
		std::pair<std::ranges::range_value_t<R>,
			std::ranges::range_value_t<R>>
{
	return std::mean_standard_deviation<
		ExecutionPolicy, std::ranges::range_value_t<R>, R>(policy, r, ddof);
}

// skewness functions

// (1)
template<class T, ranges::input_range R>
constexpr auto skewness(R&& r, bool sample=true) -> T
{
	T m1 = T(), m2 = T(), m3 = T();
	size_t n = 0;

	for (const auto& x : r)
	{
		T nA = n;
		++n;
		T d  = static_cast<T>(x) - m1;
		T y1 = -T(1)/n * d;
		T y2 = static_cast<T>(nA)/n * d;
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
template<ranges::input_range R>
constexpr auto skewness(R&& r, bool sample=true) ->
	std::ranges::range_value_t<R>
{ return std::skewness<std::ranges::range_value_t<R>, R>(r, sample); }

// (3)
template<class ExecutionPolicy, class T, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
constexpr auto skewness(
	ExecutionPolicy&& policy, R&& r, bool sample=true) -> T
{ return std::skewness<T, R>(r, sample); }

// (4)
template<class ExecutionPolicy, ranges::input_range R>
	requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
constexpr auto skewness(
	ExecutionPolicy&& policy, R&& r, bool sample=true) ->
		std::ranges::range_value_t<R>
{
	return std::skewness<
		ExecutionPolicy, std::ranges::range_value_t<R>, R>(policy, r, sample);
}

// kurtosis functions

// (1)
template<class T, ranges::input_range R>
constexpr auto kurtosis(R&& r, bool sample=true, bool excess=true) -> T
{
	T m1 = T(), m2 = T(), m3 = T(), m4 = T(), sigma2;
	size_t n = 0;

	for (const auto& x : r)
	{
		T nA  = n;
		++n;
		T d   = static_cast<T>(x) - m1;
		T y1  = -T(1)/n * d;
		T y12 = y1*y1;
		T y2  = static_cast<T>(nA)/n * d;
		T y22 = y2*y2;
		m4   += nA*y12*y12 + y22*y22 + 4*m3*y1 + 6*m2*y12;
		m3   += nA*y12*y1 + y22*y2 + 3*m2*y1;
		m2   += d*d*(n-1) / n;
		m1   += d/n;
	}

	if (sample)
	{
		sigma2 = m2/(n-1);
		m4    *= static_cast<T>(n)*(n+1) / ((n-1)*(n-2)*(n-3)*sigma2*sigma2);
		
		if (excess)
			m4 -= 3*(static_cast<T>(n)-1)*(n-1) / ((n-2)*(n-3));
	}
	else
	{
		sigma2 = m2/n;
		m4    /= (n*sigma2*sigma2);
		
		if (excess)
			m4 -= 3;
	}

	return m4;
}

// (2)
template<ranges::input_range R>
constexpr auto kurtosis(R&& r, bool sample=true, bool excess=true) ->
	std::ranges::range_value_t<R>
{ return std::kurtosis<std::ranges::range_value_t<R>, R>(r, sample, excess); }

// (3)
template<class ExecutionPolicy, class T, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
constexpr auto kurtosis(
	ExecutionPolicy&& policy, R&& r, bool sample=true, bool excess=true) -> T
{ return std::kurtosis<T, R>(r, sample, excess); }

// (4)
template<class ExecutionPolicy, ranges::input_range R>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
constexpr auto kurtosis(
	ExecutionPolicy&& policy, R&& r, bool sample=true, bool excess=true) ->
	std::ranges::range_value_t<R>
{
	return std::kurtosis<ExecutionPolicy, std::ranges::range_value_t<R>, R>(
		policy, r, sample, excess);
}

/* accumulators */

const double EPSILON = 0.000'000'1; // implementation-specific

// unweighted accumulator

template<class T>
class unweighted_accumulator
{
public:
	constexpr unweighted_accumulator() noexcept
	{
		m1_ = m2_ = m3_ = m4_ = T();
		n_                    = 0;
	}

	constexpr size_t count() const noexcept { return n_; }

	constexpr void operator()(const T& x)
	{
		T nA   = n_;
		++n_;
		T d    = static_cast<T>(x) - m1_;
		T y1   = -T(1)/n_ * d;
		T y12  = y1*y1;
		T y2   = static_cast<T>(nA)/n_ * d;
		T y22  = y2*y2;
		m4_   += nA*y12*y12 + y22*y22 + 4*m3_*y1 + 6*m2_*y12;
		m3_   += nA*y12*y1 + y22*y2 + 3*m2_*y1;
		m2_   += d*d*(n_-1) / n_;
		m1_   += d/n_;
	}

	template<ranges::input_range R>
	constexpr void operator()(R&& r)
	{
		for (const T& x : r)
			(*this)(x);
	}

	constexpr auto second_central_moment() const noexcept -> T { return m2_; }
	constexpr auto third_central_moment()  const noexcept -> T { return m3_; }
	constexpr auto fourth_central_moment() const noexcept -> T { return m4_; }

	constexpr auto mean() const noexcept -> T { return m1_; }
	
	constexpr auto variance(T ddof = T(1)) const noexcept -> T
	{
		return m2_ / (n_-ddof);
	}
	
	constexpr auto standard_deviation(T ddof = T(1)) const noexcept -> T
	{
		return static_cast<T>(sqrt_<T>(variance(ddof)));
	}
	
	constexpr auto skewness(bool sample=true) const noexcept -> T
	{
		T skewness_ = m3_ / n_ * static_cast<T>(std::pow(m2_/n_, 1.5));
		
		if (sample)
			skewness_ *= static_cast<T>(
				sqrt_<T>(static_cast<T>(n_)*(n_-1)) / (n_-2));

		return skewness_;
	}
	
	constexpr auto kurtosis(
		bool sample=true, bool excess=true) const noexcept -> T
	{
		T kurtosis_, sigma2;
	
		if (sample)
		{
			sigma2    = m2_/(n_-1);
			kurtosis_ = m4_*n_*(n_+1) / ((n_-1)*(n_-2)*(n_-3)*sigma2*sigma2);
			
			if (excess)
				kurtosis_ -= 3*(
					static_cast<T>(n_)-1)*(n_-1) / ((n_-2)*(n_-3));
		}
		else
		{
			sigma2    = m2_/n_;
			kurtosis_ = m4_/(n_*sigma2*sigma2);
			
			if (excess)
				kurtosis_ -= 3;
		}

		return kurtosis_;
	}
private:
	T      m1_, m2_, m3_, m4_;
	size_t n_;
};

// unweighted accumulator

template<class T, class W = T>
class weighted_accumulator
{
public:
	constexpr weighted_accumulator() noexcept
	{
		m1_ = m2_   = T();
		w_  = w_sq_ = W();
		n_          = 0;
	}

	constexpr W      sum_of_weights()         const noexcept { return w_; }
	constexpr W      sum_of_squared_weights() const noexcept { return w_sq_; }
	constexpr size_t non_zero_count()         const noexcept { return n_; }

	constexpr void operator()(const T& x, const W& w)
	{
		T wA     = w_;
		T w1     = static_cast<T>(w);
		w_      += w1;
		w_sq_   += w1*w1;
		T delta  = static_cast<T>(x) - m1_;
		T y1     = -w1/w_ * delta;
		T y2     = wA/w_ * delta;
		m2_     += wA*y1*y1 + w1*y2*y2;
		m1_     -= y1;

		if (static_cast<double>(w) < EPSILON)
			++n_;
	}

	template<ranges::input_range R, ranges::input_range W>
	constexpr void operator()(R&& r, W&& w)
	{
		auto it  = ranges::begin(r);
		auto it_ = ranges::begin(w);

		for (; it != ranges::end(r); ++it)
		{
			(*this)(*it, *it_);
			++it_;
		}
	}

	constexpr auto second_central_moment() const noexcept -> T { return m2_; }

	constexpr auto mean() const noexcept -> T { return m1_; }
private:
	T      m1_, m2_;
	W      w_, w_sq_;
	size_t n_;
};

}; /* namespace std */