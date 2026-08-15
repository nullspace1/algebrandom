import math

import pytest

from algebrandom.core.log import LogRandomVariable
from algebrandom.core.polynomial import PolynomialRandomVariable
from algebrandom.instances.beta import Beta
from algebrandom.instances.cauchy import Cauchy
from algebrandom.instances.chisquare import ChiSquare
from algebrandom.instances.exponential import Exponential
from algebrandom.instances.f import F
from algebrandom.instances.gamma import Gamma
from algebrandom.instances.lognormal import LogNormal
from algebrandom.instances.normal import Normal
from algebrandom.instances.pareto import Pareto
from algebrandom.instances.studentt import StudentT
from algebrandom.instances.uniform import Uniform
from algebrandom.instances.weibull import Weibull


def test_exp_normal_is_lognormal():
    normal = Normal(1, 2, "n")
    reduced = normal.exp()
    assert isinstance(reduced, LogNormal)
    assert reduced.mu == pytest.approx(1)
    assert reduced.sigma == pytest.approx(2)


def test_log_lognormal_is_normal():
    lognormal = LogNormal(1, 2, "ln")
    reduced = lognormal.log()
    assert isinstance(reduced, Normal)
    assert reduced.mu == pytest.approx(1)
    assert reduced.sigma == pytest.approx(2)


def test_sum_then_exp_is_lognormal_by_resolution_order():
    left = Normal(1, 2, "left")
    right = Normal(3, 4, "right")
    reduced = (left + right).exp()
    assert isinstance(reduced, LogNormal)
    assert reduced.mu == pytest.approx(4)
    assert reduced.sigma == pytest.approx(math.sqrt(20))


def test_lognormal_scale_and_product():
    left = LogNormal(1, 2, "left")
    right = LogNormal(3, 4, "right")
    scaled = 2 * left
    assert isinstance(scaled, LogNormal)
    assert scaled.mu == pytest.approx(1 + math.log(2))
    product = left * right
    assert isinstance(product, LogNormal)
    assert product.mu == pytest.approx(4)
    assert product.sigma == pytest.approx(math.sqrt(20))


def test_dependent_lognormals_are_not_multiplied():
    source = LogNormal(0, 1, "source")
    left = LogNormal(1, 1, "left", expression=source.symbol)
    right = LogNormal(2, 1, "right", expression=source.symbol)
    assert isinstance(left * right, PolynomialRandomVariable)


def test_log_uniform_zero_one_is_exponential():
    uniform = Uniform(0, 1, "u")
    reduced = -uniform.log()
    assert isinstance(reduced, Exponential)
    assert reduced.rate == pytest.approx(1)


def test_log_uniform_other_support_stays_log():
    uniform = Uniform(1, 2, "u")
    reduced = uniform.log()
    assert isinstance(reduced, LogRandomVariable)


def test_exponential_scale_and_sum():
    left = Exponential(2, "left")
    right = Exponential(2, "right")
    scaled = 2 * left
    assert isinstance(scaled, Exponential)
    assert scaled.rate == pytest.approx(1)
    added = left + right
    assert isinstance(added, Gamma)
    assert added.shape == pytest.approx(2)
    assert added.rate == pytest.approx(2)


def test_gamma_one_simplifies_to_exponential():
    reduced = Gamma(1, 3, "g").simplify()
    assert isinstance(reduced, Exponential)
    assert reduced.rate == pytest.approx(3)


def test_chi_square_sum_and_normal_square():
    left = ChiSquare(2, "left")
    right = ChiSquare(3, "right")
    added = left + right
    assert isinstance(added, ChiSquare)
    assert added.df == pytest.approx(5)
    squared = Normal(0, 1, "z") ** 2
    assert isinstance(squared, ChiSquare)
    assert squared.df == pytest.approx(1)


def test_gamma_ratio_and_beta_complement():
    left = Gamma(2, 1, "left")
    right = Gamma(3, 1, "right")
    ratio = left / (left + right)
    assert isinstance(ratio, Beta)
    assert ratio.alpha == pytest.approx(2)
    assert ratio.beta == pytest.approx(3)
    complement = 1 - ratio
    assert isinstance(complement, Beta)
    assert complement.alpha == pytest.approx(3)
    assert complement.beta == pytest.approx(2)


def test_normal_ratio_is_cauchy():
    left = Normal(0, 2, "left")
    right = Normal(0, 4, "right")
    reduced = left / right
    assert isinstance(reduced, Cauchy)
    assert reduced.x0 == pytest.approx(0)
    assert reduced.gamma == pytest.approx(0.5)


def test_student_t_from_normal_and_chi_square():
    normal = Normal(0, 1, "z")
    chi = ChiSquare(4, "chi")
    reduced = normal / (chi / 4) ** 0.5
    assert isinstance(reduced, StudentT)
    assert reduced.df == pytest.approx(4)
    assert reduced.mu == pytest.approx(0)
    assert reduced.sigma == pytest.approx(1)


def test_student_t_one_is_cauchy_and_square_is_f():
    t = StudentT(1, 0, 2, "t").simplify()
    assert isinstance(t, Cauchy)
    assert t.gamma == pytest.approx(2)
    standard = StudentT(5, 0, 1, "t5")
    squared = standard ** 2
    assert isinstance(squared, F)
    assert squared.dfn == pytest.approx(1)
    assert squared.dfd == pytest.approx(5)


def test_f_from_chi_square_ratio_reciprocal_and_beta():
    num = ChiSquare(4, "num")
    den = ChiSquare(6, "den")
    reduced = (num / 4) / (den / 6)
    assert isinstance(reduced, F)
    assert reduced.dfn == pytest.approx(4)
    assert reduced.dfd == pytest.approx(6)
    reciprocal = 1 / reduced
    assert isinstance(reciprocal, F)
    assert reciprocal.dfn == pytest.approx(6)
    assert reciprocal.dfd == pytest.approx(4)
    beta = (reduced.dfn * reduced) / (reduced.dfn * reduced + reduced.dfd)
    assert isinstance(beta, Beta)
    assert beta.alpha == pytest.approx(2)
    assert beta.beta == pytest.approx(3)


def test_weibull_from_exponential_power_and_scale():
    exponential = Exponential(2, "e")
    reduced = exponential ** 0.5
    assert isinstance(reduced, Weibull)
    assert reduced.shape == pytest.approx(2)
    assert reduced.scale == pytest.approx(2 ** -0.5)
    scaled = 3 * reduced
    assert isinstance(scaled, Weibull)
    assert scaled.scale == pytest.approx(3 * reduced.scale)
    identity = Weibull(1, 0.5, "w").simplify()
    assert isinstance(identity, Exponential)
    assert identity.rate == pytest.approx(2)


def test_pareto_from_exp_uniform_and_log():
    exponential = Exponential(3, "e")
    from_exp = 2 * exponential.exp()
    assert isinstance(from_exp, Pareto)
    assert from_exp.xmin == pytest.approx(2)
    assert from_exp.shape == pytest.approx(3)
    uniform = Uniform(0, 1, "u")
    from_uniform = 2 * uniform ** -0.5
    assert isinstance(from_uniform, Pareto)
    assert from_uniform.xmin == pytest.approx(2)
    assert from_uniform.shape == pytest.approx(2)
    logged = (from_exp / from_exp.xmin).log()
    assert isinstance(logged, Exponential)
    assert logged.rate == pytest.approx(3)


def test_negative_exponential_scale_does_not_reduce():
    exponential = Exponential(2, "e")
    reduced = -1 * exponential
    assert not isinstance(reduced, Exponential)
