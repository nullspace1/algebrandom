import math

import pytest
import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.polynomial import PolynomialRandomVariable
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.lognormal import LogNormal
from algebrandom.instances.normal import Normal
from algebrandom.instances.uniform import Uniform
from algebrandom.patterns.normal.addition import NormalAdditionPattern
from algebrandom.patterns.uniform.scaling import UniformScalingPattern


def test_normal_wildcard_matches_only_normal_symbols():
    wildcard = Pattern().wc_random_variable("normal", Normal)
    normal = Normal(0, 1, "normal")
    uniform = Uniform(0, 1, "uniform")

    assert normal.symbol.match(wildcard) is not None
    assert uniform.symbol.match(wildcard) is None


def test_polynomial_only_uses_operand_patterns():
    uniform = Uniform(0, 1, "uniform")
    polynomial = PolynomialRandomVariable(uniform.symbol ** 3)
    pattern_types = {type(pattern) for pattern in polynomial.relevant_patterns()}

    assert UniformScalingPattern in pattern_types
    assert NormalAdditionPattern not in pattern_types



def test_positive_constant_wildcard_rejects_non_positive():
    matcher = Pattern()
    positive = matcher.wc_constant("positive", positive=True)
    any_number = matcher.wc_constant("any")

    assert sp.Integer(2).match(positive) is not None
    assert sp.Integer(0).match(positive) is None
    assert sp.Integer(-3).match(positive) is None
    assert sp.Integer(-3).match(any_number) is not None
    assert sp.Integer(0).match(any_number) is not None


def test_matched_constant_is_returned_as_float():
    matcher = Pattern()
    coefficient = matcher.wc_constant("coefficient")
    normal_wildcard = matcher.wc_random_variable("normal", Normal)
    normal = Normal(0, 1, "normal")

    matched = matcher.get_matched_pattern(2 * normal.symbol, coefficient * normal_wildcard)

    assert matched is not None
    assert matched.get_constant(coefficient) == 2.0
    assert isinstance(matched.get_constant(coefficient), float)


def test_normal_shift():
    original = Normal(3, 4, "original")

    reduced = 2 + original

    assert isinstance(reduced, Normal)
    assert reduced.mu == pytest.approx(5)
    assert reduced.sigma == pytest.approx(4)


@pytest.mark.parametrize(
    ("scale", "expected_mu", "expected_sigma"),
    [(2, 6, 8), (-3, -9, 12)],
)
def test_normal_scaling(scale, expected_mu, expected_sigma):
    original = Normal(3, 4, "original")

    reduced = (scale * original)

    assert isinstance(reduced, Normal)
    assert reduced.mu == pytest.approx(expected_mu)
    assert reduced.sigma == pytest.approx(expected_sigma)
    assert original in reduced.dependencies


def test_independent_normal_addition():
    left = Normal(1, 2, "left")
    right = Normal(3, 4, "right")

    reduced = (left + right)

    assert isinstance(reduced, Normal)
    assert reduced.mu == pytest.approx(4)
    assert reduced.sigma == pytest.approx(math.sqrt(20))
    assert left in reduced.dependencies
    assert right in reduced.dependencies


def test_dependent_normals_are_not_combined_by_addition():
    source = Normal(0, 1, "source")
    left = Normal(1, 2, "left", expression=source.symbol)
    right = Normal(3, 4, "right", expression=source.symbol)

    reduced = (left + right)

    assert not (isinstance(reduced, Normal) and reduced.mu == pytest.approx(4) and reduced.sigma == pytest.approx(math.sqrt(20)))


def test_scaled_independent_normals_compose_with_addition():
    left = Normal(1, 2, "left")
    right = Normal(3, 4, "right")

    scaled_left = (2 * left)
    scaled_right = (2 * right)
    reduced = (scaled_left + scaled_right)

    assert isinstance(reduced, Normal)
    assert reduced.mu == pytest.approx(8)
    assert reduced.sigma == pytest.approx(math.sqrt(80))


def test_multiple_normals_are_iteratively_added():
    normals = [
        Normal(1, 2, "first"),
        Normal(3, 4, "second"),
        Normal(5, 6, "third"),
    ]

    reduced = normals[0] + normals[1] + normals[2]

    assert isinstance(reduced, Normal)
    assert reduced.mu == pytest.approx(9)
    assert reduced.sigma == pytest.approx(math.sqrt(56))
    assert all(normal in reduced.dependencies for normal in normals)


def test_multiple_scaled_normals_are_iteratively_simplified():
    inputs = [
        (2, Normal(1, 2, "first")),
        (-3, Normal(3, 4, "second")),
        (4, Normal(5, 6, "third")),
    ]

    scaled = [coefficient * normal for coefficient, normal in inputs]
    reduced = scaled[0] + scaled[1] + scaled[2]

    assert isinstance(reduced, Normal)
    assert reduced.mu == pytest.approx(13)
    assert reduced.sigma == pytest.approx(math.sqrt(736))
    assert all(normal in reduced.dependencies for _, normal in inputs)


def test_all_operations_return_random_variables_and_auto_simplify():
    left = Normal(1, 2, "left")
    right = Normal(3, 4, "right")

    operations = [
        left + right,
        left - right,
        left * right,
        left / right,
        2 + left,
        2 - left,
        2 * left,
        2 / left,
        left**2,
        left**0.5,
        2**left,
        left.log(),
        left.exp(),
    ]

    assert all(isinstance(result, RandomVariable) for result in operations)
    assert isinstance(left + right, Normal)
    assert isinstance(2 * left, Normal)
    assert isinstance(2 + left, Normal)
    assert isinstance(left.exp(), LogNormal)
    assert isinstance(2**left, LogNormal)


def test_ratio_times_denominator_cancels():
    x = Normal(0, 1, "x")
    y = Normal(0, 1, "y")
    reduced = (x / y) * y
    assert reduced is x or (isinstance(reduced, Normal) and reduced.expr() == x.expr())


def test_cancelled_ratio_keeps_other_factor_not_cauchy():
    x = Normal(0, 1, "x")
    y = Normal(0, 1, "y")
    z = Normal(0, 1, "z")
    reduced = z * (x / y) * y
    assert isinstance(reduced, PolynomialRandomVariable)
    operands = reduced._rvs
    assert x in operands
    assert z in operands
    assert y not in operands
    assert all(type(rv).__name__ != "Cauchy" for rv in operands)
    assert sp.simplify(reduced.expr() - z.symbol * x.symbol) == 0


def test_self_scale_merges_before_addition():
    x = Normal(1, 2, "x")
    reduced = x + 2 * x
    assert isinstance(reduced, Normal)
    assert reduced.mu == pytest.approx(3)
    assert reduced.sigma == pytest.approx(6)

