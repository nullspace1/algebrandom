import sympy as sp

from algebrandom.instances.uniform import Uniform


def test_log_expression_wraps_original_symbol():
    uniform = Uniform(1, 2, "uniform")

    assert uniform.log().expr() == sp.log(uniform.symbol)


def test_exp_expression_wraps_original_symbol():
    uniform = Uniform(1, 2, "uniform")

    assert uniform.exp().expr() == sp.exp(uniform.symbol)


def test_float_power_expression_wraps_original_symbol():
    uniform = Uniform(1, 2, "uniform")

    assert (uniform**0.5).expr() == uniform.symbol**0.5


def test_function_expressions_preserve_nested_structure():
    uniform = Uniform(1, 2, "uniform")
    expression = -(uniform**0.5).log().expr() - uniform.exp().expr()

    assert expression.has(sp.log(uniform.symbol**0.5))
    assert expression.has(sp.exp(uniform.symbol))
