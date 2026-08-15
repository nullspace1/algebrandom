import math

import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.exponential import Exponential
from algebrandom.instances.pareto import Pareto
from algebrandom.instances.uniform import Uniform


class ExpToParetoPattern(Pattern):
    priority = 10

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        
        rv_wildcard = self.wc_random_variable("pareto_exp", Exponential)
        xmin = self.wc_constant("pareto_xmin", positive=True)
        matched = self.get_matched_pattern(expr, xmin * sp.exp(rv_wildcard))
        if matched is not None:
            scale = matched.get_constant(xmin)
            rv = matched.get_rv(rv_wildcard, Exponential)
            reduced = Pareto(scale, rv.rate, str(scale) + "*exp(" + rv.name + ")", expression=matched.pattern)
            return matched.pattern, reduced
        matched = self.get_matched_pattern(expr, sp.exp(rv_wildcard)) # type: ignore
        if matched is None:
            return None
        rv = matched.get_rv(rv_wildcard, Exponential)
        reduced = Pareto(1.0, rv.rate, "exp(" + rv.name + ")", expression=matched.pattern)
        return matched.pattern, reduced


class UniformToParetoPattern(Pattern):
    priority = 10

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        xmin = self.wc_constant("pareto_u_xmin", positive=True)
        rv_wildcard = self.wc_random_variable("pareto_uniform", Uniform)
        exponent = self.wc_constant("pareto_neg_exp")
        matched = self.get_matched_pattern(expr, xmin * rv_wildcard ** exponent)
        scale = None
        if matched is None:
            matched = self.get_matched_pattern(expr, rv_wildcard ** exponent)
            scale = 1.0
        if matched is None:
            return None

        power = matched.get_constant(exponent)
        if power >= 0:
            return None

        rv = matched.get_rv(rv_wildcard, Uniform)
        if rv.min != 0 or rv.max != 1:
            return None

        if scale is None:
            scale = matched.get_constant(xmin)
        shape = -1 / power
        reduced = Pareto(scale, shape, str(scale) + "*" + rv.name + "^" + str(power), expression=matched.pattern)
        return matched.pattern, reduced


class ParetoScalingPattern(Pattern):

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        coefficient = self.wc_constant("pareto_scale", positive=True)
        rv_wildcard = self.wc_random_variable("scaled_pareto", Pareto)
        pattern = coefficient * rv_wildcard

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        scale = matched.get_constant(coefficient)
        if scale == 1:
            return None

        rv = matched.get_rv(rv_wildcard, Pareto)
        reduced = Pareto(scale * rv.xmin, rv.shape, rv.name + "*" + str(scale), expression=matched.pattern)
        return matched.pattern, reduced


class ParetoPowerPattern(Pattern):
    priority = 30

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        rv_wildcard = self.wc_random_variable("pow_pareto", Pareto)
        exponent = self.wc_constant("pareto_power", positive=True)
        pattern = rv_wildcard ** exponent

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        power = matched.get_constant(exponent)
        if power == 1:
            return None

        rv = matched.get_rv(rv_wildcard, Pareto)
        reduced = Pareto(rv.xmin ** power, rv.shape / power, "(" + rv.name + ")^" + str(power), expression=matched.pattern)
        return matched.pattern, reduced


class ParetoLogPattern(Pattern):
    priority = 10

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        rv_wildcard = self.wc_random_variable("log_pareto", Pareto)
        xmin = self.wc_constant("log_pareto_xmin", positive=True)
        pattern = sp.log(rv_wildcard / xmin)

        matched = self.get_matched_pattern(expr, pattern) # type: ignore
        if matched is None:
            return None

        rv = matched.get_rv(rv_wildcard, Pareto)
        scale = matched.get_constant(xmin)
        if not math.isclose(scale, rv.xmin):
            return None

        reduced = Exponential(rv.shape, "log(" + rv.name + "/" + str(scale) + ")", expression=matched.pattern)
        return matched.pattern, reduced
