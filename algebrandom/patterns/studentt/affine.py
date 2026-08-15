import math

import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.studentt import StudentT, specialize_studentt


class StudentTScalingPattern(Pattern):

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        coefficient = self.wc_constant("t_scale")
        rv_wildcard = self.wc_random_variable("scaled_t", StudentT)
        pattern = coefficient * rv_wildcard

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        scale = matched.get_constant(coefficient)
        if scale == 1:
            return None

        rv = matched.get_rv(rv_wildcard, StudentT)
        reduced = specialize_studentt(
            rv.df,
            scale * rv.mu,
            abs(scale) * rv.sigma,
            rv.name + "*" + str(scale),
            {rv},
        )
        return matched.pattern, reduced


class StudentTShiftPattern(Pattern):

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        shift = self.wc_constant("t_shift")
        rv_wildcard = self.wc_random_variable("shifted_t", StudentT)
        pattern = shift + rv_wildcard

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        constant = matched.get_constant(shift)
        if constant == 0:
            return None

        rv = matched.get_rv(rv_wildcard, StudentT)
        reduced = specialize_studentt(rv.df, rv.mu + constant, rv.sigma, rv.name + "+" + str(constant), expression=matched.pattern)
        return matched.pattern, reduced


class StudentTSquarePattern(Pattern):
    priority = 20

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        rv_wildcard = self.wc_random_variable("squared_t", StudentT)
        pattern = rv_wildcard ** 2

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        rv = matched.get_rv(rv_wildcard, StudentT)
        if rv.mu != 0 or not math.isclose(rv.sigma, 1.0):
            return None

        from algebrandom.instances.f import F

        reduced = F(1.0, rv.df, rv.name + "^2", expression=matched.pattern)
        return matched.pattern, reduced
