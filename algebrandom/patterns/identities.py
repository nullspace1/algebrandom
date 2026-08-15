import math

import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.gamma import Gamma, specialize_gamma
from algebrandom.instances.studentt import StudentT, specialize_studentt
from algebrandom.instances.weibull import Weibull, specialize_weibull


class GammaIdentityPattern(Pattern):
    priority = 0

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        rv_wildcard = self.wc_random_variable("gamma_id", Gamma)
        matched = self.get_matched_pattern(expr, rv_wildcard)
        if matched is None:
            return None
        rv = matched.get_rv(rv_wildcard, Gamma)
        reduced = specialize_gamma(rv.shape, rv.rate, rv.name, expression=matched.pattern)
        if type(reduced) is Gamma:
            return None
        return matched.pattern, reduced


class WeibullIdentityPattern(Pattern):
    priority = 0

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        rv_wildcard = self.wc_random_variable("weibull_id", Weibull)
        matched = self.get_matched_pattern(expr, rv_wildcard)
        if matched is None:
            return None
        rv = matched.get_rv(rv_wildcard, Weibull)
        if not math.isclose(rv.shape, 1.0):
            return None
        reduced = specialize_weibull(rv.shape, rv.scale, rv.name, expression=matched.pattern)
        return matched.pattern, reduced


class StudentTIdentityPattern(Pattern):
    priority = 0

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        rv_wildcard = self.wc_random_variable("t_id", StudentT)
        matched = self.get_matched_pattern(expr, rv_wildcard)
        if matched is None:
            return None
        rv = matched.get_rv(rv_wildcard, StudentT)
        if not math.isclose(rv.df, 1.0):
            return None
        reduced = specialize_studentt(rv.df, rv.mu, rv.sigma, rv.name, expression=matched.pattern)
        return matched.pattern, reduced
