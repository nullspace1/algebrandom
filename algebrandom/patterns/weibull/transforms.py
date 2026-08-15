import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.exponential import Exponential
from algebrandom.instances.weibull import Weibull, specialize_weibull


class ExponentialPowerWeibullPattern(Pattern):
    priority = 30

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        rv_wildcard = self.wc_random_variable("exp_pow", Exponential)
        exponent = self.wc_constant("weibull_exp", positive=True)
        pattern = rv_wildcard ** exponent

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        power = matched.get_constant(exponent)
        if power == 1:
            return None

        rv = matched.get_rv(rv_wildcard, Exponential)
        shape = 1 / power
        scale = rv.rate ** (-power)
        reduced = specialize_weibull(shape, scale, "(" + rv.name + ")^" + str(power), expression=matched.pattern)
        return matched.pattern, reduced


class WeibullScalingPattern(Pattern):

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        coefficient = self.wc_constant("weibull_scale", positive=True)
        rv_wildcard = self.wc_random_variable("scaled_weibull", Weibull)
        pattern = coefficient * rv_wildcard

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        scale = matched.get_constant(coefficient)
        if scale == 1:
            return None

        rv = matched.get_rv(rv_wildcard, Weibull)
        reduced = specialize_weibull(rv.shape, scale * rv.scale, rv.name + "*" + str(scale), expression=matched.pattern)
        return matched.pattern, reduced


class WeibullPowerPattern(Pattern):
    priority = 30

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        rv_wildcard = self.wc_random_variable("pow_weibull", Weibull)
        exponent = self.wc_constant("weibull_power", positive=True)
        pattern = rv_wildcard ** exponent

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        power = matched.get_constant(exponent)
        if power == 1:
            return None

        rv = matched.get_rv(rv_wildcard, Weibull)
        reduced = specialize_weibull(rv.shape / power, rv.scale ** power, "(" + rv.name + ")^" + str(power), expression=matched.pattern)
        return matched.pattern, reduced
