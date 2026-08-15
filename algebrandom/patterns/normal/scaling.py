import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.normal import Normal


class NormalScalingPattern(Pattern):

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        coefficient = self.wc_constant("normal_scale")
        rv_wildcard = self.wc_random_variable("scaled_normal", Normal)
        pattern = coefficient * rv_wildcard

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        scale = matched.get_constant(coefficient)
        if scale == 1:
            return None

        rv = matched.get_rv(rv_wildcard, Normal)
        scaled = Normal(
            scale * rv.mu,
            abs(scale) * rv.sigma,
            rv.name + "*" +  str(scale),
            expression=matched.pattern,
        )
        return matched.pattern, scaled
