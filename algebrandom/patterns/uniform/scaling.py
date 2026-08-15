from algebrandom.core.pattern import Pattern
import sympy as sp

from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.uniform import Uniform

class UniformScalingPattern(Pattern):
    
    
    def match(self, expr : sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        coefficient = self.wc_constant("uniform_scale")
        rv_wildcard = self.wc_random_variable("scaled_uniform", Uniform)
        pattern = coefficient * rv_wildcard

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        scale = matched.get_constant(coefficient)
        if scale == 1:
            return None

        rv = matched.get_rv(rv_wildcard, Uniform)
        lo = scale * rv.min
        hi = scale * rv.max
        if lo > hi:
            lo, hi = hi, lo
        scaled = Uniform(
            lo,
            hi,
            rv.name + "*" + str(scale),
            expression=matched.pattern,
        )
        return matched.pattern, scaled