import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.cauchy import Cauchy


class CauchyShiftPattern(Pattern):

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        shift = self.wc_constant("cauchy_shift")
        rv_wildcard = self.wc_random_variable("shifted_cauchy", Cauchy)
        pattern = shift + rv_wildcard

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        constant = matched.get_constant(shift)
        if constant == 0:
            return None

        rv = matched.get_rv(rv_wildcard, Cauchy)
        shifted = Cauchy(
            rv.x0 + constant,
            rv.gamma,
            rv.name + "+" + str(constant),
            expression=matched.pattern,
        )
        return matched.pattern, shifted
