import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.beta import Beta, specialize_beta

class BetaComplementPattern(Pattern):
    priority = 20

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        rv_wildcard = self.wc_random_variable("beta_complement", Beta)
        pattern = 1 - rv_wildcard

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        rv = matched.get_rv(rv_wildcard, Beta)
        reduced = specialize_beta(rv.beta, rv.alpha, "1-" + rv.name, expression=matched.pattern)
        return matched.pattern, reduced
