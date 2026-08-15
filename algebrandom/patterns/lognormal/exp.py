import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.lognormal import LogNormal
from algebrandom.instances.normal import Normal


class ExpNormalPattern(Pattern):
    priority = 10

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        rv_wildcard = self.wc_random_variable("exp_normal", Normal)
        pattern = sp.exp(rv_wildcard)

        matched = self.get_matched_pattern(expr, pattern) # type: ignore
        if matched is None:
            return None

        rv = matched.get_rv(rv_wildcard, Normal)
        reduced = LogNormal(rv.mu, rv.sigma,  rv.name , expression=matched.pattern)
        return matched.pattern, reduced
