import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.cauchy import Cauchy


class CauchyAdditionPattern(Pattern):

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        left_wildcard = self.wc_random_variable("left_cauchy", Cauchy)
        right_wildcard = self.wc_random_variable("right_cauchy", Cauchy)
        pattern = left_wildcard + right_wildcard

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        left = matched.get_rv(left_wildcard, Cauchy)
        right = matched.get_rv(right_wildcard, Cauchy)
        if not left.is_independent_from(right):
            return None

        added = Cauchy(
            left.x0 + right.x0,
            left.gamma + right.gamma,
            left.name + "+" + right.name,
            expression=matched.pattern,
        )
        return matched.pattern, added
