import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.normal import Normal


class NormalAdditionPattern(Pattern):

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        left_wildcard = self.wc_random_variable("left_normal", Normal)
        right_wildcard = self.wc_random_variable("right_normal", Normal)
        pattern = left_wildcard + right_wildcard

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        left = matched.get_rv(left_wildcard, Normal)
        right = matched.get_rv(right_wildcard, Normal)
        if not left.is_independent_from(right):
            return None

        added = Normal(
            left.mu + right.mu,
            (left.sigma ** 2 + right.sigma ** 2) ** 0.5,
            left.name + "+" + right.name,
            expression=matched.pattern,
        )
        return matched.pattern, added
