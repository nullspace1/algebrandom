import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.chisquare import ChiSquare


class ChiSquareAdditionPattern(Pattern):

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        left_wildcard = self.wc_random_variable("left_chi", ChiSquare)
        right_wildcard = self.wc_random_variable("right_chi", ChiSquare)
        pattern = left_wildcard + right_wildcard

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        left = matched.get_rv(left_wildcard, ChiSquare)
        right = matched.get_rv(right_wildcard, ChiSquare)
        if not left.is_independent_from(right):
            return None

        added = ChiSquare(left.df + right.df, left.name + "+" + right.name, expression=matched.pattern)
        return matched.pattern, added
