import math

import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.beta import specialize_beta
from algebrandom.instances.f import F


class FReciprocalPattern(Pattern):
    priority = 20

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        rv_wildcard = self.wc_random_variable("reciprocal_f", F)
        pattern = 1 / rv_wildcard

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        rv = matched.get_rv(rv_wildcard, F)
        reduced = F(rv.dfd, rv.dfn, "1/" + rv.name, expression=matched.pattern)
        return matched.pattern, reduced


class FToBetaPattern(Pattern):
    priority = 20

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        rv_wildcard = self.wc_random_variable("f_to_beta", F)
        dfn = self.wc_constant("f_beta_dfn", positive=True)
        dfd = self.wc_constant("f_beta_dfd", positive=True)
        pattern = (dfn * rv_wildcard) / (dfn * rv_wildcard + dfd)

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        rv = matched.get_rv(rv_wildcard, F)
        a = matched.get_constant(dfn)
        b = matched.get_constant(dfd)
        if not math.isclose(a, rv.dfn) or not math.isclose(b, rv.dfd):
            return None

        reduced = specialize_beta(rv.dfn / 2, rv.dfd / 2, rv.name + "->beta", expression=matched.pattern)
        return matched.pattern, reduced
