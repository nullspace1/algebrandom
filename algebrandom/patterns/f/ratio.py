import math

import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.chisquare import ChiSquare
from algebrandom.instances.f import F
from algebrandom.instances.gamma import Gamma


class FRatioPattern(Pattern):
    priority = 20

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        result = self._match_chi(expr)
        if result is not None:
            return result
        return self._match_gamma(expr)

    def _match_chi(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        cancelled = self._match_chi_cancelled(expr)
        if cancelled is not None:
            return cancelled
        num = self.wc_random_variable("f_chi_num", ChiSquare)
        den = self.wc_random_variable("f_chi_den", ChiSquare)
        dfn = self.wc_constant("f_dfn", positive=True)
        dfd = self.wc_constant("f_dfd", positive=True)
        pattern = (num / dfn) / (den / dfd)
        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None
        left = matched.get_rv(num, ChiSquare)
        right = matched.get_rv(den, ChiSquare)
        d1 = matched.get_constant(dfn)
        d2 = matched.get_constant(dfd)
        if not math.isclose(left.df, d1) or not math.isclose(right.df, d2):
            return None
        if not left.is_independent_from(right):
            return None
        reduced = F(d1, d2, left.name + "/" + right.name, expression=matched.pattern)
        return matched.pattern, reduced

    def _match_chi_cancelled(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        num = self.wc_random_variable("f_chi_num_c", ChiSquare)
        den = self.wc_random_variable("f_chi_den_c", ChiSquare)
        c1 = self.wc_constant("f_c1", positive=True)
        c2 = self.wc_constant("f_c2", positive=True)
        matched = self.get_matched_pattern(expr, (c1 * num) / (c2 * den))
        scale = None
        if matched is None:
            c = self.wc_constant("f_c", positive=True)
            matched = self.get_matched_pattern(expr, c * num / den)
            if matched is None:
                matched = self.get_matched_pattern(expr, num / den)
                scale = 1.0
            else:
                scale = matched.get_constant(c)
        else:
            scale = matched.get_constant(c1) / matched.get_constant(c2)
        if matched is None:
            return None
        left = matched.get_rv(num, ChiSquare)
        right = matched.get_rv(den, ChiSquare)
        if not math.isclose(scale, right.df / left.df):
            return None
        if not left.is_independent_from(right):
            return None
        reduced = F(left.df, right.df, left.name + "/" + right.name, expression=matched.pattern)
        return matched.pattern, reduced

    def _match_gamma(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        num = self.wc_random_variable("f_g_num", Gamma)
        den = self.wc_random_variable("f_g_den", Gamma)
        pattern = num / den
        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None
        left = matched.get_rv(num, Gamma)
        right = matched.get_rv(den, Gamma)
        if not math.isclose(left.shape, left.rate) or not math.isclose(right.shape, right.rate):
            return None
        if not left.is_independent_from(right):
            return None
        d1 = 2 * left.shape
        d2 = 2 * right.shape
        reduced = F(d1, d2, left.name + "/" + right.name, expression=matched.pattern)
        return matched.pattern, reduced
