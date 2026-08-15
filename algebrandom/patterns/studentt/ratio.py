import math

import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.chisquare import ChiSquare
from algebrandom.instances.gamma import Gamma
from algebrandom.instances.normal import Normal
from algebrandom.instances.studentt import specialize_studentt


class StudentTRatioPattern(Pattern):
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
        n_wildcard = self.wc_random_variable("t_normal", Normal)
        chi_wildcard = self.wc_random_variable("t_chi", ChiSquare)
        nu = self.wc_constant("t_nu", positive=True)
        pattern = n_wildcard / sp.sqrt(chi_wildcard / nu)
        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None
        return self._studentt_from_normal_chi(matched, n_wildcard, chi_wildcard, matched.get_constant(nu))

    def _match_chi_cancelled(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        n_wildcard = self.wc_random_variable("t_normal_c", Normal)
        chi_wildcard = self.wc_random_variable("t_chi_c", ChiSquare)
        k = self.wc_constant("t_k", positive=True)
        exponent = self.wc_constant("t_chi_pow", positive=True)
        matched = self.get_matched_pattern(expr, k * n_wildcard / (chi_wildcard ** exponent))
        if matched is None:
            matched = self.get_matched_pattern(expr, k * n_wildcard / sp.sqrt(chi_wildcard))
            power = 0.5
        else:
            power = matched.get_constant(exponent)
        if matched is None or not math.isclose(power, 0.5):
            return None
        chi = matched.get_rv(chi_wildcard, ChiSquare)
        scale = matched.get_constant(k)
        if not math.isclose(scale ** 2, chi.df):
            return None
        return self._studentt_from_normal_chi(matched, n_wildcard, chi_wildcard, chi.df)

    def _studentt_from_normal_chi(self, matched, n_wildcard, chi_wildcard, df: float) -> tuple[sp.Expr, RandomVariable] | None:
        normal = matched.get_rv(n_wildcard, Normal)
        chi = matched.get_rv(chi_wildcard, ChiSquare)
        if normal.mu != 0 or not math.isclose(normal.sigma, 1.0):
            return None
        if not math.isclose(chi.df, df):
            return None
        if not normal.is_independent_from(chi):
            return None
        reduced = specialize_studentt(df, 0.0, 1.0, normal.name + "/sqrt(" + chi.name + "/" + str(df) + ")", expression=matched.pattern)
        return matched.pattern, reduced

    def _match_gamma(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        n_wildcard = self.wc_random_variable("t_normal_g", Normal)
        g_wildcard = self.wc_random_variable("t_gamma", Gamma)
        exponent = self.wc_constant("t_half", positive=True)
        matched = self.get_matched_pattern(expr, n_wildcard / (g_wildcard ** exponent))
        if matched is None:
            matched = self.get_matched_pattern(expr, n_wildcard / sp.sqrt(g_wildcard))
            power = 0.5
        else:
            power = matched.get_constant(exponent)
        if matched is None:
            return None
        if not math.isclose(power, 0.5):
            return None
        normal = matched.get_rv(n_wildcard, Normal)
        gamma = matched.get_rv(g_wildcard, Gamma)
        if normal.mu != 0 or not math.isclose(normal.sigma, 1.0):
            return None
        if not math.isclose(gamma.shape, gamma.rate):
            return None
        df = 2 * gamma.shape
        if not normal.is_independent_from(gamma):
            return None
        reduced = specialize_studentt(df, 0.0, 1.0, normal.name + "/sqrt(" + gamma.name + ")", expression=matched.pattern)
        return matched.pattern, reduced
