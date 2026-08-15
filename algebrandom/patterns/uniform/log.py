from algebrandom.core.pattern import Pattern
import sympy as sp

from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.exponential import Exponential
from algebrandom.instances.uniform import Uniform


class LogUniformPattern(Pattern):
    priority = 10

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        x = self.wc_random_variable("log_uniform", Uniform)
        y = self.wc_constant("c", positive=True)

        pattern = -(1 / y) * sp.log(x)

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        rv = matched.get_rv(x, Uniform)
        if rv.min != 0 or rv.max != 1:
            return None

        const = matched.get_constant(y)
        log_rv = Exponential(const, expression=matched.pattern)
        return matched.pattern, log_rv
