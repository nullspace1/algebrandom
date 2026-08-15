# algebrandom

algebrandom is a small library for manipulating random variables as if they were ordinary numbers. You may add, multiply, divide, exponentiate, and take the log of any random variables. Whenever possible, the system will automatically reduce the expressions utilizing algebraic manipulation and/or known identities of distributions.

Independent variables of the same family often collapse to a named law (two normals add to a normal, a normal over a normal is Cauchy). Shared sources are not treated as independent, so those identities are not applied blindly. If nothing matches, you still have an expression you can sample.

## Use cases

- Derive a closed form instead of writing the transformation by hand (`Z = X + Y` for independent normals, `exp(X)` for a lognormal, and so on).
- Compose several identities in one expression, such as summing normals and then exponentiating, or building a t from a standard normal and a chi-square.
- Fall back to Monte Carlo on the leftover expression when there is no named reduction.
- Check that two constructions that should be the same law actually reduce to the same family (for example `StudentT(1)` and a scaled Cauchy).

## Usage

```python
from algebrandom.instances.normal import Normal

x = Normal(0, 1)
y = Normal(0, 1)

print(x + y)
## Normal[Normal[](0,1)+Normal[](0,1)](0,1.4142135623730951)

z = x / y
print(z)
## Cauchy[Normal[](0,1)/Normal[](0,1)](0.0,1.0)

u = x.exp()
print(u)
## LogNormal[Normal[](0,1)](0,1)
```

Affine maps and several variables work the same way:

```python
from algebrandom.instances.normal import Normal

x = Normal(1, 2)
y = Normal(3, 4)

print(2 * x + 5)
## Normal[...,](7,4)

print(x + y)
## Normal[...,](4, 4.472...)
```

When a reduction exists, sampling and exact summaries use the named law:

```python
from algebrandom.instances.normal import Normal

x = Normal(0, 1)
s = 2 * x + 1

print(s.mean())
## 1.0
print(s.cdf(1))
## 0.5
print(s.sample(5))
```

If the result is not a named distribution, `sample`, `mean`, `cdf`, and related methods still run by drawing the expression.

### More identities

```python
from algebrandom.instances.chisquare import ChiSquare
from algebrandom.instances.exponential import Exponential
from algebrandom.instances.gamma import Gamma
from algebrandom.instances.lognormal import LogNormal
from algebrandom.instances.normal import Normal
from algebrandom.instances.uniform import Uniform

print((Normal(1, 2) + Normal(3, 4)).exp())
## LogNormal with mu=4, sigma=sqrt(20)

print(LogNormal(1, 2) * LogNormal(3, 4))
## LogNormal with mu=4, sigma=sqrt(20)

print(-Uniform(0, 1, "u").log())
## Exponential with rate 1

print(Exponential(2) + Exponential(2))
## Gamma with shape 2, rate 2

print(Gamma(2, 1) / (Gamma(2, 1) + Gamma(3, 1)))
## Beta with alpha=2, beta=3

print(Normal(0, 1) ** 2)
## ChiSquare with 1 df

print(Normal(0, 1) / (ChiSquare(4) / 4) ** 0.5)
## StudentT with 4 df
```

Each constructor is independent of the others. Two `Normal(0, 1)` objects are independent; reusing the same object (or tying two variables to the same underlying symbol) is not.

## Distributions

| Class | Module |
| --- | --- |
| `Normal` | `algebrandom.instances.normal` |
| `LogNormal` | `algebrandom.instances.lognormal` |
| `Uniform` | `algebrandom.instances.uniform` |
| `Exponential` | `algebrandom.instances.exponential` |
| `Gamma` | `algebrandom.instances.gamma` |
| `ChiSquare` | `algebrandom.instances.chisquare` |
| `Beta` | `algebrandom.instances.beta` |
| `Cauchy` | `algebrandom.instances.cauchy` |
| `StudentT` | `algebrandom.instances.studentt` |
| `F` | `algebrandom.instances.f` |
| `Weibull` | `algebrandom.instances.weibull` |
| `Pareto` | `algebrandom.instances.pareto` |

Operations on a random variable `x`: `x + y`, `x - y`, `x * y`, `x / y`, `x ** n`, `x.log()`, `x.exp()`, and the same with a constant on either side. Named results expose the usual parameters (`mu`, `sigma`, `rate`, `df`, …).

## Installation

Clone the repository and run `pip install .`

```bash
git clone https://github.com/nullspace1/algebrandom.git
cd algebrandom
pip install .
```

Dependencies are `numpy`, `scipy`, `sympy`, and `matplotlib`. Tests:

```bash
pip install pytest
python -m pytest algebrandom/tests
```

## License

MIT

## TODOs

- Add more distributions
- Add more operations (e.g. sin, cos, etc.)
- Add more identities between random variables
- Plotting support (?)
- Vector valued RV support
- Treatment of discrete random variables

# Usage of AI

AI was utilized while creating this repository. In particular, it was used for:

- Accelerating development (quickly transforming ideas/plan into code to verify behaviour)
- A small amount of brainstorming
- Generating a useful README.md
- Test case generation (test plan -> code pipeline)

All major design decisions (abstractions, general algorithms) were human designed.
