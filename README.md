# algebrandom

algebrandom is a small library for manipulating random variables as if they were ordinary numbers. You may add, multiply, divide, exponentiate, and take the log of any random variables. Whenever possible, the system will automatically reduce the expressions utilizing algebraic manipulation and/or known identities of distributions.

## Usage example

```python

import algebrandom as ar

x = ar.instances.Normal(0,1)
y = ar.instances.Normal(0,1)

print((x + y).name())
## Normal(mu=2,sigma=1.41421)

z = x / y

print(z.name())
## Cauchy(location=0,scale=1)

u = x.exp()

print(u.name())
## LogNormal(mu=0,sigma=1)
```

## Installation

Clone the repository and run `pip install .`

```bash
git clone https://github.com/nullspace1/algebrandom.git
cd algebrandom
pip install .
```

## License

MIT

## TODOs

- Add more distributions
- Add more operations (e.g. sin, cos, etc.)
- Improve algebraic expression manipulation - currently limited by python's stack size as each operation adds a new layer to the expression tree
- Add more identities between random variables.
