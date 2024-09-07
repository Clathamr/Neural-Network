# from ad import *
import builtins


# Implementation of dual numbers type
class dual:
    def __init__(self, val, grad):
        self.val = val
        self.grad = grad

    def __add__(self, other):
        val = self.val + other.val
        grad = self.grad + other.grad
        return dual(val, grad)

    def __sub__(self, other):
        val = self.val - other.val
        grad = self.grad - other.grad
        return dual(val, grad)

    def __neg__(self):
        val = -self.val
        grad = -self.grad
        return dual(val, grad)

    def __mul__(self, other):
        val = self.val * other.val
        grad = self.grad * other.val + other.grad * self.val
        return dual(val, grad)

    def __str__(self):
        return f'({self.val},{self.grad})'

    def __repr__(self):
        return str(self)


# Override built-in maximum function
def max(u, v):
    if type(u) is not dual:
        return builtins.max(u, v)
    else:
        val = builtins.max(u.val, v.val)
        grad = u.grad * int(u.val > v.val) + v.grad * int(u.val <= v.val)
        return dual(val, grad)


def convnet(x):
    # Define hidden layer parameters
    v1 = dual(-0.3, 0)
    v2 = dual(0.6, 0)
    v3 = dual(1.3, 0)
    v4 = dual(-1.5, 0)
    # Define weights
    w1 = dual(1.2, 1)
    w2 = dual(-0.2, 0)
    # Calculate z1 through z4
    z1 = max(w1 * dual(x[0], 0) + w2 * dual(x[1], 0), dual(0, 0))
    z2 = max(w1 * dual(x[1], 0) + w2 * dual(x[2], 0), dual(0, 0))
    z3 = max(w1 * dual(x[2], 0) + w2 * dual(x[3], 0), dual(0, 0))
    z4 = max(w1 * dual(x[3], 0) + w2 * dual(x[4], 0), dual(0, 0))
    # Assign all z dual numbers to a list
    z = [z1, z2, z3, z4]
    # Calculate output y
    y = max((z1 * v1 + z2 * v2 + z3 * v3 + z4 * v4), dual(0, 0))
    # Return according to required function interface
    return y, z


# Example X inputs
x = [0.3, -1.5, 0.7, 2.1, 0.1]
print("Y outputs:", convnet(x))
