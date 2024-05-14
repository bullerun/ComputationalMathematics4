import math


def linear_func(a, b):
    return lambda x: a * x + b


def poly2_func(a, b, c):
    return lambda x: a * x ** 2 + b * x + c


def poly3_func(a, b, c, d):
    return lambda x: a * x ** 3 + b * x ** 2 + c * x + d


def exp_func(a, b):
    return lambda x: a * math.exp(b * x)


def log_func(a, b):
    return lambda x: a * math.log(x) + b


def power_func(a, b):
    return lambda x: a * x ** b
