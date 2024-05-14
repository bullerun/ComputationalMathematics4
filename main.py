import math
from functions import *
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()
origins = [
    "http://localhost:8000",
    "http://localhost:4200",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

points1 = [[0, 0],
           [0.4, 0.713],
           [0.8, 1.388],
           [1.2, 1.866],
           [1.6, 1.946],
           [2, 1.667],
           [2.4, 1.272],
           [2.8, 0.928],
           [3.2, 0.673],
           [3.6, 0.495],
           [4, 0.37]]
points2 = [[1, 1],
           [2, 2],
           [3, 3],
           [4, 4],
           [5, 5],
           [6, 6],
           [7, 7],
           [8, 8]]

def solve2(A: list[list[float]], B: list[float]) -> dict:
    determinant = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if determinant == 0:
        return {"error": "Определитель матрицы равен нулю, введите другие точки"}
    determinant1 = B[0] * A[1][1] - A[0][1] * B[1]
    determinant2 = A[0][0] * B[1] - B[0] * A[0][1]
    return {"a": determinant2 / determinant, "b": determinant1 / determinant}


def linear_approximation(x: list[float], y: list[float], n: int) -> dict:
    sx = sum(x)
    sxx = sum(i ** 2 for i in x)
    sy = sum(y)
    sxy = sum(i * j for i, j in zip(x, y))
    res = solve2(
        [[n, sx],
         [sx, sxx]],
        [sy, sxy]
    )

    return res


def prepare_matrix_for_calculation_determinant(A: list[list[float]], B: list[float], pos: int):
    local_A = np.copy(A)
    for i in range(len(A)):
        local_A[i][pos] = B[i]
    return local_A


def calc_det3(A: list[list[float]]) -> float:
    return A[0][0] * A[1][1] * A[2][2] + \
        A[0][1] * A[1][2] * A[2][0] + \
        A[0][2] * A[1][0] * A[2][1] - \
        (A[0][2] * A[1][1] * A[2][0] +
         A[0][1] * A[1][0] * A[2][2] +
         A[0][0] * A[1][2] * A[2][1])


def calc_det4(A: list[list[float]]) -> float:
    res = 0
    for i in range(len(A)):
        new_A = [[], [], []]
        k = A[0][i]
        for j in range(1, len(A)):
            for c in range(len(A)):
                if c != i:
                    new_A[j - 1].append(A[j][c])
        res += (-1 ** (i + 1)) * k * calc_det3(A)
    return res


def solve3(A: list[list[float]], B: list[float]) -> dict:
    determinant = calc_det3(A)
    if determinant == 0:
        return {"error": "Определитель матрицы равен нулю, введите другие точки"}
    determinant1 = calc_det3(prepare_matrix_for_calculation_determinant(A, B, 0))
    determinant2 = calc_det3(prepare_matrix_for_calculation_determinant(A, B, 1))
    determinant3 = calc_det3(prepare_matrix_for_calculation_determinant(A, B, 2))
    return {"a": determinant3 / determinant, "b": determinant2 / determinant, "c": determinant1 / determinant}


def quadra_approximation(x: list[float], y: list[float], n: int):
    sx = sum(x)
    sxx = sum(i ** 2 for i in x)
    sxxx = sum(i ** 3 for i in x)
    sxxxx = sum(i ** 4 for i in x)
    sy = sum(y)
    sxy = sum(i * j for i, j in zip(x, y))
    sxxy = sum(i * i * j for i, j in zip(x, y))
    return solve3(
        [
            [n, sx, sxx],
            [sx, sxx, sxxx],
            [sxx, sxxx, sxxxx]
        ],
        [sy, sxy, sxxy],
    )


def solve4(A: list[list[float]], B: list[float]) -> dict:
    determinant = calc_det4(A)
    if determinant == 0:
        return {"error": "Определитель матрицы равен нулю, введите другие точки"}
    determinant1 = calc_det4(prepare_matrix_for_calculation_determinant(A, B, 0))
    determinant2 = calc_det4(prepare_matrix_for_calculation_determinant(A, B, 1))
    determinant3 = calc_det4(prepare_matrix_for_calculation_determinant(A, B, 2))
    determinant4 = calc_det4(prepare_matrix_for_calculation_determinant(A, B, 3))
    return {"a": determinant4 / determinant, "b": determinant3 / determinant, "c": determinant2 / determinant,
            "d": determinant1 / determinant}


def cub_approximation(x: list[float], y: list[float], n: int):
    sx = sum(x)
    sxx = sum(i ** 2 for i in x)
    sxxx = sum(i ** 3 for i in x)
    sxxxx = sum(i ** 4 for i in x)
    sxxxxx = sum(i ** 5 for i in x)
    sxxxxxx = sum(i ** 6 for i in x)
    sy = sum(y)
    sxy = sum(i * j for i, j in zip(x, y))
    sxxy = sum(i * i * j for i, j in zip(x, y))
    sxxxy = sum(i * i * i * j for i, j in zip(x, y))
    return solve4(
        [
            [n, sx, sxx, sxxx],
            [sx, sxx, sxxx, sxxxx],
            [sxx, sxxx, sxxxx, sxxxxx],
            [sxxx, sxxxx, sxxxxx, sxxxxxx]
        ],
        [sy, sxy, sxxy, sxxxy]
    )


def exp_approximation(x: list[float], y: list[float], n: int):
    res = linear_approximation(x, list(map(math.log, y)), n)
    if "a" in res:
        res["a"] = math.exp(res["a"])
    return res


def log_approximation(x: list[float], y: list[float], n: int):
    return linear_approximation(list(map(math.log, x)), y, n)


def power_approximation(x: list[float], y: list[float], n: int):
    res = linear_approximation(list(map(math.log, x)), list(map(math.log, y)), n)
    if "a" in res:
        res["a"] = math.exp(res["a"])
    return res


functions = [
    (linear_approximation, "Линейная"),
    (quadra_approximation, "Полиноминальная 2-й степени"),
    (cub_approximation, "Полиноминальная 3-й степени"),
    (exp_approximation, "Экспоненциальная"),
    (log_approximation, "Логарифмическая"),
    (power_approximation, "Степенная")
]

functions2 = [
    linear_func,
    poly2_func,
    poly3_func,
    exp_func,
    log_func,
    power_func
]


def compute_coefficient_of_determination(xs, ys, fi, n):
    av_fi = sum(fi(x) for x in xs) / n
    return 1 - sum((y - fi(x)) ** 2 for x, y in zip(xs, ys)) / sum((y - av_fi) ** 2 for y in ys)


def compute_mean_squared_error(x, y, fi, n):
    return math.sqrt(sum(((fi(xi) - yi) ** 2 for xi, yi in zip(x, y))) / n)


def compute_measure_of_deviation(x, y, fi, n):
    epss = [fi(xi) - yi for xi, yi in zip(x, y)]
    return sum((eps ** 2 for eps in epss))


def compute_pearson_correlation(x, y, n):
    av_x = sum(x) / n
    av_y = sum(y) / n
    return sum((x - av_x) * (y - av_y) for x, y in zip(x, y)) / math.sqrt(
        sum((x - av_x) ** 2 for x in x) * sum((y - av_y) ** 2 for y in y))


class Points(BaseModel):
    points: list


@app.post("/approximation")
def approximation(points: Points):
    print(points.points)
    x: list[float] = []
    y: list[float] = []
    for i in points.points:
        x.append(i["x"])
        y.append(i["y"])
    print(x)
    n = len(x)
    if all(map(lambda xi: xi > 0, x)):
        if all(map(lambda yi: yi > 0, y)):
            functions = [
                (linear_approximation, "Линейная"),
                (quadra_approximation, "Полиноминальная 2-й степени"),
                (cub_approximation, "Полиноминальная 3-й степени"),
                (exp_approximation, "Экспоненциальная"),
                (log_approximation, "Логарифмическая"),
                (power_approximation, "Степенная")
            ]
        else:
            functions = [
                (linear_approximation, "Линейная"),
                (quadra_approximation, "Полиноминальная 2-й степени"),
                (cub_approximation, "Полиноминальная 3-й степени"),
                (log_approximation, "Логарифмическая"),
            ]
    else:
        if all(map(lambda yi: yi > 0, y)):
            functions = [
                (linear_approximation, "Линейная"),
                (quadra_approximation, "Полиноминальная 2-й степени"),
                (cub_approximation, "Полиноминальная 3-й степени"),
                (exp_approximation, "Экспоненциальная"),
            ]
        else:
            functions = [
                (linear_approximation, "Линейная"),
                (quadra_approximation, "Полиноминальная 2-й степени"),
                (cub_approximation, "Полиноминальная 3-й степени"),
            ]
    res = []
    for fun, i in functions:
        a = fun(x, y, n)
        a["id"] = i
        if "error" not in a and fun == linear_approximation:
            a["pirs"] = compute_pearson_correlation(x, y, n)
        res.append(a)
    for i in range(len(res)):
        if "error" not in res[i]:
            if res[i]["id"] == "Полиноминальная 2-й степени":
                res[i]["s"] = compute_measure_of_deviation(x, y, functions2[i](res[i]["a"], res[i]["b"], res[i]["c"]),
                                                           n)
                res[i]["mse"] = compute_mean_squared_error(x, y, functions2[i](res[i]["a"], res[i]["b"], res[i]["c"]),
                                                           n)
                res[i]["r2"] = compute_coefficient_of_determination(x, y, functions2[i](res[i]["a"], res[i]["b"],
                                                                                        res[i]["c"]), n)
            elif res[i]["id"] == "Полиноминальная 3-й степени":
                res[i]["s"] = compute_measure_of_deviation(x, y, functions2[i](res[i]["a"], res[i]["b"], res[i]["c"],
                                                                               res[i]["d"]), n)
                res[i]["mse"] = compute_mean_squared_error(x, y, functions2[i](res[i]["a"], res[i]["b"], res[i]["c"],
                                                                               res[i]["d"]), n)
                res[i]["r2"] = compute_coefficient_of_determination(x, y,
                                                                    functions2[i](res[i]["a"], res[i]["b"], res[i]["c"],
                                                                                  res[i]["d"]), n)
            else:
                res[i]["s"] = compute_measure_of_deviation(x, y, functions2[i](res[i]["a"], res[i]["b"]), n)
                res[i]["mse"] = compute_mean_squared_error(x, y, functions2[i](res[i]["a"], res[i]["b"]), n)
                res[i]["r2"] = compute_coefficient_of_determination(x, y, functions2[i](res[i]["a"], res[i]["b"]), n)
    print(res)
    return res

def main():
    x: list[float] = []
    y: list[float] = []
    # for i in points.points:
    #     x.append(i["x"])
    #     y.append(i["y"])
    for i in points2:
        x.append(i[0])
        y.append(i[1])
    print(x)
    n = len(x)
    if all(map(lambda xi: xi > 0, x)):
        if all(map(lambda yi: yi > 0, y)):
            functions = [
                (linear_approximation, "Линейная"),
                (quadra_approximation, "Полиноминальная 2-й степени"),
                (cub_approximation, "Полиноминальная 3-й степени"),
                (exp_approximation, "Экспоненциальная"),
                (log_approximation, "Логарифмическая"),
                (power_approximation, "Степенная")
            ]
        else:
            functions = [
                (linear_approximation, "Линейная"),
                (quadra_approximation, "Полиноминальная 2-й степени"),
                (cub_approximation, "Полиноминальная 3-й степени"),
                (log_approximation, "Логарифмическая"),
            ]
    else:
        if all(map(lambda yi: yi > 0, y)):
            functions = [
                (linear_approximation, "Линейная"),
                (quadra_approximation, "Полиноминальная 2-й степени"),
                (cub_approximation, "Полиноминальная 3-й степени"),
                (exp_approximation, "Экспоненциальная"),
            ]
        else:
            functions = [
                (linear_approximation, "Линейная"),
                (quadra_approximation, "Полиноминальная 2-й степени"),
                (cub_approximation, "Полиноминальная 3-й степени"),
            ]
    res = []
    for fun, i in functions:
        a = fun(x, y, n)
        a["id"] = i
        if "error" not in a and fun == linear_approximation:
            a["pirs"] = compute_pearson_correlation(x, y, n)
        res.append(a)
    for i in range(len(res)):
        if "error" not in res[i]:
            if res[i]["id"] == "Полиноминальная 2-й степени":
                res[i]["s"] = compute_measure_of_deviation(x, y, functions2[i](res[i]["a"], res[i]["b"], res[i]["c"]), n)
                res[i]["mse"] = compute_mean_squared_error(x, y, functions2[i](res[i]["a"], res[i]["b"], res[i]["c"]), n)
                res[i]["r2"] = compute_coefficient_of_determination(x, y, functions2[i](res[i]["a"], res[i]["b"], res[i]["c"]), n)
            elif res[i]["id"] == "Полиноминальная 3-й степени":
                res[i]["s"] = compute_measure_of_deviation(x, y, functions2[i](res[i]["a"], res[i]["b"], res[i]["c"], res[i]["d"]), n)
                res[i]["mse"] = compute_mean_squared_error(x, y, functions2[i](res[i]["a"], res[i]["b"], res[i]["c"], res[i]["d"]), n)
                res[i]["r2"] = compute_coefficient_of_determination(x, y, functions2[i](res[i]["a"], res[i]["b"], res[i]["c"], res[i]["d"]), n)
            else:
                res[i]["s"] = compute_measure_of_deviation(x, y, functions2[i](res[i]["a"], res[i]["b"]), n)
                res[i]["mse"] = compute_mean_squared_error(x, y, functions2[i](res[i]["a"], res[i]["b"]), n)
                res[i]["r2"] = compute_coefficient_of_determination(x, y, functions2[i](res[i]["a"], res[i]["b"]), n)
    return res


if __name__ == '__main__':
    print(main())
