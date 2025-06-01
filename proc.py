import math
import random
import fractions
import os
from pathlib import Path

# число размещений A_n^p
def arrange(n, p):
    if n < p:
        return 0
    return math.prod(n - i for i in range(p))

def sqn(x):
    if x == 0:
        return 0
    return abs(x)/x

# Число сочетаний С_n^p
binomial = math.comb
# Вместо deg/degl использовать build-in (**) ~ x ** y

'''
{ Вход: m - коэффициент дробления вероятностей                           }
{       sz - количество возможных значений                               }
{ Выход-распредедление: a - вектор значений                              }
{                       p - соответствующие вероятности
'''
def discrt(m, sz):
    p = [0] * sz

    for _ in range(m):
        r = random.randint(0, sz - 1)
        p[r] += 1

    return p

# Вместо order использовать build-in sort(vec)

def visk(n, l):
    """Количество строк длины n с ровно l различными цифрами"""
    total = sum((-1)**j * math.comb(l, j) * (l - j)**n for j in range(l+1))
    return total

# Фиксированный формат с обязательной точкой и знаковым пробелом слева
# p1 - длина строки, p2 - точность веществ. числа
def strfix(x, p1=1, p2=4):
    s = f"{x:.{p2}f}"
    if '.' not in s:
        s += '.'

    spaces = max(0, p1 - len(s))
    return ' ' * spaces + s

'''
{ Вывод с окончанием:    3-х, 5-ти, 7-ми       (k<=20)                }
'''
def formatplus(n: int) -> str:
    if n <= 4:
        end = 'х'
    elif n == 7 or n == 8:
        end = 'ми'
    else:
        end = 'ти'

    return f'{n}-{end}'

# Сокращение дроби и расчет вещ.части
def frac(p: int, q: int):
    f = fractions.Fraction(p, q)
    return f.numerator, f.denominator, float(f)


def create_task(path, sln_params, prm_params):
    input_dir = Path(os.path.join('TEMPLATE', path))
    output = Path(os.path.join('OUTPUT', path))
    output_dir = output.parent
    output_suf = output.stem

    # Загрузка шаблонов
    file = open(input_dir.with_suffix('.prm'), 'r', encoding='utf-8')
    prm_template = file.read()
    file.close()

    file = open(input_dir.with_suffix('.sln'), 'r', encoding='utf-8')
    sln_template = file.read()
    file.close()

    # Подстановка в шаблон задачи
    sln_filled = sln_template
    for i, val in enumerate(sln_params):
        if i < 10:
            sln_filled = sln_filled.replace(f'\\p{i}', str(val))
        else:
            sln_filled = sln_filled.replace(f'\\p-{i}', str(val))

    prm_filled = prm_template
    for i, val in enumerate(prm_params):
        if i < 10:
            prm_filled = prm_filled.replace(f'\\p{i}', str(val))
        else:
            prm_filled = prm_filled.replace(f'\\p-{i}', str(val))

    # Выгрузка шаблонов
    output_dir.mkdir(parents=True, exist_ok=True)

    file = open(output_dir.joinpath(f"{output_suf}-data.txt"), 'w', encoding='utf-8')
    file.write(prm_filled)
    file.close()

    file = open(output_dir.joinpath(f"{output_suf}-answer.txt"), 'w', encoding='utf-8')
    file.write(sln_filled)
    file.close()