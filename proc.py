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
    d = 2 + random.randint(-sz, sz)
    a = [d + i for i in range(sz)]
    p = [0] * sz

    for _ in range(m):
        r = random.randint(0, sz - 1)
        p[r] += 1

    return a, p

# Вместо order использовать build-in sort(vec)

def visk(n, l):
    """Вычисление по формуле включения-исключения"""
    j = 1
    a = 0
    b = 1

    for i in range(l):
        a += j * b * (l-i)**n
        j *= -1
        b = b * (l-i) // (i + 1)
    return a

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
    file = open(input_dir.with_suffix('.prm'), 'r', encoding='Windows-1251')
    prm_template = file.read()
    file.close()

    file = open(input_dir.with_suffix('.sln'), 'r', encoding='UTF-8')
    sln_template = file.read()
    file.close()

    # Подстановка в шаблон задачи
    sln_filled = sln_template
    for val in sln_params:
        sln_filled = sln_filled.replace('\\p', str(val), 1)

    prm_filled = prm_template
    for val in prm_params:
        prm_filled = prm_filled.replace('\\p', str(val), 1)

    # Выгрузка шаблонов
    output_dir.mkdir(parents=True, exist_ok=True)

    file = open(output_dir.joinpath(f"{output_suf}-data.txt"), 'w', encoding='Windows-1251')
    file.write(prm_filled)
    file.close()

    file = open(output_dir.joinpath(f"{output_suf}-answer.txt"), 'w', encoding='UTF-8')
    file.write(sln_filled)
    file.close()