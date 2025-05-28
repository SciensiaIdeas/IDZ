import math
import random

from proc import *


def ptc31():
    a = [30, 50, 60, 70, 80, 90]
    b = [3, 4, 5, 6, 7, 8, 10, 12]
    n = random.choice(a)
    i = random.choice(b)
    l = n // 2

    total = binomial(n, i)
    favorable = binomial(l, i)
    p, q, r = frac(favorable, total)
    res1 = [l, i, n, i, p, q, strfix(r)]
    res2 = [n, i]
    create_task('PT1-1/ptc3-1', res1, res2)

def ptc32():
    b = [3, 4, 5, 6, 7, 8, 10, 12]
    n = random.choice(b)
    res2 = [n]
    j = 54 // n
    l = j - 2
    n1 = 54 * 53
    i = j * (j - 1)

    p, q, r = frac(i, n1)
    res1 = [l, j, j, p, q, strfix(r)]
    create_task('PT1-1/ptc3-2', res1, res2)

def ptc33():
    a = [25, 30, 35, 40, 45, 50]
    b = [6, 8, 10, 12, 15, 18, 20]
    n = random.choice(a)
    j = random.choice(b)
    res2 = [n, j]

    k = n - 3
    k1 = j - 3
    p, q, r = frac(arrange(j, 3), arrange(n, 3))
    res1 = [k, k1, n, j, j, n, p, q, strfix(r)]
    create_task('PT1-1/ptc3-3', res1, res2)

def ptc34():
    d = [('параллелипипед', 288, 14),('тетраэдр', 4, 8),('математика', 24, 10),('алгебра', 2, 7),('параллелограмм', 144, 14),('плоскость', 4, 9)]
    pair = random.choice(d)
    res2 = [pair[0]]

    r = math.factorial(pair[2]) // pair[1]
    res1 = [pair[1], pair[2], r]
    create_task('PT1-1/ptc3-4', res1, res2)

def ptc35():
    b = [5, 6, 7, 8, 9]
    c = [2, 3, 4]
    n = random.choice(b)
    k = random.choice(c)
    res2 = [n, k]

    i = 6
    a = binomial(i, k)
    k1, n1, f = frac(k, i)
    r = a * f**n
    res1 = [k1, n1, n, k, strfix(r)]
    create_task('PT1-1/ptc3-5', res1, res2)

def ptc36():
    b = [5, 6, 7, 8, 9]
    c = [2, 3, 4]
    n = random.choice(b)
    k = random.choice(c)
    res2 = [n, k]

    i = 10
    a = binomial(i, k)
    k1, n1, f = frac(k, i)
    r = a * f**n
    res1 = [k1, n1, n, k, strfix(r)]
    create_task('PT1-1/ptc3-6', res1, res2)

def ptc37():
    b = [5, 6, 7, 8, 9]
    c = [3, 4]
    n = random.choice(b)
    k = random.choice(b)
    l = random.choice(c)
    res2 = [n, k, l]

    n1 = visk(n, l)
    k1 = visk(k, l)
    p = n1 * k1
    q = l ** (n+k)
    p, q, r = frac(p, q)
    res1 = [l-1, l, l, n, l-1, l, l, k, l, n, k, p, q, strfix(r)]
    create_task('PT1-1/ptc3-7', res1, res2)

def bay1():
    a = [8, 10, 12, 16, 18]
    c = [2, 3]
    n = random.choice(a)
    k = random.choice(a)
    l1 = random.choice(c)
    l2 = random.choice(c)
    res2 = [n, k, l1, l2]

    f = n / (n + k) * ((n + k) / (l2 + n + k) + l1 * l2 / (l1 + n + k) / (l2 + n + k))
    f += n * l2 / (l1 + n + k) / (l2 + n + k)
    g = 0
    for i in range(min(l1+1, l2)):
        d = binomial(n, i) * binomial(k, l1-i)
        h = d / binomial(n+k, l1)
        d1 = binomial(n+i, i+1) * binomial(k+l1-i, l2-i-1)
        d2 = binomial(n+k+l1, l2)
        h *= d1 / d2
        g += h
    g *= (n+1) / (n+k-l1+l2)

    _f = strfix(f)
    _g = strfix(g)
    res1 = [_f, _g, _f, strfix(g / f)]
    create_task('PT1-2/bay1', res1, res2)

def bay2():
    a = [8, 10, 12, 16, 18]
    c = [2, 3]
    n = random.choice(a)
    k = random.choice(a)
    n1 = random.choice(a)
    k1 = random.choice(a)
    l1 = random.choice(c)
    l2 = random.choice(c)
    res2 = [n, k, n1, k1, l1, l2]

    f = n*l1 / (n+k) / (n1+k1+l1-l2)
    f += n1 * (k1+n1-l2) / (n1+k1) / (n1+k1+l1-l2)
    g = 0
    for i in range(min(l1, l2)+1):
        d = binomial(n, i) * binomial(k, l1-i)
        h = d / binomial(n+k, l1)
        d1 = binomial(n1, i) * binomial(k1, l1-i)
        d2 = binomial(n1+k1, l2)
        h *= d1 / d2
        g += h
    g *= k1 / (n1+k1+l1-l2)

    _f = strfix(f)
    _g = strfix(g)
    res1 = [_f, _g, _f, strfix(g / f)]
    create_task('PT1-2/bay2', res1, res2)

def bay3():
    a = [8, 10, 12, 16, 18]
    n = random.choice(a)
    k = random.choice(a)
    n1 = random.choice(a)
    k1 = random.choice(a)
    res2 = [n, k, n1, k1]

    s = (n+k) * (n1+k1)
    p1 = n * k1 / s / 2
    p2 = k * n1 / s / 2
    p3 = 1 - p1 - p2
    p1 *= ((n - 1) * (n1 + 1) + (k + 1) * (n1 - 1)) / s
    g = p2 * (n + 1) * (n1 - 1) / s
    p2 = g + p2 * (k - 1) * (k1 + 1) / s
    g += p3 * k * k1 / s
    p3 *= (n * n1 + k * k1) / s
    f = p1 + p2 + p3

    _f = strfix(f)
    _g = strfix(g)
    res1 = [_f, _g, _f, strfix(g / f)]
    create_task('PT1-2/bay3', res1, res2)

def bay4():
    a = [8, 10, 12, 16, 18]
    c = [2, 3]
    n = random.choices(a, k=3)
    k = random.choices(a, k=3)
    l1 = random.choice(c)
    l2 = random.choice(c)
    res2 = [n[0], k[0], n[1], k[1], n[2], k[2], l1, l2]

    s = n[2] + l2 + k[2]
    p1 = (k[2] + n[2]) / s
    p2 = (n[1] + k[1]) * l2 / (l1 + n[1] + k[1]) / s
    p3 = 1 - p1 - p2
    p1 = p1 * n[2] / (n[2] + k[2])
    p2 = p2 * n[1] / (n[1] + k[1])
    p3 = p3 * n[0] / (n[0] + k[0])
    f = p1 + p2 + p3

    for i in range(l1+1):
        s = binomial(n[0], i) * binomial(k[0], l1-i)
        g = s / binomial(n[0]+k[0], l1)
        s = binomial(k[1]+l1-i, l2-1)*(n[2]+i)
        p = s / binomial(n[1]+k[1]+l1, l2)
        g *= p*(n[2]+1)/(n[2]+k[2]+l2)

    _f = strfix(f)
    _g = strfix(g)
    res1 = [_f, _g, _f, strfix(g / f)]
    create_task('PT1-2/bay4', res1, res2)

def bay5():
    a = [8, 10, 12, 16, 18]
    c = [2, 3]
    n = random.choices(a, k=3)
    k = random.choices(a, k=3)
    l = random.choice(c)
    res2 = [n[0], k[0], n[1], k[1], n[2], k[2], l]

    i = n[0] + k[0]
    f = binomial(n[0], l) / binomial(i, l)
    g = f / 3
    i = n[1] + k[1]
    f += binomial(n[1], l) / binomial(i, l)
    i = n[2] + k[2]
    f = (f + binomial(n[1], l) / binomial(i, l)) / 3

    _f = strfix(f)
    _g = strfix(g)
    res1 = [_f, _g, _f, strfix(g / f)]
    create_task('PT1-2/bay5', res1, res2)

def bay6():
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [10, 15, 20, 30, 50]
    c = [4, 5, 6]
    p = random.choices(a, k=5)
    p.sort()
    _, q = discrt(10, 5)
    n = random.choice(b)
    k = random.choice(c)
    res2 = [*(f'0.0{x}' for x in p), *(f'{x}0' for x in p), n, k]

    n1 = k*n // 100
    h = 0
    for j1 in range(5):
        g = 0
        p4 = p[j1] / 100
        p1 = 1 - p4
        for j in range(n1+1):
            k3 = binomial(n, j)
            m = n - j
            r = k3 * p4**j * p1**m
            g += r
        h += g * q[j1]
    h /= 10
    res1 = [n1, n, n, strfix(h)]
    create_task('PT1-2/bay6', res1, res2)

def bay7():
    a = [1000, 1500, 2000, 2500, 3000]
    b = [50, 100, 150, 200, 300]
    c = [2, 3, 4]
    n = random.choice(a)
    _, p = discrt(10, 5)
    k = random.choice(b)
    n1 = random.choice(c)
    res2 = [n, *p, k, n1]

    p1 = 0
    k1 = k - 1
    for j in range(1, n1+1):
        k2 = n - j
        k3 = binomial(k2, k1)
        p1 += j * k3 * p[j-1]
    p2 = p1
    n1 += 1
    for j in range(n1, 6):
        k2 = n - j
        k3 = binomial(k2, k1)
        p1 += j * k3 * p[j-1]
    p3 = p2 / p1
    res1 = [n1, n, k1, n, k1, strfix(p3)]
    create_task('PT1-2/bay7', res1, res2)

def bay8():
    a = [1000, 1500, 2000, 2500, 3000]
    b = [1, 2, 3, 4, 5]
    c = [2, 3, 4]
    d = [50, 100, 150, 200, 250, 300]
    n = random.choice(a)
    k = random.choice(b)
    n1 = random.choice(c)
    k1 = random.choice(d)
    res2 = [n, k, n1, k1]

    if n1 <= 1:
        r = '1'
    else:
        p1 = n * k / 1000
        p3 = 0
        p2 = 1
        p = [0] * (n1+1)
        for j in range(n1+1):
            p2 *= p1 / j if j > 0 else 1
            p[j] = p2 * math.exp(-p1)
            p3 += p[j]
        p = [x / p3 for x in p]
        p1 = 1
        p2 = n
        p3 = n - k1
        p4 = n
        g = p[0] + p[1]
        for j in range(2, n1+1):
            p4 -= 1
            p1 *= p3
            p2 *= p4
            p3 -= 1
            p[j] *= p1 * j * k1 / p2
            g += p[j]
        r = strfix(g)

    res1 = [n1, n, k1, n, k1-1, n, f'0.00{k}', f'0.99{k}', n, n, k1, n1, n, f'0.00{k}', f'0.99{10-k}', n, r]
    create_task('PT1-2/bay8', res1, res2)

def bay9():
    a = [3, 4 ,5, 6, 7, 8, 9, 10]
    c = [2, 3, 4]
    n = random.choice(a)
    k = random.choice(a)
    ending = 'а' if k <= 4 else 'ов'
    n1 = random.choice(c)
    res2 = [n, k, ending, n1]

    g = 0
    for j in range(n1+1):
        k3 = binomial(n, j) * binomial(k, n1 - j)
        k2 = 5**j
        k4 = 6**j
        k2 = k4 - k2
        k3 = k3 * k2
        m = k + n
        k1 = binomial(m, n1) * k4
        g += k3 / k1

    res1 = [n1, n, k, n1, n+k, n1, strfix(g)]
    create_task('PT1-2/bay9', res1, res2)

def bay10():
    a = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    b = ['«хорошо»', '«отлично»', '«удовлетворительно»', '«хорошо» или «отлично»', '«хорошо» или «удовлетворительно»', '«не удовлетворительно»']
    p = random.choices(a, k=3)
    k1 = random.randint(0, len(b)-1)
    s = b[k1]
    res2 = [*map(formatplus, p), s]

    k = sum(p) * 3
    match k1:
        case 1:
            k2 = p[1] + p[2]
        case 2:
            k2 = 3 * p[0] + p[1]
        case 3:
            k2 = p[1] + p[2]
        case 4:
            k2 = 3 * p[0] + 2 * p[1] + p[2]
        case 5:
            k2 = 2 * (p[1] + p[2])
        case _:
            k2 = p[2]
    k2, k, g = frac(k2, k)
    k3, k4, h = frac(3*p[1], 3*p[1]+p[2])
    res1 = [k2, k, strfix(g), k3, k4, strfix(h)]
    create_task('PT1-2/bay10', res1, res2)

def bay11():
    a = [32, 36, 52]
    b = [2, 3, 4]
    c = [5, 6, 7, 8, 9]
    d = [2, 3, 4, 5]
    l = random.randint(0, len(a)-1)
    n = a[l]
    k = random.choice(b)
    m = random.choice(c)
    n1 = random.choice(d)
    j = l+1
    end = 'карт' if j == 2 else 'карты'
    end1 = 'тузов' if n1 > 4 else 'туза'
    l = random.choice(b)
    res2 = [n, end, k, m, n1, end1, l]

    k4 = n + 2 * k
    k1 = k4 - 4
    k2 = m - n1
    k3 = n - 4
    k5 = 4
    h = 0
    q = 0
    for i in range(k+1):
        p = 0
        for j in range(i+1):
            k6 = binomial(k5, j) * binomial(k3, k - j)
            k7 = binomial(k5, i - j)
            k6 *= k7
            k7 = binomial(k3, k - i + j)
            k6 *= k7
            p += k6
        p /= binomial(n, k) ** 2
        k6 = binomial(4 + i, n1)
        k7 = binomial(k1 - i, k2)
        k6 *= k7
        k7 = binomial(k4, m)
        g = k6 / k7
        p *= g
        h += p
        if i == l:
            q = p

    p = 0 if h == 0 else q/h
    res1 = [l, strfix(p), n1, k1, k2, k3, k, k3, k, k4, m, n, k]
    create_task('PT1-2/bay11', res1, res2)

def nez1():
    a = [300, 400, 500, 600, 800]
    c = [1, 2, 3]
    n = random.choice(a)
    k = random.choice(a)
    l = random.choice(c)
    res2 = [n, k, l]

    n = 2*(n+k) // 100
    g = l / 10
    k = 3
    g = 1 - g**k
    f = 1 - g**n

    res1 = [l, n, strfix(f)]
    create_task('PT1-3/nez1', res1, res2)

def nez2():
    c = [1, 2, 3, 5]
    n = random.choice(c)
    k = random.choice(c)
    res2 = [n, k]

    g1 = n / 10
    g2 = k / 10
    f0 = (1 - g1) * (1 - g2)
    f1 = 1 - f0
    f = f1 ** 2
    f = g2 * (1 - f)
    n = 10 - k
    f1 = (1 - g2) * (1 - g1 * g1) * (1 - g2 * g2)
    f += f1

    res1 = [k, strfix(f0, p2=2), n, strfix(g1, 1, 1), strfix(g2, 1, 1), strfix(f)]
    create_task('PT1-3/nez2', res1, res2)

def nez3():
    c = [5, 6, 7, 8, 9]
    n = random.choice(c)
    k = random.choice(c)
    res2 = [n, k]

    g1 = n / 10
    g2 = k / 10
    f0 = (g1*g2) ** 3
    g3 = 1 - g1
    g4 = 1 - g2
    f1 = (g1*g2) ** 2 *g3*g4
    f2 = (g3*g4) ** 2 *g1*g2
    f3 = (g3*g4) ** 3
    f = f0 + 3*f1 + 3*f2 + f3

    res1 = [strfix(f0), strfix(f1), strfix(f2), strfix(f3), strfix(f)]
    create_task('PT1-3/nez3', res1, res2)

def nez4():
    a = [5, 6, 9]
    b = [3, 4, 5, 6]
    c = [1, 2, 3, 4, 5]
    n = random.choice(b)
    k = random.choice(a)
    l = random.choice(c)
    res2 = [n, k, l]

    g = l / 10
    f = (1 - g**n)**k

    res1 = [l, n, k, strfix(f)]
    create_task('PT1-3/nez4', res1, res2)

def nez5():
    a = [15,18,20,25,30]
    b = [8,9,10,11,12]
    c = [1,2,3,4,5,6,7,8,9]
    n = random.choice(a)
    np = random.choice(c)
    k = random.choice(b)
    m = random.choice(b)
    res2 = [n, np, k, n-m, n]

    n1 = k + m + 1
    f = np / 100
    g = 1 - f
    if n1 <= n:
        g2 = g * g
        g3 = g2 * g
        r1 = f'(1-{strfix(f, p2=2)}^2)^'
        r2 = f'{{{n1}}} '
        n2 = n - n1
        if n2 == 0:
            r3 = f'(1-{strfix(f, p2=2)}^3)^'
        elif n2 == 1:
            r3 = '{}; '
        else:
            r3 = f'{{{n2}}}; '
    else:
        g2 = g * g
        g3 = g
        n2 = k+m-n
        if n2 == 0:
            r1 = f'(1-{strfix(f, p2=2)})^'
        elif n2 == 1:
            r1 = '{} '
        else:
            r1 = f'{{{n2}}} '
        n1 = n - n2
        r2 = f'(1-{strfix(f, p2=2)}^2)^'
        r3 = '{}; ' if n1==1 else f'{{{n1}}}; '

    g = g2**n1 * g3**n2

    res1 = [r1, r2, r3, strfix(g)]
    create_task('PT1-3/nez5', res1, res2)

if __name__ == '__main__':
    nez5()
