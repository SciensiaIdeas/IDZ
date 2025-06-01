import math
import random
from fractions import Fraction

from proc import *
from collections import Counter
from scipy.stats import binom, hypergeom


def ptc31():
    a = [30, 50, 60, 70, 80, 90]
    b = [3, 4, 5, 6, 7, 8, 10, 12]
    n = random.choice(a)
    i = random.choice(b)
    l = n // 2

    total = binomial(n, i)
    favorable = binomial(l, i)
    p, q, r = frac(favorable, total)
    res1 = [l, i, n, p, q, strfix(r)]
    res2 = [n, i]
    create_task('PT1-1/ptc3-1', res1, res2)

def ptc32():
    b = [3, 4, 5, 6, 7, 8, 10, 12]
    n = random.choice(b)
    res2 = [n]
    l = 54 // n
    n1 = l * (l-1)
    n2 = 54 * 53

    p, q, r = frac(n1, n2)
    res1 = [l, p, q, strfix(r)]
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
    res1 = [k, k1, n, j, p, q, strfix(r)]
    create_task('PT1-1/ptc3-3', res1, res2)

def ptc34():
    d = ['параллелепипед','тетраэдр','математика','алгебра','параллелограмм','плоскость']
    w = random.choice(d)
    res2 = [w]

    l = len(w)
    freqs = Counter(w).values()
    n = math.prod(math.factorial(x) for x in freqs)

    r = math.factorial(l) // n
    res1 = [n, l, r]
    create_task('PT1-1/ptc3-4', res1, res2)

def ptc35():
    b = [5, 7, 8, 9, 10]
    n = random.choice(b)
    res2 = [n]

    n1 = 15 * (2**n-2)
    n2 = 6**n
    p, q, r = frac(n1, n2)
    res1 = [n, p, q, strfix(r)]
    create_task('PT1-1/ptc3-5', res1, res2)

def ptc36():
    b = [5, 6, 7, 8, 9]
    c = [2, 3, 4]
    n = random.choice(b)
    k = random.choice(c)
    res2 = [n, k]

    n1 = binomial(10, k) * visk(n, k)
    n2 = 10 ** n
    p, q, r = frac(n1, n2)
    res1 = [k, n, p, q, strfix(r)]
    create_task('PT1-1/ptc3-6', res1, res2)

def ptc37():
    b = [5, 6, 7, 8, 9]
    c = [3, 4]
    n = random.choice(b)
    k = random.choice(b)
    l = random.choice(c)
    res2 = [n, k, l]

    fr = 1 / l
    p1 = 1 - (1-fr)**n
    p2 = 1 - (1-fr)**k
    frr = (p1*p2)**l
    res1 = [n, k, l, strfix(frr)]
    create_task('PT1-1/ptc3-7', res1, res2)


def bay1():
    a = [8, 10, 12, 16, 18]
    c = [2, 3]
    n = random.choice(a)
    l = random.choice(a)
    x = random.choice(c)
    k = random.choice(c)
    res2 = [n, l, x, k]

    p = 1 / l
    r = sum(binom.pmf(b, n, p) / binom.cdf(x, n, p) * hypergeom.cdf(1, n, b, k) for b in range(x+1))
    res1 = [*res2, strfix(r)]
    create_task('PT1-2/bay1', res1, res2)

def bay2():
    a = [8, 10, 12, 16, 18]
    c = [2, 3]
    n1 = random.choice(a)
    n2 = random.choice(a)
    m1 = random.choice(a)
    m2 = random.choice(a)
    k1 = random.choice(c)
    k2 = random.choice(c)
    res2 = [n1, n2, m1, m2, k1, k2]

    N1 = n1 + n2
    N2 = m1 + m2
    M = N2 + k1
    l1 = max(0, k1 - n2)
    l2 = min(k1, n1)
    numerator = 0.0
    denominator = 0.0
    for w in range(l1, l2+1):
        p_w = hypergeom.pmf(w, N1, n1, k1)  # P(W = w)
        p_x_eq_w = hypergeom.pmf(w, M, m1 + w, k2)  # P(X = w | W = w)
        p_white_draw = (m1 + w) / M  # P(белый из 2-го ящика при W = w)

        numerator += p_w * p_x_eq_w * p_white_draw
        denominator += p_w * p_white_draw

    r = numerator / denominator
    res1 = [strfix(numerator), strfix(denominator), strfix(r)]
    create_task('PT1-2/bay2', res1, res2)

def bay3():
    a = [8, 10, 12, 16, 18]
    n1 = random.choice(a)
    n2 = random.choice(a)
    m1 = random.choice(a)
    m2 = random.choice(a)
    res2 = [n1, n2, m1, m2]

    p_w1 = [n2 / (n1 + n2), n1 / (n1 + n2)]  # P(W1=0), P(W1=1)
    p_w2 = [m2 / (m1 + m2), m1 / (m1 + m2)]  # P(W2=0), P(W2=1)

    numerator = 0.0
    denominator = 0.0
    for w1 in [0, 1]:
        for w2 in [0, 1]:
            pw = p_w1[w1] * p_w2[w2]
            white1 = n1 - w1 + w2
            white2 = m1 - w2 + w1
            total1 = n1 + n2
            total2 = m1 + m2
            p_white1 = white1 / total1
            p_white2 = white2 / total2
            psame = p_white1 * p_white2 + (1 - p_white1) * (1 - p_white2)
            if w1 == w2:
                numerator += pw * psame
            denominator += pw * psame


    r = numerator / denominator
    res1 = [strfix(numerator), strfix(denominator), strfix(r)]
    create_task('PT1-2/bay3', res1, res2)

def bay4():
    a = [8, 10, 12, 16, 18]
    c = [2, 3]
    n = random.choices(a, k=3)
    m = random.choices(a, k=3)
    k1 = random.choice(c)
    k2 = random.choice(c)
    res2 = [n[0], m[0], n[1], m[1], n[2], m[2], k1, k2]

    numerator = 0.0
    denominator = 0.0
    total1 = n[0] + m[0]
    total3 = n[2] + m[2]

    for b1 in range(max(0, k1 - m[0]), min(k1, n[0]) + 1):
        p_b1 = hypergeom.pmf(b1, total1, n[0], k1)
        white2 = n[1] + b1
        black2 = m[1] + k1 - b1
        total2_new = white2 + black2

        p_b2_eq_0 = hypergeom.pmf(0, total2_new, white2, k2)
        p_white_draw_if_b2_0 = n[2] / (total3 + k2)
        numerator += p_b1 * p_b2_eq_0 * p_white_draw_if_b2_0

        for b2 in range(max(0, k2 - black2), min(k2, white2) + 1):
            p_b2 = hypergeom.pmf(b2, total2_new, white2, k2)
            p_white_draw = (n[2] + b2) / (total3 + k2)
            denominator += p_b1 * p_b2 * p_white_draw

    r = numerator / denominator
    res1 = [*res2, strfix(numerator), strfix(denominator), strfix(r)]
    create_task('PT1-2/bay4', res1, res2)

def bay5():
    a = [8, 10, 12, 16, 18]
    c = [2, 3]
    n = random.choices(a, k=3)
    m = random.choices(a, k=3)
    k = random.choice(c)
    res2 = [n[0], m[0], n[1], m[1], n[2], m[2], k]
    urns = zip(n, m)

    likelihoods = []
    for ni, mi in urns:
        if k > ni:
            likelihoods.append(0.0)
        else:
            N = ni + mi
            likelihood = hypergeom.pmf(k, N, ni, k)  # all k must be white
            likelihoods.append(likelihood)

    numerator = likelihoods[0] / 3
    denominator = sum(likelihoods) / 3

    r = numerator / denominator
    res1 = [strfix(numerator), strfix(denominator), strfix(r)]
    create_task('PT1-2/bay5', res1, res2)

def bay6():
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [10, 15, 20, 30, 50]
    c = [4, 5, 6]
    p = random.choices(a, k=5)
    p.sort()
    q = discrt(10, 5)
    n = random.choice(b)
    k = random.choice(c)
    q_display = dict(zip(q, ('G', 'F', 'R', 'C', 'B')))
    q_display = dict(sorted(q_display.items(), reverse=True)).items()
    first = True
    l1 = 0; v1 = 0; s = []
    for val, lit in q_display:
        if first:
            l1 = lit; v1 = val * 10
            first = False
        else:
            if val > 0:
                s.append(f'{lit} - {10*val}\\%')
            else:
                s1 = " , остальные не поставляют"
                break
    else:
        s1 = ""
    literal = ", ".join(s)
    res2 = [*p, l1, v1, literal, s1, n, k]

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
    res1 = [n1, n, strfix(h)]
    create_task('PT1-2/bay6', res1, res2)

def bay7():
    a = [1000, 1500, 2000, 2500, 3000]
    b = [50, 100, 150, 200, 300]
    c = [2, 3, 4]
    n = random.choice(a)
    p = discrt(10, 5)
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
    res1 = [n1, n, k1, strfix(p3)]
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

    res1 = [n1, n, k1, k1-1, f'0.00{k}', f'0.99{k}', f'0.99{10-k}', r]
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

    res1 = [n1, n, k, n+k, strfix(g)]
    create_task('PT1-2/bay9', res1, res2)

def bay10():
    a = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    b = [('«хорошо»', [1,2]), ('«отлично»', [0,1]), ('«удовлетворительно»', [1,2]), ('«хорошо» или «отлично»', [0,1,2]),
         ('«хорошо» или «удовлетворительно»', [1,2]), ('«не удовлетворительно»', [2])]
    c = ['отличник', 'хорошо успевающий', 'слабо занимающийся']
    p = random.choices(a, k=3)
    k1 = random.randint(0, len(b)-1)
    s = b[k1][0]
    st = random.choice(b[k1][1])
    res2 = [*map(formatplus, p), s, c[st]]

    k = sum(p)
    match k1:
        case 1:
            s = f'\\frac{{{p[0]}}}{{{k}}}+\\frac{{{p[1]}}}{{{k}}} \\cdot \\frac{{1}}{{3}}'
            sr = Fraction(p[0], k) + Fraction(p[1], k) / 3
            if st == 0:
                s1 = f'\\frac{{{p[0]}}}{{{k}}}'
                sr1 = Fraction(p[0], k) / sr
            else:
                s1 = f'\\frac{{{p[1]}}}{{{k}}} \\cdot \\frac{{1}}{{3}}'
                sr1 = Fraction(p[1], k) / 3 / sr
        case 3:
            s = f'\\frac{{{p[0]}}}{{{k}}}+\\frac{{{p[1]}}}{{{k}}} \\cdot \\frac{{2}}{{3}}+\\frac{{{p[2]}}}{{{k}}} \\cdot \\frac{{1}}{{3}}'
            sr = Fraction(p[0], k) + (Fraction(p[1], k)*2 + Fraction(p[2], k)) / 3
            if st == 0:
                s1 = f'\\frac{{{p[0]}}}{{{k}}}'
                sr1 = Fraction(p[0], k) / sr
            elif st == 1:
                s1 = f'\\frac{{{p[1]}}}{{{k}}} \\cdot \\frac{{2}}{{3}}'
                sr1 = Fraction(p[0], k)*Fraction(2,3) / sr
            else:
                s1 = f'\\frac{{{p[2]}}}{{{k}}} \\cdot \\frac{{1}}{{3}}'
                sr1 = Fraction(p[2], k) / 3 / sr
        case 4:
            s = f'\\frac{{{p[1]}}}{{{k}}} \\cdot \\frac{{2}}{{3}}+\\frac{{{p[2]}}}{{{k}}} \\cdot \\frac{{2}}{{3}}'
            sr = (Fraction(p[1], k) + Fraction(p[2], k))* 2/3
            if st == 1:
                s1 = f'\\frac{{{p[1]}}}{{{k}}} \\cdot \\frac{{2}}{{3}}'
                sr1 = Fraction(p[1], k)*Fraction(2,3) / sr
            else:
                s1 = f'\\frac{{{p[2]}}}{{{k}}} \\cdot \\frac{{2}}{{3}}'
                sr1 = Fraction(p[2], k)*Fraction(2,3) / sr
        case 5:
            s = f'\\frac{{{p[2]}}}{{{k}}} \\cdot \\frac{{1}}{{3}}'
            sr = Fraction(p[2], k*3)
            s1 = f'\\frac{{{sr.numerator}}}{{{sr.denominator}}}'
            sr1 = Fraction(1, 1)
        case _:
            s = f'\\frac{{{p[1]}}}{{{k}}} \\cdot \\frac{{1}}{{3}}+\\frac{{{p[2]}}}{{{k}}} \\cdot \\frac{{1}}{{3}}'
            sr = (Fraction(p[1], k) + Fraction(p[2], k)) / 3
            if st == 1:
                s1 = f'\\frac{{{p[1]}}}{{{k}}} \\cdot \\frac{{1}}{{3}}'
                sr1 = Fraction(p[1], k) / sr
            else:
                s1 = f'\\frac{{{p[2]}}}{{{k}}} \\cdot \\frac{{1}}{{3}}'
                sr1 = Fraction(p[2], k) / 3 / sr

    s1 += f' : \\frac{{{sr.numerator}}}{{{sr.denominator}}}'
    res1 = [s, sr.numerator, sr.denominator, strfix(float(sr)), s1, sr1.numerator, sr1.denominator, strfix(float(sr1))]
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
    res1 = [l, strfix(p), n1, k1, k2, k3, k, k4, m, n]
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
    g = 1 - g**3
    f = 1 - g**n

    res1 = [*res2, strfix(f)]
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
    b = [2, 3, 4]
    p1 = random.choice(c)
    p2 = random.choice(c)
    n = random.choice(b)
    res2 = [n, p1, p2]
    p1 /= 10; p2 /= 10

    r = sum(binom.pmf(i, n, p1) * binom.pmf(i, n, p2) for i in range(n+1))
    res1 = [n, strfix(r)]
    create_task('PT1-3/nez3', res1, res2)

def nez4():
    a = [5, 6, 9]
    b = [3, 4, 5, 6]
    c = [5, 6, 8, 10, 12]
    x = random.choice(a)
    n = random.choice(b)
    l = random.choice(c)
    res2 = [x, n, l]

    p = l / 100
    N = n * x
    r = 1 - sum(binom.pmf(i, N, p) for i in range(n))

    res1 = [n-1, strfix(r), l, N]
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
    nez4()
