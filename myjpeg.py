# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:26:26 2022

@author: mijap
"""
import math


def ppm_tokenize(stream):
    reading = stream.readlines()
    for line in reading:
        line.strip()
        line1 = line.split('#')
        line2 = line1[0].split()
        yield from line2


def ppm_load(stream):
    reading = [i for i in ppm_tokenize(stream)]
    w, h = int(reading[1]), int(reading[2])
    x = len(reading) - 3
    img = [reading[4+j:4+j+3] for j in range(0, x-1, 3)]
    img1 = img[:3]
    img2 = img[3:]
    img1 = [tuple(j) for j in img1]
    img2 = [tuple(j) for j in img2]
    # this is only to match the return syntax, it returns tuples
    return (w, h, [[(int(x), int(y), int(z)) for (x, y, z) in img1], [(int(a), int(b), int(c)) for (a, b, c) in img2]])


def ppm_save(w, h, img, output):
    with open(output, 'w') as o:
        o.write('P3\n')
        o.write(f'{w} {h}\n')
        o.write('255\n')
        for i in range(len(img)):
            for j in range(len(img[0])):
                a = [str(p) for p in img[i][j]]
                o.write(' '.join(a) + '\n')


def RGB2YCbCr(r, g, b):

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5*b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    prep = [round(y), round(cb), round(cr)]
    for i in range(len(prep)):
        if prep[i] < 0:
            prep[i] = 0
        if prep[i] > 255:
            prep[i] = 255
    return tuple(prep)


def YCbCr2RGB(Y, Cb, Cr):

    R = Y + 1.402*(Cr-128)
    G = Y - 0.344136*(Cb-128) - 0.714136*(Cr-128)
    B = Y + 1.772 * (Cb-128)
    prep = [round(R), round(G), round(B)]
    for i in range(len(prep)):
        if prep[i] < 0:
            prep[i] = 0
        elif prep[i] > 255:
            prep[i] = 255
    return tuple(prep)


# print(RGB2YCbCr(200, 20, 200))
# sandros: 80, 188, 203
# mine: (94, 188, 203)


def img_RGB2YCbCr(img):
    n = len(img)  # rows
    m = len(img[0])  # columns
    Y = [[0 for _ in range(n)] for _ in range(m)]
    Cb = [[0 for _ in range(n)] for _ in range(m)]
    Cr = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            r, g, b = img[i][j]
            y, cb, cr = RGB2YCbCr(r, g, b)
            Y[i][j] = y
            Cb[i][j] = cb
            Cr[i][j] = cr
    return (Y, Cb, Cr)


def img_YCbCr2RGB(Y, Cb, Cr):
    img = []
    for i in range(len(Y)):
        for j in range(len(Y[0])):
            img.append((YCbCr2RGB(Y[i][j], Cb[i][j], Cr[i][j])))
    img1 = img[:3]
    img2 = img[3:]
    return [img1, img2]


def subsampling(w, h, C, a, b):
    # a width
    # b height
    height = h//b + (h % b != 0)
    width = w//a + (w % a != 0)
    new = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(0, h, b):
        for j in range(0, w, a):
            rows = min(h, i+b)
            columns = min(w, j+a)
            numbers = [C[n][m]
                       for n in range(j, columns) for m in range(i, rows)]
            print(numbers)
            average = round(sum(numbers) / len(numbers))
            new[i][j] = average
    return new


def extrapolate(w, h, C, a, b):
    new = [[0 for _ in range(w)] for _ in range(h)]
    for i in range(h):
        for j in range(w):
            new[i][j] = C[i // b][j//a]
    return new

# f = open('file.ppm')
# img = ppm_load(f)[2]
# a = 3
# b = 2
# sub = subsampling(3, 2, img[0], a, b)
# print(sub)
# ext = extrapolate(3, 2, sub, a, b)
# print(ext)


def block_splitting(w, h, C):
    M = [[C[i][j] for j in range(w)] for i in range(h)]
    # then I extend the width and height of M
    wr, hr = w % 8, h % 8
    if wr != 0:
        for row in M:
            for _ in range(8-wr):
                row.append(row[-1])
    if hr != 0:
        for _ in range(8-hr):
            M.append(M[-1])
    # then I do the splitting: we run horizontally then vertically,
    # from left to right and then from top to bottom
    w, h = len(M[0]), len(M)
    wd, hd = w//8, h//8
    for m in range(hd):  # m is the index of the horizontal cut we're at
        for n in range(wd):  # m is the index of the vertical cut we're at
            matrix = [[0 for _ in range(8)] for _ in range(8)]
            for i in range(8):
                for j in range(8):
                    matrix[i][j] = M[8*m+i][8*n+j]
            yield matrix


C = [
    [1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
    [2,  3,  4,  5,  6,  7,  8,  9, 10,  1],
    [3,  4,  5,  6,  7,  8,  9, 10,  1,  2],
    [4,  5,  6,  7,  8,  9, 10,  1,  2,  3],
    [5,  6,  7,  8,  9, 10,  1,  2,  3,  4],
    [6,  7,  8,  9, 10,  1,  2,  3,  4,  5],
    [7,  8,  9, 10,  1,  2,  3,  4,  5,  6],
    [8,  9, 10,  1,  2,  3,  4,  5,  6,  7],
    [9, 10,  1,  2,  3,  4,  5,  6,  7,  8],
]


#print(block_splitting(10, 9, C))


def DCT(v):
    n = len(v)
    answer = [0 for _ in range(n)]
    for i in range(n):
        delta = 1
        if i == 0:
            delta = (1 / math.sqrt(2))
        big = sum([v[j] * math.cos((math.pi / n)*(j + 0.5)*i)
                  for j in range(n)])
        answer[i] = delta * math.sqrt(2/n) * big
    return [round(i, 2) for i in answer]


# print(DCT([8, 16, 24, 32, 40, 48, 56, 64]))


def C(n):
    c = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            delta = 1
            if i == 0:
                delta = (1 / math.sqrt(2))
            c[i][j] = delta * math.sqrt(2/n) * \
                math.cos((math.pi / n)*(j + 0.5)*i)
    return c


def matrixMult(A, B):
    a = len(A)
    b = len(A[0])
    c = len(B)
    d = len(B[0])
    out = [[0 for _ in range(d)] for _ in range(a)]

    if b != c:
        print(f'omg what are u doing')
        return

    for i in range(a):
        for j in range(d):
            out[i][j] = sum([A[i][n] * B[n][j] for n in range(b)])

    return out


def trans(A):
    a = len(A)
    b = len(A[0])
    out = [[0 for _ in range(a)] for _ in range(b)]
    for i in range(b):
        for j in range(a):
            out[i][j] = A[j][i]

    return out


def DCT(v):
    n = len(v)
    t = matrixMult([v], trans(C(n)))

    return [round(i, 3) for i in t[0]]


def IDCT(v):
    n = len(v)
    t = matrixMult([v], C(n))
    return [round(i, 3) for i in t[0]]


# v = [
#     float(random.randrange(-10**5, 10**5))
#     for _ in range(random.randrange(1, 128))
# ]
# v2 = IDCT(DCT(v))


# assert (all(math.isclose(v[i], v2[i], abs_tol=1) for i in range(len(v))))

def DCT2(m, n, A):
    part = matrixMult(C(m), A)
    answer = matrixMult(part, trans(C(n)))
    return [[round(i, 3) for i in j] for j in answer]


# m = [
#     [140,  144,  147,  140,  140,  155,  179,  175],
#     [144,  152,  140,  147,  140,  148,  167,  179],
#     [152,  155,  136,  167,  163,  162,  152,  172],
#     [168,  145,  156,  160,  152,  155,  136,  160],
#     [162,  148,  156,  148,  140,  136,  147,  162],
#     [147,  167,  140,  155,  155,  140,  136,  162],
#     [136,  156,  123,  167,  162,  144,  140,  147],
#     [148,  155,  136,  155,  152,  147,  147,  136],
# ]
# print(DCT2(8, 8, m))


def IDCT2(m, n, A):
    part = matrixMult(trans(C(m)), A)
    answer = matrixMult(part, C(n))
    return [[round(i, 3) for i in j] for j in answer]


# m = random.randrange(1, 128)
# n = random.randrange(1, 128)
# A = [
#     [
#         float(random.randrange(-10**5, 10**5))
#         for _ in range(n)
#     ]
#     for _ in range(m)
# ]
# A2 = IDCT2(m, n, DCT2(m, n, A))
# assert (all(
#     math.isclose(A[i][j], A2[i][j], abs_tol=1)
#     for i in range(m) for j in range(n)
# ))
def redalpha(i):
    if i > 32:
        return redalpha(i-32)
    else:
        if i <= 8:
            s = 1
            k = i
        elif i <= 24 and i > 8:
            s = -1
            k = abs(i-16)
        else:
            s = 1
            k = abs(i-32)
    return (s, k)


# print(redalpha(60))


def ncoeff8(i, j):
    if i == 0:
        return redalpha(4)
    out = redalpha(i*(2*j+1))
    return out


M8 = [
    [ncoeff8(i, j) for j in range(8)]
    for i in range(8)
]


def M8_to_str(M8):
    def for1(s, i):
        return f"{'+' if s >= 0 else '-'}{i:d}"

    return "\n".join(
        " ".join(for1(s, i) for (s, i) in row)
        for row in M8
    )


print(M8_to_str(M8))

C1 = [[0 for _ in range(8)] for _ in range(8)]
for i in range(8):
    for j in range(8):
        s, k = ncoeff8(i, j)
        C1[i][j] = (s * math.cos(k*math.pi/16))/2
transpose = trans(C1)


def chenvec(v):
    result = [0 for _ in range(len(v))]
    result[0] = transpose[0][0] * sum(v)
    result[2] = transpose[0][2] * \
        (v[0] + v[7] - v[3] - v[4]) + \
        transpose[1][2] * (v[1] + v[6] - v[2] - v[5])
    result[4] = transpose[0][4] * \
        (v[0] + v[3] + v[4] + v[7] - v[1] - v[2] - v[5] - v[6])
    result[6] = transpose[0][6] * \
        (v[0] + v[7] - v[3] - v[4]) + \
        transpose[1][6] * (v[1] + v[6] - v[2] - v[5])
    for i in range(1, len(v), 2):
        for j in range(4):
            result[i] += transpose[j][i] * (v[j] - v[7-j])
    return result


def DCT_Chen(A):
    a = [0 for _ in range(len(A))]
    b = [0 for _ in range(len(A[0]))]
    for i in range(len(A)):
        a[i] = chenvec(A[i])
    a2 = trans(a)
    for j in range(len(A[0])):
        b[j] = chenvec(a2[j])
    return trans(b)


# ups = [[i+j for i in range(8)] for j in range(8)]
# print(DCT_Chen(ups))


Theta = [[0 for _ in range(4)] for _ in range(4)]
for i in range(4):
    for j in range(4):
        Theta[i][j] = C1[2*i+1][j]  # odd rows of C

Omega = [[0 for _ in range(4)] for _ in range(4)]
for i in range(4):
    for j in range(4):
        Omega[i][j] = C1[2*i][j]


def ichenvec(v):
    res0 = 0
    res1 = 0
    res2 = 0
    res3 = 0
    v0a41 = (v[0] + v[4]) * C1[0][0]
    v0a42 = (v[0] - v[4]) * C1[0][0]
    v2a2 = v[2] * C1[2][0]
    v2a6 = v[2] * C1[2][1]
    v6a2 = v[6] * C1[2][0]
    v6a6 = v[6] * C1[2][1]
    result = [0 for _ in range(len(v))]
    for j in range(4):
        res0 += v[2*j+1] * Theta[j][0]
        res1 += v[2*j + 1] * Theta[j][1]
        res2 += v[2*j + 1] * Theta[j][2]
        res3 += v[2*j+1] * Theta[j][3]
    result0 = v0a41 + v2a2 + v6a6
    result1 = v0a42 + v2a6 - v6a2
    result2 = v0a42 - v2a6 + v6a2
    result3 = v0a41 - v2a2 - v6a6
    result[0] = result0 + res0
    result[1] = result1 + res1
    result[2] = result2 + res2
    result[3] = result3 + res3
    result[4] = result0 - res0
    result[5] = result1 - res1
    result[6] = result2 - res2
    result[7] = result3 - res3
    return [result[0], result[1], result[2], result[3], result[7], result[6], result[5], result[4]]


def IDCT_Chen(A):
    a = [0 for _ in range(len(A))]
    b = [0 for _ in range(len(A[0]))]
    for i in range(len(A)):
        a[i] = ichenvec(A[i])
    a2 = trans(a)
    for j in range(len(A[0])):
        b[j] = ichenvec(a2[j])
    return trans(b)


m = [
    [140,  144,  147,  140,  140,  155,  179,  175],
    [144,  152,  140,  147,  140,  148,  167,  179],
    [152,  155,  136,  167,  163,  162,  152,  172],
    [168,  145,  156,  160,  152,  155,  136,  160],
    [162,  148,  156,  148,  140,  136,  147,  162],
    [147,  167,  140,  155,  155,  140,  136,  162],
    [136,  156,  123,  167,  162,  144,  140,  147],
    [148,  155,  136,  155,  152,  147,  147,  136],
]
print(IDCT_Chen(DCT_Chen(m)))


def quantization(A, Q):
    qinv = [[0 for _ in range(len(Q[0]))] for _ in range(len(Q))]
    for i in range(len(Q)):
        for j in range(len(Q[0])):
            qinv[i][j] = round((A[i][j])/(Q[i][j]))
    return qinv


def quantizationI(A, Q):
    qinv = [[0 for _ in range(len(Q[0]))] for _ in range(len(Q))]
    for i in range(len(Q)):
        for j in range(len(Q[0])):
            qinv[i][j] = round((A[i][j])*(Q[i][j]))
    return qinv


# m = [
#     [140,  144,  147,  140,  140,  155,  179,  175],
#     [144,  152,  140,  147,  140,  148,  167,  179],
#     [152,  155,  136,  167,  163,  162,  152,  172],
#     [168,  145,  156,  160,  152,  155,  136,  160],
#     [162,  148,  156,  148,  140,  136,  147,  162],
#     [147,  167,  140,  155,  155,  140,  136,  162],
#     [136,  156,  123,  167,  162,  144,  140,  147],
#     [148,  155,  136,  155,  152,  147,  147,  136],
# ]

# print(quantizationI(quantization(m, m), m))
def S(phi):
    if phi >= 50:
        ans = 200 - 2*phi
    ans = round(5000/phi)
    return ans


def Qmatrix(isY, phi):
    LQM = [
        [16, 11, 10, 16,  24,  40,  51,  61],
        [12, 12, 14, 19,  26,  58,  60,  55],
        [14, 13, 16, 24,  40,  57,  69,  56],
        [14, 17, 22, 29,  51,  87,  80,  62],
        [18, 22, 37, 56,  68, 109, 103,  77],
        [24, 35, 55, 64,  81, 104, 113,  92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103,  99],
    ]

    CQM = [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
    one = [[0 for _ in range(8)] for _ in range(8)]
    if isY == True:
        one[i][j] = math.ceil((50 + S(phi)*LQM[i][j])/100)
    else:
        one[i][j] = math.ceil((50 + S(phi)*CQM[i][j])/100)

    return one

# first creating a list of the path, then yielding from it


def zigzag(A):
    solution = [[] for _ in range(15)]
    i = 0
    j = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            sum = i + j
            if sum % 2 != 0:
                solution[sum].append(A[i][j])
            else:
                solution[sum].insert(0, A[i][j])
    for one in solution:
        for two in one:
            yield two


def rle0(g):
    counter = 0
    for i in g:
        if i == 0:
            counter += 1
        else:
            yield (counter, i)
            counter = 0
