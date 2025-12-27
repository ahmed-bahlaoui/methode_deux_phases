import numpy as np


def simplex(tableau):
    m, n = tableau.shape

    while True:
        ok = 1
        for j in range(n - 1):
            if tableau[m - 1, j] < -1e-9:
                ok = 0

        if ok == 1:
            return tableau, "OPTIMAL"

        col = np.argmin(tableau[m - 1, :-1])

        borne = 0
        for i in range(m - 1):
            if tableau[i, col] > 1e-9:
                borne = 1

        if borne == 0:
            return tableau, "UNBOUNDED"

        ratios = []
        for i in range(m - 1):
            if tableau[i, col] > 1e-9:
                ratios.append(tableau[i, -1] / tableau[i, col])
            else:
                ratios.append(np.inf)

        row = np.argmin(ratios)

        if ratios[row] == np.inf:
            return tableau, "UNBOUNDED"

        pivot = tableau[row, col]
        tableau[row, :] = tableau[row, :] / pivot

        for i in range(m):
            if i != row:
                facteur = tableau[i, col]
                tableau[i, :] = tableau[i, :] - facteur * tableau[row, :]


def phase1(A, b):
    m, n = A.shape
    I = np.eye(m)

    tableau = np.hstack([A, I, b.reshape(-1, 1)])

    w = -np.sum(tableau, axis=0)
    tableau = np.vstack([tableau, w])

    for i in range(m):
        idx = n + i
        f = tableau[-1, idx]
        tableau[-1, :] = tableau[-1, :] - f * tableau[i, :]

    print("phase 1")
    return simplex(tableau)


def phase2(A, b, c):
    tab1, status = phase1(A, b)

    if abs(tab1[-1, -1]) > 1e-6:
        print("probleme impossible")
        return None, "INFEASIBLE"

    m, n = A.shape

    tab2 = tab1[:-1, :n]
    rhs = tab1[:-1, -1].reshape(-1, 1)
    tab2 = np.hstack([tab2, rhs])

    c_row = np.append(c, 0)
    tab2 = np.vstack([tab2, c_row])

    for j in range(n):
        col = tab2[:-1, j]
        if np.sum(np.abs(col)) == 1 and np.max(col) == 1:
            i = np.argmax(col)
            coeff = tab2[-1, j]
            if coeff != 0:
                tab2[-1, :] = tab2[-1, :] - coeff * tab2[i, :]

    print("phase 2")
    return simplex(tab2)


def saisir():
    print("simplexe 2 phases")

    typ = input("min ou max : ")
    n = int(input("nombre de variables : "))

    c = list(map(float, input("coefficients objectif : ").split()))
    c = np.array(c)

    if typ == "max":
        c = -c

    m = int(input("nombre de contraintes : "))

    A = []
    b = []
    slack = []

    for i in range(m):
        print("contrainte", i + 1)
        coeffs = list(map(float, input("coeffs : ").split()))
        op = input("op (<= >= =) : ")
        rhs = float(input("rhs : "))

        col = [0] * m
        if op == "<=":
            col[i] = 1
            slack.append(col)
        elif op == ">=":
            col[i] = -1
            slack.append(col)

        A.append(coeffs)
        b.append(rhs)

    A = np.array(A)
    b = np.array(b)

    if slack:
        S = np.array(slack).T
        A = np.hstack([A, S])
        c = np.concatenate([c, np.zeros(len(slack))])

    return A, b, c, typ


if __name__ == "__main__":
    A, b, c, typ = saisir()

    tab, status = phase2(A, b, c)

    if tab is not None:
        z = tab[-1, -1]
        if typ == "max":
            z = -z

        print("status :", status)
        print("z =", z)

        x = np.zeros(len(c))
        for j in range(len(c)):
            col = tab[:-1, j]
            if np.sum(np.abs(col)) == 1 and np.max(col) == 1:
                i = np.argmax(col)
                x[j] = tab[i, -1]

        print("variables :", x)
