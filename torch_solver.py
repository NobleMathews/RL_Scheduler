import numpy as np
from gurobipy import *
from itertools import chain


# min c^T x, Ax <= b, x>=0


def generatecutzeroth(row, NI):
    ###
    # generate cut that includes cost/obj row as well
    ###
    # n = row.size
    # a = row[1:n]
    # b = row[0]
    # cut_a = a - np.floor(a)
    # cut_b = b - np.floor(b)
    # return cut_a, cut_b
    n = row.size
    a = row[1:n]
    b = row[0]
    fij = a - np.floor(a)
    fio = b - np.floor(b)
    for i,fj in enumerate(fij):
        if i in NI:
            pass
    sense = ">"
    return cut_a, fio, sense


def updatetab(tab, cut_a, cut_b, basis_index):
    cut_a = -cut_a
    cut_b = -cut_b
    m, n = tab.shape
    A_ = tab[1:m, 1:n]
    b_ = tab[1:m, 0]
    c_ = tab[0, 1:n]
    obj = tab[0, 0]
    Anew1 = np.column_stack((A_, np.zeros(m - 1)))
    Anew2 = np.append(cut_a, 1)
    Anew = np.vstack((Anew1, Anew2))
    bnew = np.append(b_, cut_b)
    cnew = np.append(c_, 0)
    M1 = np.append(obj, cnew)
    M2 = np.column_stack((bnew, Anew))
    newtab = np.vstack((M1, M2))
    basis_index = np.append(basis_index, n - 1)
    return newtab, basis_index, Anew, bnew


def gurobi_solve(A, b, c, sense, Method=0):
    c = -c  # Gurobi default is maximization
    varrange = range(c.size)
    crange = range(b.size)
    m = Model("LP")
    m.params.OutputFlag = 0  # suppress output
    X = m.addVars(
        varrange, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=c, name="X"
    )
    _C = m.addConstrs((sum(A[i, j] * X[j] for j in varrange) == b[i] for i in crange), "C")
    # _C_e = m.addConstrs(
    #     (sum(A[i, j] * X[j] for j in varrange) == b[i] for i in crange if not sense or sense and sense[i] == "="), "C"
    # )
    # _C_l = m.addConstrs(
    #     (sum(A[i, j] * X[j] for j in varrange) == b[i] for i in crange if sense and sense[i] == "<"), "C"
    # )
    # _C_g = m.addConstrs(
    #     (sum(A[i, j] * X[j] for j in varrange) == b[i] for i in crange if sense and sense[i] == ">"), "C"
    # )
    # _C = m.addConstrs(
    #     (get_constr(A, b, X, i, varrange, sense) for i in crange), "C"
    # )
    # chained = chain.from_iterable(
    #     [(sum(A[i, j] * X[j] for j in varrange) == b[i] for i in crange if not sense or sense and sense[i] == "="),
    #      (sum(A[i, j] * X[j] for j in varrange) <= b[i] for i in crange if sense and sense[i] == "<"),
    #      (sum(A[i, j] * X[j] for j in varrange) >= b[i] for i in crange if sense and sense[i] == ">")]
    # )
    # _C = m.addConstrs(chained, "C")
    m.params.Method = Method  # primal simplex Method = 0
    # print('start optimizing...')
    m.optimize()
    # obtain results
    solution = []
    basis_index = []
    RC = []
    for i in X:
        solution.append(X[i].X)
        RC.append(X[i].getAttr("RC"))
        if X[i].getAttr("VBasis") == 0:
            basis_index.append(i)
    # for i in _C:
    #     if _C[i].getAttr("CBasis") == 0:
    #         print(i)
    # cb1 = [x for x in _C_l if _C_l[x].getAttr("CBasis") == 0]
    # cb2 = [x for x in _C_g if _C_g[x].getAttr("CBasis") == 0]
    # cb3 = [x for x in _C_e if _C_e[x].getAttr("CBasis") == 0]
    cb = [x for x in _C if _C[x].getAttr("CBasis") == 0]
    solution = np.asarray(solution)
    RC = np.asarray(RC)
    basis_index = np.asarray(basis_index)
    identity_index = np.asarray(cb)
    # print('solving completes')
    return m.ObjVal, solution, basis_index, identity_index, RC


def roundmarrays(x, delta=1e-7):
    """
    if certain components of x are very close to integers, round them
    """
    index = np.where(abs(np.round(x) - x) < delta)
    x[index] = np.round(x)[index]
    return x


def computeoptimaltab(A, b, RC, obj, basis_index, identity_index):
    m, n = A.shape
    assert m == b.size
    assert n == RC.size
    B = A[:, basis_index]
    if identity_index:
        B = np.concatenate((B, np.eye(m)[:, identity_index]), axis=1)
    try:
        INV = np.linalg.inv(B)
    except:
        print("basisindex length:", basis_index.size)
        print("Ashape:", A.shape)
        raise ValueError
    x = np.dot(INV, b)
    A_ = np.dot(INV, A)
    firstrow = np.append(-obj, RC)
    secondrow = np.column_stack((x, A_))
    tab = np.vstack((firstrow, secondrow))
    return tab


def compute_state(A, b, c, sense, integrality):
    m, n = A.shape
    assert m == b.size and n == c.size
    factor = []
    for x in sense:
        if x == "=":
            factor.append(0)
        elif x == ">":
            factor.append(-1)
        else:
            factor.append(1)
    A_tilde = np.column_stack((A, np.eye(m)*np.array(factor)[:, None]))
    b_tilde = b
    c_tilde = np.append(c, np.zeros(m))
    obj, sol, basis_index, identity_index, rc = gurobi_solve(A_tilde, b_tilde, c_tilde, sense)
    tab = computeoptimaltab(A_tilde, b_tilde, rc, obj, basis_index, identity_index)
    tab = roundmarrays(tab)
    x = tab[:, 0]
    # print(tab)
    done = True
    if np.sum(abs(np.round(x) - x) > 1e-2) >= 1:
        done = False
    cuts_a = []
    cuts_b = []
    for i in range(x.size):
        if i==0 or integrality[i+1] != "C":
            if abs(round(x[i]) - x[i]) > 1e-2:
                # fractional rows used to compute cut
                cut_a, cut_b, sense = generatecutzeroth(tab[i, :], integrality)
                # a^T x + e^T y >= d
                assert cut_a.size == m + n
                a = cut_a[0:n]
                e = cut_a[n:]
                newA = np.dot(A.T, e) - a
                newb = np.dot(e, b) - cut_b
                cuts_a.append(newA)
                cuts_b.append(newb)
    cuts_a, cuts_b = np.array(cuts_a), np.array(cuts_b)
    return A, b, cuts_a, cuts_b, done, obj, x, tab
