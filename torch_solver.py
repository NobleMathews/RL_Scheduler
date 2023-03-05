import numpy as np
from gurobipy import *
from itertools import chain


# min c^T x, Ax <= b, x>=0


def generatecutzeroth(row, p):
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
    fj = a - np.floor(a)
    f0 = b - np.floor(b)
    coeff = np.zeros(n - 1)
    for j, fj in enumerate(fj):
        if j < p:
            if fj <= f0:
                coeff[j] += fj / f0
            else:
                coeff[j] += (1 - fj) / (1 - f0)
        else:
            aj = a[j]
            if aj > 0:
                coeff[j] += aj / f0
            else:
                coeff[j] += -1 * aj / (1 - f0)
    # sense = ">"
    return coeff, 1


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


def gurobi_int_solve(A, b, c, sense, vtype, maximize=True):
    if maximize:
        c = -c  # Gurobi default is maximization
    varrange = range(c.size)
    crange = range(b.size)
    m = Model("LP")
    m.params.OutputFlag = 0  # suppress output
    X = m.addVars(
        varrange, lb=0.0, ub=GRB.INFINITY, vtype=[GRB.CONTINUOUS if vt == "C" else GRB.BINARY for vt in vtype], obj=c,
        name="X"
    )
    _C = m.addConstrs(
        (
            sum(A[i, j] * X[j] for j in varrange) >= b[i]
            if sense[i] == ">"
            else
            sum(A[i, j] * X[j] for j in varrange) <= b[i]
            if sense[i] == "<"
            else
            sum(A[i, j] * X[j] for j in varrange) == b[i]
            for i in crange
        ), "C")
    # _C_e = m.addConstrs(
    #     (sum(A[i, j] * X[j] for j in varrange) == b[i] for i in crange if not sense or sense and sense[i] == "="), "C"
    # )
    # _C_l = m.addConstrs(
    #     (sum(A[i, j] * X[j] for j in varrange) <= b[i] for i in crange if sense and sense[i] == "<"), "C"
    # )
    # _C_g = m.addConstrs(
    #     (sum(A[i, j] * X[j] for j in varrange) >= b[i] for i in crange if sense and sense[i] == ">"), "C"
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
    m.params.Method = -1  # primal simplex Method = 0
    # print('start optimizing...')
    m.optimize()
    # obtain results
    solution = []
    # basis_index = []
    # RC = []
    for i in X:
        solution.append(X[i].X)
        # RC.append(X[i].getAttr("RC"))
        # if X[i].getAttr("VBasis") == 0:
        #     basis_index.append(i)
    # for i in _C:
    #     if _C[i].getAttr("CBasis") == 0:
    #         print(i)
    # cb1 = [x for x in _C_l if _C_l[x].getAttr("CBasis") == 0]
    # cb2 = [x for x in _C_g if _C_g[x].getAttr("CBasis") == 0]
    # cb3 = [x for x in _C_e if _C_e[x].getAttr("CBasis") == 0]
    # cb = cb1 + cb2 + cb3
    # cb = []
    # for x in _C:
    #     RC.append(-1 * _C[x].getAttr("Pi"))
    #     solution.append(_C[x].Slack)
    #     if _C[x].getAttr("CBasis") == 0:
    #         cb.append(x)
    # solution = np.asarray(solution)
    # RC = np.asarray(RC)
    # basis_index = np.asarray(basis_index)
    # identity_index = np.asarray(cb)
    # # print('solving completes')
    # , basis_index, identity_index, RC
    return m.ObjVal, solution


def gurobi_solve(A, b, c, sense, Method=0, maximize=True):
    if maximize:
        c = -c  # Gurobi default is maximization
    varrange = range(c.size)
    crange = range(b.size)
    # assert b.size == len(sense)
    m = Model("LP")
    m.params.OutputFlag = 0  # suppress output
    X = m.addVars(
        varrange, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=c, name="X"
    )
    _C = m.addConstrs(
        (
            sum(A[i, j] * X[j] for j in varrange) >= b[i]
            if sense[i] == ">"
            else
            sum(A[i, j] * X[j] for j in varrange) <= b[i]
            if sense[i] == "<"
            else
            sum(A[i, j] * X[j] for j in varrange) == b[i]
            for i in crange
        ), "C")
    # _C_e = m.addConstrs(
    #     (sum(A[i, j] * X[j] for j in varrange) == b[i] for i in crange if not sense or sense and sense[i] == "="), "C"
    # )
    # _C_l = m.addConstrs(
    #     (sum(A[i, j] * X[j] for j in varrange) <= b[i] for i in crange if sense and sense[i] == "<"), "C"
    # )
    # _C_g = m.addConstrs(
    #     (sum(A[i, j] * X[j] for j in varrange) >= b[i] for i in crange if sense and sense[i] == ">"), "C"
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
    # cb = cb1 + cb2 + cb3
    cb = []
    for x in _C:
        RC.append(-1 * _C[x].getAttr("Pi"))
        solution.append(_C[x].Slack)
        if _C[x].getAttr("CBasis") == 0:
            cb.append(x)
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
    # assert n == RC.size
    B = A[:, basis_index]
    if len(identity_index):
        B = np.concatenate((B, np.eye(m)[:, identity_index]), axis=1)
    try:
        INV = np.linalg.inv(B)
    except:
        print("basisindex length:", basis_index.size)
        print("Ashape:", A.shape)
        raise ValueError
    A_tilde = np.column_stack((A, np.eye(m)))
    b_tilde = b
    x = np.dot(INV, b_tilde)
    A_ = np.dot(INV, A_tilde)
    firstrow = np.append(-obj, RC)
    secondrow = np.column_stack((x, A_))
    tab = np.vstack((firstrow, secondrow))
    return tab


# def get_row_integrality(integrality, basis_index, tab):
#     b_col_tab = basis_index + 1
#     tab[:,b_col_tab]


def compute_state(A, b, c, sense, integrality, maximize=True):
    m, n = A.shape
    assert m == b.size and n == c.size

    # factor = []
    # delete = []
    # for i, x in enumerate(sense):
    #     if x == "=":
    #         delete.append(i)
    #         factor.append(0)
    #     elif x == ">":
    #         factor.append(-1)
    #     else:
    #         factor.append(1)

    #         np.delete(np.eye(m)*np.array(factor)[:, None],delete, 1)
    # delete = []
    # A_tilde = np.column_stack((A, np.eye(m) * np.array(factor)[:, None]))
    # b_tilde = b
    # c_tilde = np.append(c, np.zeros(m - len(delete)))
    A_tilde = A
    b_tilde = b
    c_tilde = c
    obj, sol, basis_index, identity_index, rc = gurobi_solve(A_tilde, b_tilde, c_tilde, sense, maximize=maximize)
    print(len(A))
    print(obj)
    tab = computeoptimaltab(A_tilde, b_tilde, rc, obj, basis_index, identity_index)
    tab = roundmarrays(tab)
    x = tab[:, 0]
    # print(tab)
    # done = True
    # if np.sum(abs(np.round(x) - x) > 1e-2) >= 1:
    #     done = False
    cuts_a = []
    cuts_b = []
    cut_rows = []
    done = 0
    # which row corresponds to which variable - verify while solving
    # row_status = np.asarray(integrality)[basis_index[:len(integrality)]]
    # print(np.unique(integrality, return_counts=True))
    for i in range(x.size):
        if i >= len(integrality):
            break
        # row i -> basis of which
        # Sol => Integrality check  and integrality[i] != "C"
        if i != 0 and integrality[i - 1] != "C":
            if abs(round(x[i]) - x[i]) > 1e-2:
                # print(abs(round(x[i]) - x[i]))
                done += 1
                # fractional rows used to compute cut
                cut_a, cut_b = generatecutzeroth(tab[i, :], n)
                # a^T x + e^T y >= d
                assert cut_a.size == m + n
                a = cut_a[0:n]
                e = cut_a[n:]
                newA = np.dot(A.T, e) - a
                newb = np.dot(e, b) - cut_b
                cuts_a.append(newA)
                cuts_b.append(newb)
                cut_rows.append(i)
    print(done)
    cuts_a, cuts_b = np.array(cuts_a), np.array(cuts_b)
    # sense cuts a  <= cuts b
    return A, b, cuts_a, cuts_b, done, obj, x, tab, cut_rows
