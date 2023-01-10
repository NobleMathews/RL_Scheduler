import numpy as np
from gurobipy import *

# min c^T x, Ax <= b, x>=0


def generatecutzeroth(row):
    ###
    # generate cut that includes cost/obj row as well
    ###
    n = row.size
    a = row[1:n]
    b = row[0]
    cut_a = a - np.floor(a)
    cut_b = b - np.floor(b)
    return cut_a, cut_b


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


def gurobi_solve(A, b, c, Method=0):
    c = -c  # Gurobi default is maximization
    varrange = range(c.size)
    crange = range(b.size)
    # m = Model("LP")
    m = read("model.lp")
    for con in m.getConstrs():
        con.Sense = '='
    m.params.OutputFlag = 0  # suppress output
    X = m.addVars(
        varrange, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=c, name="X"
    )
    _C = m.addConstrs(
        (sum(A[i, j] * X[j] for j in varrange) == b[i] for i in crange), "C"
    )
    m.params.Method = Method  # primal simplex Method = 0
    # print('start optimizing...')
    m.optimize()
    # obtain results
    solution = []
    basis_index = []
    RC = []
    for i in m.getVars():
        solution.append(i.X)
        RC.append(i.getAttr("RC"))
        if i.getAttr("VBasis") == 0:
            basis_index.append(i.index)
    solution = np.asarray(solution)
    RC = np.asarray(RC)
    basis_index = np.asarray(basis_index)
    # print('solving completes')
    return m.ObjVal, solution, basis_index, RC


def roundmarrays(x, delta=1e-7):
    """
    if certain components of x are very close to integers, round them
    """
    index = np.where(abs(np.round(x) - x) < delta)
    x[index] = np.round(x)[index]
    return x


def computeoptimaltab(A, b, RC, obj, basis_index):
    m, n = A.shape
    assert m == b.size
    assert n == RC.size
    B = A[:, basis_index]
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


def compute_state(A, b, c):
    m, n = A.shape
    assert m == b.size and n == c.size
    A_tilde = np.eye(m)
    b_tilde = b
    c_tilde = np.zeros(m)
    obj, sol, basis_index, rc = gurobi_solve(A_tilde, b_tilde, c_tilde)
    tab = computeoptimaltab(np.column_stack((A, np.eye(m))), b, rc, obj, basis_index)
    tab = roundmarrays(tab)
    x = tab[:, 0]
    # print(tab)
    done = True
    if np.sum(abs(np.round(x) - x) > 1e-2) >= 1:
        done = False
    cuts_a = []
    cuts_b = []
    for i in range(x.size):
        if abs(round(x[i]) - x[i]) > 1e-2:
            # fractional rows used to compute cut
            cut_a, cut_b = generatecutzeroth(tab[i, :])
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
