import numpy
from gurobipy import *
import pyomo.environ as pe

A = numpy.asarray([[1, 1], [5, 9]])
b = numpy.asarray([6, 45])
c = numpy.asarray([5, 8])

m = Model()
m._cut_count = 0
c = -c  # Gurobi default is maximization
varrange = range(c.size)
crange = range(b.size)
X = m.addVars(
    varrange, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=c, name="X"
)
_C = m.addConstrs((sum(A[i, j] * X[j] for j in varrange) <= b[i] for i in crange), "C")

def cut_counter(model, where):
    cut_names = {
        'Clique:', 'Cover:', 'Flow cover:', 'Flow path:', 'Gomory:',
        'GUB cover:', 'Inf proof:', 'Implied bound:', 'Lazy constraints:',
        'Learned:', 'MIR:', 'Mod-K:', 'Network:', 'Projected Implied bound:',
        'StrongCG:', 'User:', 'Zero half:'}
    if where == GRB.Callback.MESSAGE:
        # Message callback
        msg = model.cbGet(GRB.Callback.MSG_STRING)
        if any(name in msg for name in cut_names):
            model._cut_count += int(msg.split(':')[1])


m.optimize(cut_counter)
print(m._cut_count)

# opt = pe.SolverFactory('gurobi_persistent')
# opt.set_instance(m)
# opt.set_gurobi_param('PreCrush', 1)
# opt.set_gurobi_param('LazyConstraints', 1)
#
#
# def my_callback(cb_m, cb_opt, cb_where):
#     var_list = [v for v in m.component_data_objects(pe.Var, descend_into=True)]
#     if cb_where == GRB.Callback.MIPSOL:
#         cb_opt.cbGetSolution(vars=var_list)
#         if m.y.value < (m.x.value - 2) ** 2 - 1e-6:
#             cb_opt.cbLazy(_add_cut(m.x.value))
#
#
# opt.set_callback(my_callback)