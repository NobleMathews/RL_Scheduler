import numbers
import os

import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pe
# import gurobipy as gb


class STN(object):
    def __init__(self):
        # simulation objects
        self.states = set()  # set of state names
        self.tasks = set()  # set of task names
        self.units = set()  # set of unit names
        self.TIME = []  # time grid

        # dictionaries indexed by task name
        self.S = {}  # sets of states feeding each task (inputs)
        self.S_ = {}  # sets of states fed by each task (outputs)
        self.K = {}  # sets of units capable of each task
        self.p = {}  # task durations

        # dictionaries indexed by state name
        self.T = {}  # tasks fed from each state (task output)
        self.T_ = {}  # tasks feeding each state (task inputs)
        self.C = {}  # capacity of each task
        self.init = {}  # initial level
        self.price = {}  # prices of each state
        self.storage_cost = {}

        # dictionary indexed by unit
        self.I = {}  # sets of tasks performed by each unit

        # dictionaries indexed by (task,state)
        self.rho = {}  # input feed fractions
        self.rho_ = {}  # output product dispositions
        self.P = {}  # time to finish output from task to state

        # characterization of units indexed by (task, unit)
        self.Bmax = {}  # max capacity of unit j for task i
        self.Bmin = {}  # minimum capacity of unit j for task i
        self.cost = {}
        self.vcost = {}

        # dictionaries indexed by (task,task)
        self.changeoverTime = {}  # switch over times required for task1 -> task2

    # defines states as .state(name, capacity, init)
    def state(self, name, capacity=float("inf"), init=0, price=0, storage_cost=0):
        self.states.add(name)  # add to the set of states
        self.storage_cost[name] = storage_cost
        self.C[name] = capacity  # state capacity
        self.init[name] = init  # state initial value
        self.T[name] = set()  # tasks which feed this state (inputs)
        self.T_[name] = set()  # tasks fed from this state (outputs)
        self.price[name] = price  # per unit price of each state

    def task(self, name):
        self.tasks.add(name)  # add to set of tasks
        self.S[name] = set()  # states which feed this task (inputs)
        self.S_[name] = set()  # states fed by this task (outputs)
        self.p[name] = 0  # completion time for this task
        self.K[name] = set()

    def stArc(self, state, task, rho=1):
        if state not in self.states:
            self.state(state)
        if task not in self.tasks:
            self.task(task)
        self.S[task].add(state)
        self.rho[(task, state)] = rho
        self.T[state].add(task)

    def tsArc(self, task, state, rho=1, dur=1):
        if state not in self.states:
            self.state(state)
        if task not in self.tasks:
            self.task(task)
        self.S_[task].add(state)
        self.T_[state].add(task)
        self.rho_[(task, state)] = rho
        self.P[(task, state)] = dur
        self.p[task] = max(self.p[task], dur)

    def unit(self, unit, task, Bmin=0, Bmax=float("inf"), cost=0, vcost=0):
        if unit not in self.units:
            self.units.add(unit)
            self.I[unit] = set()
        if task not in self.tasks:
            self.task(task)
        self.I[unit].add(task)
        self.K[task].add(unit)
        self.Bmin[(task, unit)] = Bmin
        self.Bmax[(task, unit)] = Bmax
        self.cost[(task, unit)] = cost
        self.vcost[(task, unit)] = vcost

    def changeover(self, task1, task2, dur):
        self.changeoverTime[(task1, task2)] = dur

    def pprint(self):
        for task in sorted(self.tasks):
            print("\nTask:", task)
            print("    S[{0:s}]:".format(task), self.S[task])
            print("    S_[{0:s}]:".format(task), self.S_[task])
            print("    K[{0:s}]:".format(task), self.K[task])
            print("    p[{0:s}]:".format(task), self.p[task])

        for state in sorted(self.states):
            print("\nState:", state)
            print("    T[{0:s}]:".format(state), self.T[state])
            print("    T_[{0:s}]:".format(state), self.T_[state])
            print("    C[{0:s}]:".format(state), self.C[state])
            print("    init[{0:s}]:".format(state), self.init[state])

        for unit in sorted(self.units):
            print("\nUnit:", unit)
            print("    I[{0:s}]:".format(unit), self.I[unit])

        print("\nState -> Task Arcs")
        for (task, state) in sorted(self.rho.keys()):
            print("    {0:s} -> {1:s}:".format(state, task))
            print("        rho:", self.rho[(task, state)])

        print("\nTask -> State Arcs")
        for (task, state) in sorted(self.rho_.keys()):
            print("    {0:s} -> {1:s}:".format(task, state))
            print("        rho_:", self.rho_[(task, state)])
            print("           P:", self.P[(task, state)])

    def build(self, TIME):

        self.TIME = np.array([t for t in TIME])
        self.H = max(self.TIME)
        self.model = pe.ConcreteModel()
        m = self.model
        # m.cons = pe.ConstraintList()
        m.cons = pe.Constraint(pe.Any)

        # W[i,j,t] 1 if task i starts in unit j at time t
        m.W = pe.Var(self.tasks, self.units, self.TIME, domain=pe.Boolean)  # domain=pe.Boolean)

        # B[i,j,t] size of batch assigned to task i in unit j at time t
        m.B = pe.Var(self.tasks, self.units, self.TIME, domain=pe.NonNegativeReals)

        # S[s,t] inventory of state s at time t
        m.S = pe.Var(self.states, self.TIME, domain=pe.NonNegativeReals)

        # Q[j,t] inventory of unit j at time t
        m.Q = pe.Var(self.units, self.TIME, domain=pe.NonNegativeReals)

        # objectve
        m.Cost = pe.Var(domain=pe.NonNegativeReals)
        m.StorageCost = pe.Var(domain=pe.NonNegativeReals)
        m.Value = pe.Var(domain=pe.NonNegativeReals)
        m.cons["Value"]=(
            m.Value == sum([self.price[s] * m.S[s, self.H] for s in self.states])
        )
        m.cons["Cost"]=(
            m.Cost
            == sum(
                [
                    self.cost[(i, j)] * m.W[i, j, t] + self.vcost[(i, j)] * m.B[i, j, t]
                    for i in self.tasks
                    for j in self.K[i]
                    for t in self.TIME
                ]
            )
        )
        m.cons["StorageCost"]=(
            m.StorageCost
            == sum(
                [
                    self.storage_cost[s] * m.S[s, t]
                    for s in self.states
                    for t in self.TIME
                ]
            )
        )

        m.Obj = pe.Objective(expr=m.Value - m.Cost - m.StorageCost, sense=pe.maximize)

        # unit constraints
        for j in self.units:
            rhs = 0
            for t in self.TIME:
                # a unit can only be allocated to one task
                lhs = 0
                for i in self.I[j]:
                    for tprime in self.TIME[
                        (self.TIME <= t) & (self.TIME >= t - self.p[i] + 1)
                    ]:
                        lhs += m.W[i, j, tprime]
                m.cons[f"unit_constraint_{j}_time_{t}"]=(lhs <= 1)

                # capacity constraints (see Konkili, Sec. 3.1.2)
                for i in self.I[j]:
                    m.cons[f"capacity_constraint1_{j}_time_{t}_{i}"]=(m.W[i, j, t] * self.Bmin[i, j] <= m.B[i, j, t])
                    m.cons[f"capacity_constraint2_{j}_time_{t}_{i}"]=(m.B[i, j, t] <= m.W[i, j, t] * self.Bmax[i, j])

                # unit mass balance
                rhs += sum([m.B[i, j, t] for i in self.I[j]])
                for i in self.I[j]:
                    for s in self.S_[i]:
                        if t >= self.P[(i, s)]:
                            rhs -= (
                                    self.rho_[(i, s)]
                                    * m.B[
                                        i,
                                        j,
                                        max(self.TIME[self.TIME <= t - self.P[(i, s)]]),
                                    ]
                            )
                m.cons[f"unit_mass_balance_{j}_time_{t}"]=(m.Q[j, t] == rhs)
                rhs = m.Q[j, t]

                # switchover time constraints
                for (i1, i2) in self.changeoverTime.keys():
                    if (i1 in self.I[j]) and (i2 in self.I[j]):
                        for t1 in self.TIME[self.TIME <= (self.H - self.p[i1])]:
                            for t2 in self.TIME[
                                (self.TIME >= t1 + self.p[i1])
                                & (
                                        self.TIME
                                        < t1 + self.p[i1] + self.changeoverTime[(i1, i2)]
                                )
                            ]:
                                m.cons[f"switchover_{i1}_{i2}_{t1}_{t2}_{j}_time_{t}"]=(m.W[i1, j, t1] + m.W[i2, j, t2] <= 1)

                # terminal condition
                m.cons[f"terminal_condition_{j}_time_{t}"]=(m.Q[j, self.H] == 0)

        # state constraints
        for s in self.states:
            rhs = self.init[s]
            for t in self.TIME:
                # state capacity constraint
                m.cons[f"state_{s}_capacity_constraint_time_{t}"]=(m.S[s, t] <= self.C[s])
                # state mass balance
                for i in self.T_[s]:
                    for j in self.K[i]:
                        if t >= self.P[(i, s)]:
                            rhs += (
                                    self.rho_[(i, s)]
                                    * m.B[
                                        i,
                                        j,
                                        max(self.TIME[self.TIME <= t - self.P[(i, s)]]),
                                    ]
                            )
                for i in self.T[s]:
                    rhs -= self.rho[(i, s)] * sum([m.B[i, j, t] for j in self.K[i]])
                m.cons[f"state_{s}_mass_balance_time_{t}"]=(m.S[s, t] == rhs)
                rhs = m.S[s, t]

    # def solve(self, solver="glpk"):
    #     self.solver = pe.SolverFactory(solver)
    #     self.solver.solve(self.model).write()

    def gantt(self):
        model = self.model
        C = self.C
        H = self.H
        I = self.I
        p = self.p

        gap = H / 400
        idx = 1
        lbls = []
        ticks = []

        # create a list of units sorted by time of first assignment
        jstart = {j: H + 1 for j in self.units}
        for j in self.units:
            for i in I[j]:
                for t in self.TIME:
                    if self.model.W[i, j, t]() > 0:
                        jstart[j] = min(jstart[j], t)
        jsorted = [j for (j, t) in sorted(jstart.items(), key=lambda x: x[1])]

        # number of horizontal bars to draw
        nbars = -1
        for j in jsorted:
            for i in sorted(I[j]):
                nbars += 1
            nbars += 0.5
        plt.figure(figsize=(12, (nbars + 1) / 2))

        for j in jsorted:
            idx -= 0.5
            for i in sorted(I[j]):
                idx -= 1
                ticks.append(idx)
                lbls.append("{0:s} -> {1:s}".format(j, i))
                plt.plot([0, H], [idx, idx], lw=24, alpha=0.3, color="y")
                for t in self.TIME:
                    if model.W[i, j, t]() > 0:
                        plt.plot(
                            [t, t + p[i]],
                            [idx, idx],
                            "k",
                            lw=24,
                            alpha=0.5,
                            solid_capstyle="butt",
                        )
                        plt.plot(
                            [t + gap, t + p[i] - gap],
                            [idx, idx],
                            "b",
                            lw=20,
                            solid_capstyle="butt",
                        )
                        txt = "{0:.2f}".format(model.B[i, j, t]())
                        plt.text(
                            t + p[i] / 2,
                            idx,
                            txt,
                            color="white",
                            weight="bold",
                            ha="center",
                            va="center",
                        )
        plt.xlim(0, self.H)
        plt.ylim(-nbars - 0.5, 0)
        plt.gca().set_yticks(ticks)
        plt.gca().set_yticklabels(lbls)

    def sim(self):
        # build a dataframe with columns t, j, i, batchsize, time2go

        df = []

        TIME = self.TIME
        for t in TIME:
            for j in self.units:
                for i in self.I[j]:
                    df.append([t, i, j, self.model.W[i, j, t](), self.p[i]])

        for li in df:
            print(li)

    def mprint(self):
        self.model.pprint()
        # model.cons1.pprint()  # For entire ConstraintList
        # print(model.cons2[i].expr)  # For only one index of ConstraintList
        # model.write()  # To write the model into a file using .nl format

    def trace(self):
        # abbreviations
        model = self.model
        TIME = self.TIME
        dT = np.mean(np.diff(TIME))

        print("\nStarting Conditions")
        print("\n    Initial State Inventories are:")
        for s in self.states:
            print("        {0:10s}  {1:6.1f} kg".format(s, self.init[s]))

        # for tracking unit assignments
        # t2go[j]['assignment'] contains the task to which unit j is currently assigned
        # t2go[j]['t'] is the time to go on equipment j
        time2go = {j: {"assignment": "None", "t": 0} for j in self.units}

        for t in TIME:
            print("\nTime =", t, "hr")

            # create list of instructions
            strList = []

            # first unload units
            for j in self.units:
                time2go[j]["t"] -= dT
                fmt = "Transfer {0:.2f} kg from {1:s} to {2:s}"
                for i in self.I[j]:
                    for s in self.S_[i]:
                        ts = t - self.P[(i, s)]
                        if ts >= 0:
                            amt = (
                                    self.rho_[(i, s)]
                                    * model.B[i, j, max(TIME[TIME <= ts])]()
                            )
                            if amt > 0:
                                strList.append(fmt.format(amt, j, s))

            for j in self.units:
                # release units from tasks
                fmt = "Release {0:s} from {1:s}"
                for i in self.I[j]:
                    if t - self.p[i] >= 0:
                        if model.W[i, j, max(TIME[TIME <= t - self.p[i]])]() > 0:
                            strList.append(fmt.format(j, i))
                            time2go[j]["assignment"] = "None"
                            time2go[j]["t"] = 0

                # assign units to tasks
                fmt = "Assign {0:s} to {1:s} for {2:.2f} kg batch for {3:.1f} hours"
                for i in self.I[j]:
                    amt = model.B[i, j, t]()
                    if amt > 0:
                        strList.append(fmt.format(j, i, amt, self.p[i]))
                        time2go[j]["assignment"] = i
                        time2go[j]["t"] = self.p[i]

                # transfer from states to tasks/units
                fmt = "Transfer {0:.2f} from {1:s} to {2:s}"
                for i in self.I[j]:
                    for s in self.S[i]:
                        amt = self.rho[(i, s)] * model.B[i, j, t]()
                        if amt > 0:
                            strList.append(fmt.format(amt, s, j))

            if len(strList) > 0:
                print()
                idx = 0
                for str in strList:
                    idx += 1
                    print("   {0:2d}. {1:s}".format(idx, str))

            print("\n    State Inventories are now:")
            for s in self.states:
                print("        {0:10s}  {1:6.1f} kg".format(s, model.S[s, t]()))

            print("\n    Unit Assignments are now:")
            fmt = "        {0:s}: {1:s}, {2:.2f} kg, {3:.1f} hours to go."
            for j in self.units:
                if time2go[j]["assignment"] != "None":
                    print(
                        fmt.format(
                            j,
                            time2go[j]["assignment"],
                            model.Q[j, t](),
                            time2go[j]["t"],
                        )
                    )
                else:
                    print("        {0:s} is unassigned".format(j))


stn = STN()

# states
stn.state('FeedA', init=200)
stn.state('FeedB', init=200)
stn.state('FeedC', init=200)
stn.state('HotA', price=-1, storage_cost=1)
stn.state('IntAB', price=-1)
stn.state('IntBC', price=-1)
stn.state('ImpureE', price=-1)
stn.state('Product_1', price=10)
stn.state('Product_2', price=10)

# state to task arcs
stn.stArc('FeedA', 'Heating')
stn.stArc('FeedB', 'Reaction_1', rho=0.5)  # input feed fractions
stn.stArc('FeedC', 'Reaction_1', rho=0.5)
stn.stArc('HotA', 'Reaction_2', rho=0.4)
stn.stArc('IntBC', 'Reaction_2', rho=0.6)
stn.stArc('FeedC', 'Reaction_3', rho=0.2)
stn.stArc('IntAB', 'Reaction_3', rho=0.8)
stn.stArc('ImpureE', 'Separation')

# task to state arcs
stn.tsArc('Heating', 'HotA', rho=1.0, dur=1)
stn.tsArc('Reaction_1', 'IntBC', dur=2)
stn.tsArc('Reaction_2', 'IntAB', rho=0.6, dur=2)
stn.tsArc('Reaction_2', 'Product_1', rho=0.4, dur=2)
stn.tsArc('Reaction_3', 'ImpureE', dur=1)
stn.tsArc('Separation', 'IntAB', rho=0.1, dur=2)
stn.tsArc('Separation', 'Product_2', rho=0.9, dur=1)

# unit-task data
stn.unit('Heater', 'Heating', Bmin=0, Bmax=100)  # min,max capacity of unit for task
stn.unit('Reactor_1', 'Reaction_1', Bmin=0, Bmax=80)
stn.unit('Reactor_1', 'Reaction_2', Bmin=0, Bmax=80)
stn.unit('Reactor_1', 'Reaction_3', Bmin=0, Bmax=80)
stn.unit('Reactor_2', 'Reaction_1', Bmin=0, Bmax=50)
stn.unit('Reactor_2', 'Reaction_2', Bmin=0, Bmax=50)
stn.unit('Reactor_2', 'Reaction_3', Bmin=0, Bmax=50)
stn.unit('Still', 'Separation', Bmin=0, Bmax=200)

H = 10
stn.build(range(0, H + 1))

# stn.solve('gurobi')

m = stn.model

filename = os.path.join(os.path.dirname(__file__), 'model.lp')
m.write(filename, io_options={'symbolic_solver_labels': True})

# model.getObjective()

# denseMatrix = pd.DataFrame(data=sp.sparse.csr_matrix.todense(A))
# denseMatrix.to_csv("Amatrix.csv", index=False)

# solver = SolverFactory("gurobi_persistent", model=model)
# A = solver._solver_model.getA()


# def _add_cut(xval):
#     # a function to generate the cut
#     m.x.value = xval
#     return m.cons.add(m.y >= (m.x - 2) ** 2)
#
#
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
# opt.solve().write()

# model_vars = model.component_map(ctype=pe.Var)
#
# serieses = []   # collection to hold the converted "serieses"
# for k in model_vars.keys():   # this is a map of {name:pyo.Var}
#     v = model_vars[k]
#
#     # make a pd.Series from each
#     s = pd.Series(v.extract_values(), index=v.extract_values().keys())
#
#     # if the series is multi-indexed we need to unstack it...
#     if type(s.index[0]) == tuple:  # it is multi-indexed
#         s = s.unstack(level=1)
#     else:
#         s = pd.DataFrame(s)         # force transition from Series -> df
#     #print(s)
#
#     # multi-index the columns
#     s.columns = pd.MultiIndex.from_tuples([(k, t) for t in s.columns])
#
#     serieses.append(s)
#
# df = pd.concat(serieses, axis=1)
# print(df)

# m.rc = pe.Suffix(direction=pe.Suffix.IMPORT)
# solver = pe.SolverFactory('gurobi')
# solver.solve(m)
#
# # .write()
# # m.rc.display()
# ObjVal = m.Value.value
# # RC = m.rc.items()
# RC = m.rc.extractValues()

# for con in m.cons.items():
#     print(con[1].expr)
#     num, var = (1, con[1].expr.args[0])
#     if "args" in dir(con[1].expr.args[1]):
#         for comp in con[1].expr.args[1].args:
#             if "args" in dir(comp):
#                 num, var = comp.args
#             else:
#                 num, var = (1, comp)
#     else:
#         num, var = (con[1].expr.args[1], None)
#     assert isinstance(num, numbers.Number)
#     assert var is None or pe.is_variable_type(var)

# # opt = pe.SolverFactory("gurobi_persistent")
# # opt.set_instance(m)
# # opt.set_gurobi_param('PreCrush', 1)
# # opt.set_gurobi_param('LazyConstraints', 1)
# #
# #
# # def my_callback(cb_m, cb_opt, cb_where):
# #     var_list = [v for v in m.component_data_objects(pe.Var, descend_into=True)]
# #     if cb_where == GRB.Callback.MIPNODE:
# #         status = cb_opt.cbGet(GRB.Callback.MIPNODE_STATUS)
# #         if status == GRB.OPTIMAL:
# #             print(cb_opt.cbGetNodeRel(var_list))
# #     # if cb_where == GRB.Callback.MIPSOL:
# #     #     cb_opt.cbGetSolution(vars=var_list)
# #     #     print()
# #         # if m.y.value < (m.x.value - 2) ** 2 - 1e-6:
# #         #     cb_opt.cbLazy(_add_cut(m.x.value))
# #
# # opt.set_callback(my_callback)
# # opt.solve()
