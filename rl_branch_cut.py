import numpy as np
import json
import time

from defaultlist import defaultlist
from gurobipy import *
from queue import PriorityQueue, Queue, LifoQueue
from torch_solver import compute_state, roundmarrays, gurobi_solve


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
    feasible = True
    try:
        for i in X:
            solution.append(X[i].X)
            # RC.append(X[i].getAttr("RC"))
            # if X[i].getAttr("VBasis") == 0:
            #     basis_index.append(i)
        objval = m.ObjVal
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
    except:
        feasible = False
        objval = None
    return feasible, objval, solution


class Node(object):
    def __init__(self, A, b, c, sense, VType, maximize, solution=None, reward_type='simple'):
        self.A = A
        self.b = b
        self.c = c
        self.sense = defaultlist(lambda: "<")
        for i in range(len(sense)):
            self.sense[i] = sense[i]
        self.VType = VType
        self.maximize = maximize
        self.solution = solution
        self.reward_type = reward_type


class NodeList(object):
    def __init__(self):
        self.nodes = []
        self.priorities = []

    def append(self, node, priority=1.0):
        self.nodes.append(node)
        self.priorities.append(priority)

    def sample(self):
        # choose the node with highest priority
        idx = np.argmax(self.priorities)
        node = self.nodes.pop(idx)
        self.priorities.pop(idx)
        return node

    def __len__(self):
        return len(self.nodes)


class NodeFIFOQueue(object):
    def __init__(self):
        self.nodes = Queue()
        self.priorities = []

    def append(self, node, prioriy=1.0):
        self.nodes.put(node)
        self.priorities.append(prioriy)

    def sample(self):
        return self.nodes.get()

    def __len__(self):
        return self.nodes.qsize()


class NodeLIFOQueue(object):
    def __init__(self):
        self.nodes = LifoQueue()
        self.priorities = []

    def append(self, node, prioriy=1.0):
        self.nodes.put(node)
        self.priorities.append(prioriy)

    def sample(self):
        return self.nodes.get()

    def __len__(self):
        return self.nodes.qsize()


def checkintegral(x):
    x = np.array(x)
    if np.sum(abs(np.round(x) - x) > 1e-2) >= 1:
        return False
    else:
        return True


class CutAdder(object):
    def __init__(self):
        pass

    def add_cuts(self):
        raise NotImplementedError


class timelimit_wrapper(object):
    def __init__(self, env, timelimit):
        self.env = env
        self.timelimit = timelimit
        self.counter = 0

    def reset(self):
        self.counter = 0
        return self.env.reset()

    def step(self, action, fake=False):
        if fake:
            return self.env.step(action, fake)
        self.counter += 1
        obs, reward, done, info = self.env.step(action, fake)
        if self.counter >= self.timelimit:
            done = 0
            print('forced return due to timelimit')
        return obs, reward, done, info


def make_float64(lists):
    newlists = []
    for e in lists:
        newlists.append(np.float64(e))
    return newlists


def check_feasibility(A, b, solution):
    RHS = np.dot(A, solution)
    # print(RHS - b)
    # print(RHS - (1.0 - 1e-10) * b)
    if np.sum(RHS - (1.0 - 1e-10) * b > 1e-5) >= 1:
        return False
    else:
        return True


class GurobiOriginalCutBBEnv(object):
    def __init__(self, A, b, c, sense, VType, maximize, solution=None, reward_type='simple'):
        """
        min c^T x, Ax <= b, x>=0
        """
        self.A0 = A.copy()
        self.A = A.copy()
        self.b0 = b.copy()
        self.b = b.copy()
        self.c0 = c.copy()
        self.c = c.copy()
        self.x = None
        self.sense0 = sense.copy()
        self.VType0 = VType.copy()
        self.sense = sense.copy()
        self.VType = VType.copy()
        self.maximize = maximize
        self.reward_type = reward_type
        assert reward_type in ['simple', 'obj']
        assert A.shape[1] == solution.size
        self.IPsolution = solution  # convention

    # upon init, check if the ip problem can be solved by lp
    # try:
    #	_, done = self._reset()
    #	assert done is False
    # except NotImplementedError:
    #	print('the env needs to be initialized with nontrivial ip')

    def check_init(self):
        _, done, _ = self._reset()
        return done

    def _reset(self):
        self.A, self.b, self.cuts_a, self.cuts_b, self.done, self.oldobj, self.x, self.tab, self.cut_rows = compute_state(
            self.A0,
            self.b0,
            self.c0,
            self.sense0,
            self.VType0,
            self.maximize)
        return (self.A, self.b, self.c0, self.cuts_a, self.cuts_b, self.x), self.done, self.cut_rows

    def reset(self):
        s, d, rows = self._reset()
        return s, rows

    def step(self, action, fake=False):
        cut_a, cut_b = self.cuts_a[action, :], self.cuts_b[action]
        if fake:
            _, _, _, _, done, newobj, _, _, _ = compute_state(
                np.vstack((self.A, cut_a)), np.append(self.b, cut_b), self.c,
                self.sense + ["<"], self.VType, self.maximize)
            reward = np.abs(self.oldobj - newobj)
            return (self.A, self.b, self.c0, self.cuts_a, self.cuts_b, self.x), reward, done, {}
        self.A = np.vstack((self.A, cut_a))
        self.b = np.append(self.b, cut_b)
        self.sense.append("<")
        try:
            self.A, self.b, self.cuts_a, self.cuts_b, self.done, self.newobj, self.x, self.tab, self.cut_rows = compute_state(
                self.A, self.b, self.c,
                self.sense, self.VType, self.maximize)
            if self.reward_type == 'simple':
                reward = -1.0
            elif self.reward_type == 'obj':
                reward = np.abs(self.oldobj - self.newobj)
        except Exception as e:
            print(e)
            print('error in lp iteration')
            self.done = 0
            reward = 0.0
        self.oldobj = self.newobj
        self.A, self.b, self.cuts_a, self.cuts_b = map(roundmarrays, [self.A, self.b, self.cuts_a, self.cuts_b])
        return (self.A, self.b, self.c0, self.cuts_a, self.cuts_b, self.x), reward, self.done, {}


class BaselineCutAdder(CutAdder):
    def __init__(self, max_num_cuts, backtrack, mode, policy, window=None, threshold=None):
        CutAdder.__init__(self)
        self.max_num_cuts = max_num_cuts
        self.backtrack = backtrack
        self.mode = mode
        self.policy = policy
        self.window = window
        self.threshold = threshold
        assert self.mode in ['random', 'maxviolation', 'maxnormviolation', 'rl']
        if self.mode == 'rl':
            assert policy is not None

    def add_cuts(self, A, b, c, sense, VType, maximize, solution):
        env = timelimit_wrapper(GurobiOriginalCutBBEnv(A, b, c, sense, VType, maximize, solution),
                                timelimit=self.max_num_cuts)
        A, b, feasible = elementaryrollout(env, self.policy, rollout_length=self.max_num_cuts, gamma=1.0,
                                           mode=self.mode, backtrack=self.backtrack, window=self.window,
                                           threshold=self.threshold)
        return A, b, feasible


def elementaryrollout(env, policy, rollout_length, gamma, mode, backtrack, window=None, threshold=None):
    # take in an environment
    # run cutting plane adding until termination
    # return both the LP bound and two newly branched LPs

    if backtrack:
        assert window is not None and threshold is not None

    A_orig = env.env.A.copy()
    b_orig = env.env.b.copy()

    if mode == 'rl':
        assert policy is not None
    rewards = []
    objs = []
    cutoffs = []
    times = []
    if True:
        ob, _ = env.reset()
        factor = 1.0
        # ob = env.reset()
        done = False
        t = 0
        rsum = 0
        cutoff = []
        obj = []
        backtrack_stats = []
        while not done and t <= rollout_length:
            # try:
            if True:
                if mode == 'rl':
                    action = policy.act(ob)
                # random acttion
                elif mode == 'random':
                    _, _, _, cutsa, cutsb = ob
                    if cutsb.size >= 1:
                        action = np.random.randint(0, cutsb.size, size=1)[0]
                    else:
                        action = []
                elif mode == 'maxviolation':
                    x = env.env.x.copy()
                    # reduce the solution to fractional part only
                    x_frac = []
                    for i in env.env.cut_rows:
                        if abs(x[i] - round(x[i])) > 1e-2:
                            x_frac.append(abs(x[i] - round(x[i])))
                    if len(x_frac) >= 1:
                        action = np.argmax(x_frac)
                    else:
                        action = []
                elif mode == 'maxnormviolation':
                    x = env.env.x.copy()
                    tab = env.env.tab.copy()
                    # reduce the solution to fractional part only
                    x_frac = []
                    # print(x.shape)
                    for i in env.env.cut_rows:
                        if abs(x[i] - round(x[i])) > 1e-2:
                            x_frac.append(abs(x[i] - round(x[i])) / np.linalg.norm(tab[i, 1:] + 1e-8))
                    if len(x_frac) >= 1:
                        action = np.argmax(x_frac)
                    else:
                        action = []
                else:
                    raise NotImplementedError
            # except:
            else:
                print('breaking')
                print(env.env.x)
                # print(env.env.done)
                break  # this case is when adding one branch terminates the process
            # print(action)
            # random
            # _,_,_,cutsa,cutsb = ob
            # action = np.random.randint(0, cutsb.size, size=1)[0]
            # ob, r, done = env.step(action)
            ob, r, done, info = env.step(action)
            rsum += r * factor
            factor *= gamma
            t += 1
            if r < 0:
                # cut off
                cutoff.append(1)
                gap = r + 1000.0
                obj.append(gap)
            else:
                cutoff.append(0)
                gap = r
                obj.append(r)

            # ==== backtracking mechanism ====
            if backtrack:
                backtrack_stats.append(r / np.sum(obj))
                if len(backtrack_stats) >= window:
                    last_stats = backtrack_stats[-window:]
                    last_stats = np.array(last_stats)
                    if np.sum(last_stats <= threshold) == window:
                        # save the cuts when backtrack stops
                        # np.save('backtrack_cuts/A_{}'.format(COUNT), env.env.A)
                        # np.save('backtrack_cuts/b_{}'.format(COUNT), env.env.b)
                        # OUNT += 1
                        break
            # ==========
        # np.save('random_backtrack_cuts/A_{}'.format(COUNT), env.env.A)
        # np.save('random_backtrack_cuts/b_{}'.format(COUNT), env.env.b)
        # COUNT += 1
        objs.append(obj)
        cutoffs.append(cutoff)
        rewards.append(rsum)
        times.append(t)

    A = env.env.A.copy()
    b = env.env.b.copy()

    # here we check if the cuts have cut off the optimal solution
    # if the original LP is infeasible - this does not matter
    feasible_original = check_feasibility(A_orig, b_orig, env.env.IPsolution)
    feasible_later = check_feasibility(A, b, env.env.IPsolution)
    feasible_cut = True
    if feasible_original:
        if not feasible_later:
            feasible_cut = False

    return A, b, feasible_cut


class NodeExpander(object):
    def __init__(self):
        pass

    def expandnode(self, node):
        # return expanded result and modify the node
        raise NotImplementedError


class LPExpander(NodeExpander):
    def __init__(self):
        NodeExpander.__init__(self)

    def expandnode(self, node):
        A, b, c, sense, vtype, maximize = node.A, node.b, node.c, node.sense, node.vtype, node.maximize
        feasible, objective, solution = gurobi_int_solve(A, b, c, sense, vtype, maximize)
        return feasible, objective, solution, True


# TODO: add parent node
class BaselineCutExpander(NodeExpander):
    def __init__(self, max_num_cuts, backtrack=False, mode=None, policy=None, window=None, threshold=None):
        NodeExpander.__init__(self)
        self.cutadder = BaselineCutAdder(max_num_cuts, backtrack, mode, policy, window, threshold)

    def expandnode(self, node):
        A, b, c, sense, vtype, maximize = node.A, node.b, node.c, node.sense, node.VType, node.maximize
        ipsolution = node.solution
        # solve lp to check if the problem is feasible
        lpfeasible, objective, lpsolution = gurobi_int_solve(A, b, c, sense, vtype, maximize)
        print('lp feasible', lpfeasible)
        if lpfeasible:
            # we can add cuts
            Anew, bnew, cutfeasible = self.cutadder.add_cuts(A, b, c, sense, VType, maximize, ipsolution)
            # solve the new lp
            newlpfeasible, newobjective, newlpsolution = gurobi_int_solve(Anew, bnew, c, sense, vtype, maximize)
            # modify nodes
            node.A = Anew
            node.b = bnew
            return newlpfeasible, newobjective, newlpsolution, cutfeasible
        else:
            return lpfeasible, objective, lpsolution, True


if __name__ == '__main__':

    initial_lp_objective = -2908.3599999999997

    # hyperparameters
    RATIO_THRESHOLD = 0.0001  # termination condition for BB
    TIMELIMIT = 1000  # termination time step
    max_num_cuts = 10  # max cuts added to each node
    backtrack = False  # do backtrack
    window = 5  # backtrack window
    threshold = 0.01  # backtrack threshold

    baselinemode = 'maxnormviolation'
    policy = None
    policyNAME = ''  # directory to load policy

    load_dir = "instances/kondili.json"

    # policy = load_policy(seed=0, n_directions=10, step_size=0.01, delta_std=0.02, policy_type='attention', numvars=30,
    #                      logdir=policyNAME)
    # baselinemode = 'rl'

    expander = BaselineCutExpander(max_num_cuts=max_num_cuts, backtrack=backtrack, mode=baselinemode, policy=policy,
                                   window=window, threshold=threshold)

    with open(load_dir, "r") as inputfile:
        input_data = json.load(inputfile)

    A0 = np.asarray(input_data["A0"])
    b0 = np.asarray(input_data["b0"])
    c0 = np.asarray(input_data["c0"])
    sense = input_data["sense"]
    VType = input_data["VType"]
    maximize = input_data["maximize"]
    solution = np.asarray(input_data["solution"])
    IPsolution = solution

    # create an initial node
    node = Node(A0, b0, c0, sense, VType, maximize, solution)

    # nodelist = NodeList()
    nodelist = NodeFIFOQueue()
    # nodelist = NodeLIFOQueue()

    # create a list to keep track of fractional solution
    # to form the lower bound on the objective
    fractionalsolutions = []
    childrennodes = []
    expanded = []

    # create initial best obj and solution
    BestObjective = np.inf
    BestSolution = None

    nodelist.append(node)

    # book keepinng
    timecount = 0
    ratios = []
    optimalitygap = []

    # main loop
    while len(nodelist) >= 1:

        # pop a node
        node = nodelist.sample()

        # load and expand a node
        # feasible, objective, solution = GurobiIntSolve2(A, b, c)
        originalnumcuts = node.A.shape[0]
        feasible, objective, solution, cutfeasible = expander.expandnode(node)
        A, b, c = node.A, node.b, node.c
        newnumcuts = node.A.shape[0]
        print('adding num of cuts {}'.format(newnumcuts - originalnumcuts))

        if feasible:
            assert objective is not None

        # check if thte popped node is the child node of some parent node
        for idx in range(len(childrennodes)):
            if childrennodes[idx][0] == node:
                expanded[idx][0] = 1
                if expanded[idx][1] == 1:
                    # pop the corresponding child node
                    childrennodes.pop(idx)
                    expanded.pop(idx)
                    fractionalsolutions.pop(idx)
                break
            elif childrennodes[idx][1] == node:
                expanded[idx][1] = 1
                if expanded[idx][0] == 1:
                    # pop the corresponding child node
                    childrennodes.pop(idx)
                    expanded.pop(idx)
                    fractionalsolutions.pop(idx)
                break

        # check cases
        if feasible and objective > BestObjective:
            # prune the node
            pass
        elif not feasible:
            # prune the node
            pass
        elif checkintegral(solution) is False:
            # the solution is not integer
            # need to branch

            # now we choose branching randomly
            # we choose branching based on how fraction variables are
            index = np.argmax(np.abs(np.round(solution) - solution))
            print(index)

            # add the corresponding constraints and create nodes
            lower_constraint = np.zeros(A.shape[1])
            lower_constraint[index] = 1.0
            lower = np.floor(solution[index])
            Alower = np.vstack((A, lower_constraint))
            blower = np.append(b, lower)
            node1 = Node(Alower, blower, c, sense, VType, maximize, IPsolution)

            upper_constraint = np.zeros(A.shape[1])
            upper_constraint[index] = -1.0
            upper = -np.ceil(solution[index])
            Aupper = np.vstack((A, upper_constraint))
            bupper = np.append(b, upper)
            node2 = Node(Aupper, bupper, c, sense, VType, maximize, IPsolution)

            # add nodes to the queue
            nodelist.append(node1)
            nodelist.append(node2)

            # record the newly added child nodes and the fractional solution
            fractionalsolutions.append(objective)
            childrennodes.append([node1, node2])
            expanded.append([0, 0])

        elif checkintegral(solution) is True:
            # check if better than current best
            if objective <= BestObjective:
                BestSolution = solution
                BestObjective = objective
        else:
            raise NotImplementedError

        if len(fractionalsolutions) == 0:
            break

        print('obj', BestObjective, 'sol', BestSolution, 'num of remaining nodes', len(nodelist), 'check int',
              checkintegral(solution), 'feasible', feasible)
        print('lower bound', np.min(fractionalsolutions), 'len of fractional solutions', len(fractionalsolutions))
        print('lower bound set', fractionalsolutions)
        print('cut is feasible?', cutfeasible)

        # compute optimality gap (old way)
        # ratiogap = (np.min(fractionalsolutions) - objslp) / (objsip[idxinstance] - objslp)
        # print('objective ratio gap', ratiogap)
        # optimalitygap.append(ratiogap)

        # increment time count
        timecount += 1
        if BestSolution is not None:
            # compute the ratio
            gap_now = BestObjective - np.min(fractionalsolutions)
            base_gap = BestObjective - initial_lp_objective
            ratio = gap_now / base_gap
            # print('success statistics', ratio)
            # ratios.append(ratio)
            if ratio <= RATIO_THRESHOLD:
                break

        time.sleep(.2)

        if timecount >= TIMELIMIT:
            break
