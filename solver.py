import copy
import json
import math
import sys
from decimal import *


class Solver:
    def __init__(self, A, b, c, x, e=None, type=0):
        # "c": [-4, 1], "x": [1, 1], "A": [[7, -2], [0, 1], [2, -2]], "b": [14, 3, 3], "e": [0, 1, 0], "type": 0

        if e is None:
            e = [0 for i in range(len(A))]
        # j = json.loads(jsonString)

        self.A = A
        self.b = b
        self.c = c
        self.x = x
        self.e = e
        self.type = type
        self.success = True

    def setupTablue(self):

        self.tablue = []

        noEqualityConst = self.e.count(2)
        noGreaterThanConst = self.e.count(1)
        self.artificial = [0 for i in range(noGreaterThanConst + noEqualityConst)]
        self.artificialObjectiveValue = [
            1000000000 for i in range(noGreaterThanConst + noEqualityConst)
        ]

        self.slack = [0 for i in range(len(self.A) - noEqualityConst)]
        self.tablue.append(self.c[:])
        self.tablue[0].extend(self.slack)
        self.tablue[0].extend(self.artificialObjectiveValue)
        self.tablue[0].append(0)

        self.x.extend([2 for i in range(len(self.A) - noEqualityConst)])

        e1 = 0
        e2 = 0
        for i in range(len(self.A)):
            self.tablue.append(self.A[i][:])
            self.tablue[-1].extend(self.slack)
            self.tablue[-1].extend(self.artificial)
            self.tablue[-1].append(0)
            try:
                if self.e[i] == 0:
                    self.tablue[-1][len(self.A[0]) + i - e2] = 1
                elif self.e[i] == 1:
                    self.tablue[-1][len(self.A[0]) + i - e2] = -1
                    self.tablue[-1][len(self.A[0]) + len(self.slack) + e1] = 1
                    e1 += 1
                elif self.e[i] == 2:
                    self.tablue[-1][len(self.A[0]) + len(self.slack) + e1] = 1
                    e1 += 1
                    e2 += 1
            except:
                self.tablue[-1][len(self.A[0]) + i] = 1
            self.tablue[-1][-1] = self.b[len(self.tablue) - 2]

        if len(self.A) - noEqualityConst > 0:
            matrix = self.getTablue()
            for row in matrix:
                row[len(self.x) : -1] = []
            self.steps.append(
                {
                    "step": self.step,
                    "text": "Added slackvariables",
                    "s": [
                        ("s%i" % i) for i in range(1, len(self.A) - noEqualityConst + 1)
                    ],
                    "type": 0,
                    "matrix": matrix,
                }
            )
            self.step += 1

        self.x.extend([3 for i in range(len(self.artificial))])

        if noGreaterThanConst + noEqualityConst > 0:
            self.steps.append(
                {
                    "step": self.step,
                    "text": "Added artificial variables",
                    "s": [
                        ("a%i" % i)
                        for i in range(1, noGreaterThanConst + noEqualityConst + 1)
                    ],
                    "type": 0,
                    "matrix": self.getTablue(),
                }
            )
            self.step += 1

        if noGreaterThanConst + noEqualityConst > 0:
            self.updateObjectiveFunction()
            self.steps.append(
                {
                    "step": self.step,
                    "text": "Updated objective row",
                    "s": [
                        "<p class='text-danger'> z always calulated outside pivot operations</p>"
                    ],
                    "type": 0,
                    "matrix": self.getTablue(),
                }
            )
            self.step += 1

    def updateObjectiveFunction(self):
        for row in range(1, len(self.tablue)):
            start = len(self.A[0]) + len(self.slack)
            if self.tablue[row][start:-1].count(1) > 0:
                self.tablue[0] = [
                    self.tablue[0][i] - self.tablue[row][i] * 1000000000
                    for i in range(len(self.tablue[0]))
                ]
                self.tablue[0][-1] = 0

    def solve(self):

        self.success = True

        self.steps = []
        self.step = 1
        self.setupTablue()

        q = 1
        solved = False

        while not solved:

            while (
                any(round(i, 5) < 0 for i in self.tablue[0][:-1])
                and q < 1000
                and self.success
            ):

                if self.getPrimalPivotIndexs() == False:
                    self.steps.append(
                        {
                            "step": self.step,
                            "text": "The problem is unbounded",
                            "s": [],
                            "type": 0,
                            "matrix": self.getTablue(),
                        }
                    )
                    self.success = False
                    continue

                pivotCol, pivotRow = self.getPrimalPivotIndexs()
                self.pivot(pivotRow, pivotCol)

                self.updateZ()

                self.steps.append(
                    {
                        "step": self.step,
                        "text": "Pivot",
                        "row": pivotRow,
                        "column": pivotCol + 1,
                        "type": 1,
                        "matrix": self.getTablue(),
                    }
                )
                self.step += 1

                q += 1

            self.getSolution()

            if not self.success:
                solved = True
            elif self.checkInfeasibility():
                solved = True
            elif self.isSolved():
                solved = True
            elif q >= 1000:
                solved = True
                self.success = False
                self.steps.append(
                    {
                        "step": self.step,
                        "text": "Failed to many steps",
                        "s": [],
                        "type": 0,
                        "matrix": self.getTablue(),
                    }
                )
            else:
                return self.addCuts()
            q += 1
            return []

    def getPrimalPivotIndexs(self):

        pivotCol = self.tablue[0][:-1].index(min(self.tablue[0][:-1]))

        t = []

        for i in range(1, len(self.tablue)):
            if (
                self.tablue[i][pivotCol] != 0
                and float(self.tablue[i][-1]) / self.tablue[i][pivotCol] > 0
                and self.tablue[i][-1] >= 0
            ):
                t.append(float(self.tablue[i][-1]) / self.tablue[i][pivotCol])
            else:
                t.append(1000000000)

        pivotRow = t.index(min(t)) + 1

        if sum(i >= 1000000000 for i in t) == len(t):
            return False
        else:
            return pivotCol, pivotRow

    def possiblePivot(self, pivotCol):
        if self.tablue[0][pivotCol] < 0:
            t = []
            for i in range(len(self.b)):
                if (
                    self.tablue[i + 1][pivotCol] != 0
                    and float(self.tablue[i + 1][-1]) / self.tablue[i + 1][pivotCol] > 0
                    and self.tablue[i][-1] >= 0
                ):
                    t.append(
                        float(self.tablue[i + 1][-1]) / self.tablue[i + 1][pivotCol]
                    )
                else:
                    t.append(1000000000)
            pivotRow = t.index(min(t)) + 1

            if sum(i >= 1000000000 for i in t) != len(t):
                return 1
        return 0

    def pivot(self, pivotRow, pivotCol):
        try:
            self.tablue[pivotRow] = [
                x / float(self.tablue[pivotRow][pivotCol])
                for x in self.tablue[pivotRow]
            ]

            for row in range(len(self.tablue)):
                if row != pivotRow and self.tablue[row][pivotCol] != 0:
                    subAmount = self.tablue[row][pivotCol]
                    self.tablue[row] = [
                        self.tablue[row][col]
                        - subAmount * float(self.tablue[pivotRow][col])
                        for col in range(len(self.tablue[row]))
                    ]

        except:
            self.success = False
            self.steps.append(
                {
                    "step": self.step,
                    "text": "Failed numbers got to small",
                    "s": [],
                    "type": 0,
                    "matrix": self.getTablue(),
                }
            )

    def addCuts(self):

        maxCuts = 10
        cuts = 0
        cut_options = []

        if len(self.tablue) >= len(self.A) * 10:
            self.success = False
            self.steps.append(
                {
                    "step": self.step,
                    "text": "Failed to many cuts",
                    "s": [],
                    "type": 0,
                    "matrix": self.getTablue(),
                }
            )
        else:
            for i in range(len(self.solution)):
                if cuts < maxCuts and len(self.tablue) < len(self.A) * 10:
                    if self.x[self.solution[i][1]] == 1:
                        if (
                            self.solution[i][2] >= 0
                            and not float(
                                round(Decimal(self.solution[i][0]), 4)
                            ).is_integer()
                        ):
                            if self.e[self.solution[i][2] - 1] == 1:
                                cut_options.append(self.addGomoryMinMIRCut(i))
                                cuts += 1
                            else:
                                cut_options.append(self.addGomoryMaxMIRCut(i))
                                cuts += 1
            return cut_options
            # self.makeFeasable()

    def addGomoryMaxMIRCut(self, i):

        cut = self.tablue[self.solution[i][2]][:]
        cut[-1] = -float(cut[-1] - math.floor(cut[-1]))

        for j in range(len(cut) - 1):
            if j < len(self.x) and self.x[j] == 1:
                cut[j] = -float(cut[j] - math.floor(cut[j]))
                if cut[j] <= cut[-1]:
                    cut[j] = cut[-1] * (1 + cut[j]) / (1 + cut[-1])
            else:
                cut[j] = -cut[j]
                if cut[j] >= 0:
                    cut[j] = cut[j] * cut[-1] / (1 + cut[-1])

        cut.insert(-1, 1)
        self.x.append(2)

        for i in self.tablue:
            i.insert(-1, 0)

        self.tablue.append(cut)
        self.e.append(0)

        cutText = self.cutToText(cut)
        self.steps.append(
            {
                "step": self.step,
                "text": "Added max cut",
                "cut": cutText,
                "type": 2,
                "matrix": self.getTablue(),
            }
        )
        self.step += 1
        return cut

    def addGomoryMinMIRCut(self, i):

        cut = self.tablue[self.solution[i][2]][:]
        cut[-1] = float(cut[-1] - math.ceil(cut[-1]))

        for j in range(len(cut) - 1):
            if j < len(self.x) and self.x[j] == 1:
                cut[j] = -float(math.ceil(cut[j]) - cut[j])
                if cut[j] <= cut[-1]:
                    cut[j] = -cut[-1] * (1 + cut[j]) / (1 + cut[-1])
            else:
                cut[j] = cut[j]
                if cut[j] >= 0:
                    cut[j] = cut[j] * cut[-1] / (1 + cut[-1])

        cut.insert(-1, 1)
        self.x.append(2)

        for i in range(len(self.tablue)):
            self.tablue[i].insert(-1, 0)

        self.tablue.append(cut)
        self.e.append(1)

        cutText = self.cutToText(cut)
        self.steps.append(
            {
                "step": self.step,
                "text": "Added min cut",
                "cut": cutText,
                "type": 2,
                "matrix": self.getTablue(),
            }
        )
        self.step += 1
        return cut

    def makeFeasable(self):

        unfeasible = True
        k = 0

        while unfeasible and self.success:

            k += 1
            minimum = 0
            pivotRow = 0

            for i in range(1, len(self.tablue)):
                if minimum > round(self.tablue[i][-1], 7):
                    minimum = self.tablue[i][-1]
                    pivotRow = i

            if pivotRow == 0:
                unfeasible = False
            elif k >= 50:
                unfeasible = False
                self.success = False
                self.steps.append(
                    {
                        "step": self.step,
                        "text": "Failed on dual pivot",
                        "s": [],
                        "type": 0,
                        "matrix": self.getTablue(),
                    }
                )
            else:
                pivotCol = self.getDualPivotCol(pivotRow)
                self.pivot(pivotRow, pivotCol)

                self.updateZ()
                self.steps.append(
                    {
                        "step": self.step,
                        "text": "Pivot, dual",
                        "row": pivotRow,
                        "column": pivotCol + 1,
                        "type": 1,
                        "matrix": self.getTablue(),
                    }
                )
                self.step += 1

    def getDualPivotCol(self, pivotRow):
        t = []

        for i in range(len(self.tablue[0]) - 1):

            if (
                self.tablue[pivotRow][i] != 0
                and float(-self.tablue[0][i]) / self.tablue[pivotRow][i] > 0
            ):
                t.append(float(-self.tablue[0][i]) / self.tablue[pivotRow][i])
            else:
                t.append(1000000000)

        return t.index(min(t))

    def getSolution(self):

        self.solution = []

        for j in range(len(self.tablue[0]) - 1):

            i = 0
            # Ok is to protect against invalid state where
            # there are two 1 values in same column, which is invalid
            ok = True
            x = -1

            while ok and i <= len(self.tablue) - 1:
                if self.tablue[i][j] == 1 and x == -1:
                    x = i
                elif self.tablue[i][j] != 0:
                    ok = False
                i += 1

            if x > 0 and ok:
                self.solution.append((self.tablue[x][-1], j, x))
            else:
                self.solution.append((0, j, -1))

    def isSolved(self):
        Solved = True
        j = 0
        while Solved and j < len(self.tablue[0]) - 1:
            value = self.solution[j][0]
            valueUp = math.ceil(value * 10000) / 10000
            valueDown = math.floor(value * 10000) / 10000
            if (
                not float(valueUp).is_integer()
                and not float(valueDown).is_integer()
                and self.x[j] == 1
            ):
                Solved = False
            j += 1

        return Solved

    def checkInfeasibility(self):
        Infeasible = False
        j = 0
        while not Infeasible and j < len(self.tablue[0]) - 1:
            value = self.solution[j][0]
            if self.x[j] == 3 and value != 0:
                Infeasible = True
                self.success = False
                self.steps.append(
                    {
                        "step": self.step,
                        "text": "The problem is infeasible",
                        "s": [],
                        "type": 0,
                        "matrix": self.getTablue(),
                    }
                )
            j += 1

        return Infeasible

    def cutToText(self, cut):

        if self.x[-1] == 2:
            cutText = "s%i " % (self.x.count(2))
        elif self.x[-1] == 3:
            cutText = "a%i " % (self.x.count(3))

        for i in range(len(cut)):
            value = cut[i]

            if cut[i] != 0 and (i < len(cut) - 2):

                if cut[i] > 0:
                    cutText = cutText + "+ "
                elif cut[i] < 0:
                    cutText = cutText + "- "
                    value = value * -1

                if float(value).is_integer():
                    if not value == 1:
                        cutText = cutText + "%i" % value
                else:
                    cutText = cutText + str(round(value, 3))

                if i < len(self.c):
                    cutText = cutText + " x%i " % (i + 1)
                else:
                    if self.x[i] == 2:
                        cutText = cutText + "s%i " % (self.x[: i + 1].count(2))
                    elif self.x[i] == 3:
                        cutText = cutText + "a%i " % (self.x[: i + 1].count(3))

            elif len(cutText) > 0 and i == len(cut) - 1:

                cutText = cutText + "= "

                if value < 0:
                    cutText = cutText + "- "
                    value = value * -1

                if float(cut[i]).is_integer():
                    cutText = cutText + ("%i" % value)
                else:
                    cutText = cutText + str(round(value, 3))

        return cutText

    def updateZ(self):
        self.getSolution()
        Z = 0
        j = 0
        for i in self.solution:
            if j < len(self.c):
                Z += i[0] * -self.c[j]
            j += 1

        if self.type == 1:
            self.tablue[0][-1] = -Z
        else:
            self.tablue[0][-1] = Z

    def getTablue(self):
        matrix = [[]]
        j = 0
        for i in self.x:
            j += 1
            if i < 2:
                matrix[0].append("x%i" % (self.x[0:j].count(1) + self.x[0:j].count(0)))
            elif i == 2:
                matrix[0].append("s%i" % (self.x[0:j].count(i)))
            elif i == 3:
                matrix[0].append("a%i" % (self.x[0:j].count(i)))

        matrix[0].append("z, b")
        matrix.extend(copy.deepcopy(self.tablue))

        return matrix

    def printTablue(self):
        tablue = []
        for i in self.tablue:
            tablue.append(str(i))
        return tablue

    def printSolution(self):
        Z = 0
        j = 0
        text = ""

        for i in self.solution:
            text = text + str("x%i = %.2f " % (i[1] + 1, i[0]))
            if j < len(self.c):
                Z += i[0] * -self.c[j]
            j += 1

        return text + str("Z  = %.2f" % Z)

    def getJSONsolution(self):

        solution = {"x": [], "s": [], "a": []}

        Z = 0
        j = 0

        for i in self.solution:
            if self.x[i[1]] < 2:
                solution["x"].append(float("%.2f" % i[0]))
            elif self.x[i[1]] == 2:
                solution["s"].append(float("%.2f" % i[0]))
            else:
                solution["a"].append(float("%.2f" % i[0]))
            if j < len(self.c):
                Z += i[0] * -self.c[j]
            j += 1

        solution["z"] = float("%.2f" % Z)
        if self.type == 1:
            solution["z"] = solution["z"] * -1
        solution["solved"] = self.success
        solution["steps"] = self.steps
        jsonSolution = {"solution": solution}

        return json.dumps(jsonSolution, separators=(",", ":"))


# if __name__ == "__main__":
#     # os.system('clear')
#
#     try:
#         jsonString = sys.argv[1]
#         j = json.loads(jsonString)
#     except:
#         jsonString = '{"c":[-4,1],"x":[1,1],"A":[[7,-2],[0,1],[2,-2]],"b":[14,3,3],"e":[0,1,0],"type":0}'
#
#     y = Solver(jsonString)
#     y.solve()
#     x = y.getJSONsolution()
#     print(x)
