import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gurobipy as gb
from torchviz import make_dot

from torch_solver import compute_state


class LstmNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super(LstmNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size, hidden_size, bidirectional=bidirectional, batch_first=True
        )

    def forward(self, inp):
        _hidden = self.init_hidden()
        # inputs = torch.FloatTensor(inp).view(1, -1, self.input_size)
        inputs = torch.unsqueeze(torch.FloatTensor(inp), 0)
        output, _ = self.lstm(inputs)
        # output[-1] is same as last hidden state
        output = output[-1].reshape(-1, self.hidden_size)
        return output

    def init_hidden(self):
        return (
            torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size),
            torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size),
        )


class AttentionNetwork(nn.Module):
    # input_dim = 12
    # h1 lstm_hidden = 10
    # h2 dense_hidden = 64
    def __init__(self, input_size, n_embed, head_size):
        super(AttentionNetwork, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(n_embed)
        self.hidden_size2 = int(head_size)
        # constraint and cuts dimension
        # self.lstm_enc1 = nn.LSTM(1, input_size)
        # self.lstm_enc2 = nn.LSTM(1, input_size)

        self.lstm1 = LstmNetwork(input_size, n_embed)
        self.lstm2 = LstmNetwork(input_size, n_embed)

        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size2)
        self.linear2 = nn.Linear(self.hidden_size2, self.hidden_size2)
        self.tanh = nn.Tanh()

        # self.A_key = nn.Linear(n_embed, head_size, bias=False)
        # self.A_query = nn.Linear(n_embed, head_size, bias=False)
        # self.A_value = nn.Linear(n_embed, head_size, bias=False)
        #
        # self.D_key = nn.Linear(n_embed, head_size, bias=False)
        # self.D_query = nn.Linear(n_embed, head_size, bias=False)
        # self.D_value = nn.Linear(n_embed, head_size, bias=False)
        #
        # self.dropout = nn.Dropout()
        # # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # #
        # # self.dropout = nn.Dropout(dropout)

    def forward(self, constraints, cuts):
        constraints = torch.FloatTensor(constraints)
        cuts = torch.FloatTensor(cuts)

        # constraints = torch.stack([self.lstm_enc1(constr.view(1, -1, 1))[0][0][-1] for constr in constraints])
        # cuts = torch.stack([self.lstm_enc1(cut.view(1, -1, 1))[0][0][-1] for cut in cuts])

        # lstm
        A_embed = self.lstm1.forward(constraints)
        D_embed = self.lstm2.forward(cuts)

        # dense
        A = self.linear2(self.tanh(self.linear1(A_embed)))
        D = self.linear2(self.tanh(self.linear1(D_embed)))

        # A_T, A_C = A_embed.shape
        # # A_tril = torch.tril(torch.ones(A_T, A_T))
        # k = self.A_key(A_embed)  # (B,T,C)
        # q = self.A_query(A_embed)  # (B,T,C)
        # # compute attention scores ("affinities")
        # A_wei = q @ k.transpose(-2, -1) * A_C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        # # A_wei = A_wei.masked_fill(A_tril[:A_T, :A_T] == 0, float('-inf'))  # (B, T, T)
        # A_wei = F.softmax(A_wei, dim=-1)  # (B, T, T)
        # A_wei = self.dropout(A_wei)
        # # perform the weighted aggregation of the values
        # A_v = self.A_value(A_embed)  # (B,T,C)
        # A_out = A_wei @ A_v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        #
        # D_T, D_C = D_embed.shape
        # k = self.D_key(D_embed)  # (B,T,C)
        # q = self.D_query(D_embed)  # (B,T,C)
        # # compute attention scores ("affinities")
        # D_wei = q @ k.transpose(-2, -1) * D_C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        # # D_tril = torch.tril(torch.ones(D_T, D_T))
        # # D_wei = D_wei.masked_fill(D_tril[:D_T, :D_T] == 0, float('-inf'))  # (B, T, T)
        # D_wei = F.softmax(D_wei, dim=-1)  # (B, T, T)
        # D_wei = self.dropout(D_wei)
        # # perform the weighted aggregation of the values
        # D_v = self.D_value(D_embed)  # (B,T,C)
        # D_out = D_wei @ D_v  # (B, T, T) @ (B, T, C) -> (B, T, C)

        # attention
        # noinspection PyArgumentList
        logits = torch.sum(torch.mm(D, A.T), axis=1)

        return logits


class Policy(object):
    def __init__(self, input_size, hidden_size, hidden_size2, lr):
        self.model = AttentionNetwork(input_size, hidden_size, hidden_size2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def compute_prob(self, constraints, cuts):
        constraints = torch.FloatTensor(constraints)
        cuts = torch.FloatTensor(cuts)
        y = self.model(constraints, cuts)
        # make_dot(y.mean(), params=dict(self.model.named_parameters())).render("attached", format="png")
        prob = torch.nn.functional.softmax(y, dim=-1)
        return prob.cpu().data.numpy()

    @staticmethod
    def _to_one_hot(y, num_classes):
        """
        convert an integer vector y into one-hot representation
        """
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        return zeros.scatter(scatter_dim, y_tensor, 1)

    def train(self, constraints, cuts, actions, Qs):
        """
        states: numpy array (states)
        actions: numpy array (actions)
        Qs: numpy array (Q values)
        """
        actions = torch.LongTensor(actions)
        Qs = torch.FloatTensor(Qs)

        total_loss = 0
        # for a bunch of constraints and cuts, need to go one by one
        for i in range(len(constraints)):
            curr_constraints = constraints[i]
            curr_cuts = cuts[i]
            curr_action = actions[i]
            # COMPUTE probability vector pi(s) for all s in states
            logits = self.model(curr_constraints, curr_cuts)
            prob = torch.nn.functional.softmax(logits, dim=-1)
            # Compute probaility pi(s,a) for all s,a
            action_onehot = self._to_one_hot(curr_action, curr_cuts.shape[0])
            # noinspection PyArgumentList
            prob_selected = torch.sum(prob * action_onehot, axis=-1)

            # FOR ROBUSTNESS
            prob_selected += 1e-8
            loss = -torch.mean(Qs[i] * torch.log(prob_selected))
            # BACKWARD PASS
            self.optimizer.zero_grad()
            loss.backward()
            # UPDATE
            self.optimizer.step()
            total_loss += loss.detach().cpu().data.numpy()

        return total_loss


def normalized(A, b, E, d):
    all_coeff = np.concatenate((A, E), axis=0)
    all_constraint = np.concatenate((b, d))
    max_1, max_2 = np.max(all_coeff), np.max(all_constraint)
    min_1, min_2 = np.min(all_coeff), np.min(all_constraint)
    norm_A = (A - min_1) / (max_1 - min_1)
    norm_E = (E - min_1) / (max_1 - min_1)
    norm_b = (b - min_2) / (max_2 - min_2)

    norm_d = (d - min_2) / (max_2 - min_2)

    return norm_A, norm_b, norm_E, norm_d


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    lr = 1e-2
    # initialize networks
    input_dim = 6
    lstm_hidden = 10
    dense_hidden = 64
    actor = Policy(input_size=input_dim, hidden_size=lstm_hidden, hidden_size2=dense_hidden, lr=lr)
    # min {cx | Ax <= b, x >= 0, x integer}
    # (411, 323)

    gen_problem_files = True
    instance = "instances/kondili.json"

    if gen_problem_files:
        model = gb.read("model.lp")
        sense = model.getAttr("Sense", model.getConstrs())
        VType = model.getAttr("VType", model.getVars())
        VarName = model.getAttr("VarName", model.getVars())

        VNameType = [f"{var}_{typ}" for var, typ in zip(VarName, VType)]
        index = model.getAttr("ConstrName", model.getConstrs())

        A0 = model.getA().toarray()
        # * factor[:, None]
        b0 = np.asarray(model.getAttr("RHS", model.getConstrs())) \
            # * factor
        c0 = np.asarray(model.getAttr("Obj", model.getVars()))
        maximize = True

        # df = pd.DataFrame(A0, index=index, columns=VNameType)
        # df['B0'] = b0
        # df.loc["c0"] = np.append(c0, 0)
        # df.to_html('df.html', index=True, header=True)

        # load_dir = "instances/train_100_n60_m60"
        # idx = 0
        # A0 = np.load('{}/A_{}.npy'.format(load_dir, idx))
        # b0 = np.load('{}/b_{}.npy'.format(load_dir, idx))
        # c0 = np.load('{}/c_{}.npy'.format(load_dir, idx))
        # sense = None

        # A0 = np.array([[7, -2], [0, 1], [2, -2]])
        # b0 = np.array([14, 3, 3])
        # c0 = np.array([4, -1])
        # sense = ["<", "<", "<"]
        # VType = ["Z", "C"]
        # maximize = True

        model.optimize()
        # obtain results
        solution = []
        # basis_index = []
        # RC = []
        X = model.getVars()
        for i in X:
            solution.append(i.X)

        data_format = {
            "A0": A0, "b0": b0, "c0": c0, "sense": sense, "VType": VType, "maximize": maximize, "solution": solution,
        }

        with open(instance, "w") as outputfile:
            json.dump(data_format, outputfile, cls=NumpyEncoder)
        sys.exit()
    with open(instance, "r") as inputfile:
        input_data = json.load(inputfile)

    A0 = np.asarray(input_data["A0"])
    b0 = np.asarray(input_data["b0"])
    c0 = np.asarray(input_data["c0"])
    sense = input_data["sense"]
    VType = input_data["VType"]
    maximize = input_data["maximize"]

    A, b, cuts_a, cuts_b, done, oldobj, A_embed, tab, _ = compute_state(A0, b0, c0, sense, VType, maximize=maximize)

    A, b, cuts_a, cuts_b = normalized(A, b, cuts_a, cuts_b)

    # concatenate [a, b] [e, d]
    curr_constraints = np.concatenate((A, b[:, None]), axis=1)
    available_cuts = np.concatenate((cuts_a, cuts_b[:, None]), axis=1)

    # compute probability distribution
    prob = actor.compute_prob(curr_constraints, available_cuts)
    prob /= np.sum(prob)
    # print(prob)
