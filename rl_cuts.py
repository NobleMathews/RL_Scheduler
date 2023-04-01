import json
import logging

import ray
import psutil
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random

from torch_policy import Policy
from torch_solver import compute_state, roundmarrays, gurobi_solve, gurobi_int_solve


class GurobiOriginalEnv(object):
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

    def max_gap(self):
        """
        this method computes the max achivable gap
        """
        # preprocessing
        A, b, c, sense, vtype = self.A0.copy(), self.b0.copy(), self.c0.copy(), self.sense0, self.VType0
        m, n = A.shape
        assert m == b.size and n == c.size
        # compute gaps
        objint, solution_int = gurobi_int_solve(A, b, c, sense, vtype, maximize=self.maximize)
        objlp, solution_lp, _, _ = gurobi_solve(A, b, c, sense, maximize=self.maximize)
        return np.abs(objint - objlp), solution_int, solution_lp


class TimelimitWrapper(object):
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
        return obs, reward, done, info


class MultipleEnvs(object):
    def __init__(self, envs):
        self.envs = envs
        self.all_indices = list(range(len(self.envs)))
        self.available_indices = list(range(len(self.envs)))
        self.env_index = None
        self.env_now = None

    def reset(self):
        self.env_index = np.random.choice(self.available_indices)
        self.available_indices.remove(self.env_index)
        if len(self.available_indices) == 0:
            self.available_indices = self.all_indices[:]

        self.env_now = self.envs[self.env_index]
        return self.env_now.reset()

    def step(self, action, fake=False):
        assert self.env_now is not None
        return self.env_now.step(action, fake)


def make_multiple_env(load_dir, idx_list, timelimit, reward_type):
    envs = []
    for idx in idx_list:
        # Currently multiple env will only contain the kondili example
        assert idx == 0
        # print('loading training instances, dir {} idx {}'.format(load_dir, idx))
        # A = np.load('{}/A_{}.npy'.format(load_dir, idx))
        # b = np.load('{}/b_{}.npy'.format(load_dir, idx))
        # c = np.load('{}/c_{}.npy'.format(load_dir, idx))
        with open(load_dir, "r") as inputfile:
            input_data = json.load(inputfile)

        A0 = np.asarray(input_data["A0"])
        b0 = np.asarray(input_data["b0"])
        c0 = np.asarray(input_data["c0"])
        sense = input_data["sense"]
        VType = input_data["VType"]
        maximize = input_data["maximize"]

        # A0 = np.asarray([[1]*6 + [0]*18, [0]*6+[1]*6+[0]*12, [0]*12+[1]*6+[0]*6, [0]*18+[1]*6,
        #                  [15, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0, 70, 0, 0, 0, 0, 0],
        #                  [0, 15, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0, 70, 0, 0, 0, 0],
        #                  [0, 0, 15, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0, 70, 0, 0, 0],
        #                  [0, 0, 0, 15, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0, 70, 0, 0],
        #                  [0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0, 70, 0],
        #                  [0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0, 70]])
        # b0 = np.asarray([10, 5, 7, 4, 45, 120, 165, 214, 64, 93])
        # c0 = np.asarray([23, 12, 34, 25, 27, 16, 29, 24, 43, 35, 28, 19, 43, 31, 52, 36, 30, 21, 54, 36, 54, 46, 34, 27])
        # sense = ["<"]*4+[">"]*6
        # VType = ["Z"]*24
        # maximize = False

        # A0 = np.asarray([[3, 2], [0, 1]])
        # b0 = np.asarray([5, 2])
        # c0 = np.asarray([1, 1])
        # sense = ["<", "<"]
        # VType = ["Z", "C"]
        # maximize = True

        env = TimelimitWrapper(
            GurobiOriginalEnv(A0, b0, c0, sense, VType, maximize, solution=None, reward_type=reward_type), timelimit)
        envs.append(env)
    env_final = MultipleEnvs(envs)
    return env_final, A0.shape[1]


try_config = {
    "load_dir": 'instances/kondili.json',
    # this is the location of the randomly generated instances (you may specify a different directory)
    "idx_list": list(range(1)),  # take the first n instances from the directory
    "timelimit": 10000,  # the maximum horizon length
    "reward_type": 'obj'  # DO NOT CHANGE reward_type
}


def get_option_reward(i, env):
    new_state, r, d, _ = env.step([i], True)
    return r, d


def normalization(A, b, E, d):
    # print(A)
    # print(E)
    all_coeff = np.concatenate((A, E), axis=0)
    all_constraint = np.concatenate((b, d))
    max_1, max_2 = np.max(all_coeff), np.max(all_constraint)
    min_1, min_2 = np.min(all_coeff), np.min(all_constraint)
    norm_A = (A - min_1) / (max_1 - min_1)
    norm_E = (E - min_1) / (max_1 - min_1)
    norm_b = (b - min_2) / (max_2 - min_2)

    norm_d = (d - min_2) / (max_2 - min_2)

    return norm_A, norm_b, norm_E, norm_d


def discounted_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0, len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)


if __name__ == "__main__":
    session_type = 2

    training = True
    explore = True
    PATH = "models/try1.pt"
    # PATH = "models/easy_config_best_model_3.pt"
    # PATH = "models/hard_config_best_model3.pt"
    # input_dim = 12

    # create env
    env, input_dim = make_multiple_env(**try_config)
    input_dim = input_dim + 1
    lr = 1e-2
    # initialize networks
    lstm_hidden = 10
    dense_hidden = 64

    explore_rate = 1.0
    min_explore_rate = 0.05
    max_explore_rate = 0.1
    explore_decay_rate = 0.01
    best_rew = 0

    # if training:
    #     actor = Policy(input_size=input_dim, hidden_size=lstm_hidden, hidden_size2=dense_hidden, lr=lr)
    # else:
    #     actor = torch.load(PATH)

    sigma = 0.2
    gamma = 0.99  # discount
    rrecord = []
    for e in range(1000):
        print(f"Starting Episode {e}")
        # gym loop
        # To keep a record of states actions and reward for each episode
        obss_constraint = []  # states
        obss_cuts = []
        acts = []
        rews = []

        s, cut_rows = env.reset()  # samples a random instance every time env.reset() is called
        d = 1
        remaining_vars = d
        repisode = 0
        og_repisode = 0
        i = 0
        while d != 0:
            i = i + 1
            A, b, c0, cuts_a, cuts_b, x_LP = s

            # print(A)
            # print(b)
            # print(c0)
            # print(cuts_a)
            # print(cuts_b)
            # print(x_LP)

            # normalization
            A, b, cuts_a, cuts_b = normalization(A, b, cuts_a, cuts_b)

            # concatenate [a, b] [e, d]
            curr_constraints = np.concatenate((A, b[:, None]), axis=1)
            available_cuts = np.concatenate((cuts_a, cuts_b[:, None]), axis=1)

            # # compute probability distribution
            # # torch.onnx.export(actor.model, (torch.FloatTensor(curr_constraints), torch.FloatTensor(available_cuts)), "lstm_model.onnx")
            # prob = actor.compute_prob(curr_constraints, available_cuts)
            # # [cut_rows, :]
            # prob = prob / np.sum(prob)
            #
            # explore_rate = min_explore_rate + \
            #                (max_explore_rate - min_explore_rate) * np.exp(-explore_decay_rate * (e))

            # option_rewards = []
            # # if training or explore:
            # #     for i in range(s[-2].size):
            # #         new_state, r, d, _ = env.step([i], True)
            # #         option_rewards.append(r)
            # # save a lookup table for option_rewards so it is not recomputed every run during training.
            # # Create a dictionary with key as numpy array curr_constraints and value as option_rewards
            # # Save the dictionary to a file and load it initially
            # if os.path.exists('option_rewards_dict.json'):
            #     with open('option_rewards_dict.json', 'r') as f:
            #         option_rewards_dict = json.load(f)
            # else:
            #     option_rewards_dict = {}
            # if str(curr_constraints.tobytes()) in option_rewards_dict:
            #     option_rewards = option_rewards_dict[str(curr_constraints.tobytes())]
            # else:
            # if training or explore:
            # parallelize this loop with as many workers as cpu cores
            # for i in range(s[-2].size):
            #     new_state, r, d, _ = env.step([i], True)
            #     option_rewards.append(r)

            # Initialize the ray runtime
            # ray.init()

            # Get the number of CPU cores available on the machine
            num_cpus = psutil.cpu_count(logical=False)
            n_cuts = 1
            # Create a list of tasks to run in parallel
            if n_cuts >= s[-2].size:
                a = range(s[-2].size)
            else:
                # Lexicographic
                if session_type == 0:
                    a = 0
                # Max violation
                elif session_type == 1:
                    a = np.argmax(np.abs(s[-2]-np.round(s[-2])))
                # Max normalized violation
                elif session_type == 2:
                    row_norms = np.apply_along_axis(np.linalg.norm, 1, s[-3])
                    a = np.argmax(np.abs(s[-2] - np.round(s[-2]))/row_norms)
                # Random cut selection
                else:
                    a = np.random.choice(s[-2].size, n_cuts, replace=False)
                try:
                    iter(a)
                except TypeError:
                    a = [a]
                else:
                    pass
            index = 0
            # #     .remote
            # opt_rewards = [get_option_reward(i, env) for i in a]
            #
            # # Use ray to fetch the results of the tasks in parallel
            # # option_rewards = ray.get(tasks)
            #
            # # Shut down the ray runtime
            # # ray.shutdown()
            #
            # # option_rewards_dict[str(curr_constraints.tobytes())] = option_rewards
            #
            # # normalize option rewards
            #
            #
            #
            # # option_rewards = np.array(tasks)
            # # option_rewards = (option_rewards - np.mean(option_rewards)) / np.std(option_rewards)
            #
            #
            #
            # # option_penalty = 100 * option_rewards
            #
            # # # get index of the top 5% of options with most positive rewards
            # # top_10_percent = int(0.05 * len(option_rewards))
            # # if top_10_percent == 0:
            # #     top_10_percent = 1
            # # top_10_percent_indices = np.argpartition(option_rewards, -top_10_percent)[-top_10_percent:]
            #
            # # # epsilon greedy for exploration
            # # if training and explore:
            # #     random_num = random.uniform(0, 1)
            # #     if random_num <= explore_rate:
            # #         print("manually directed picking")
            # #         a = np.random.randint(0, s[-2].size, 1)
            # #         # randomly choose between the top 10% of options
            # #         # a = np.random.choice(top_10_percent_indices, 1)
            # #     else:
            # #         print("agent probability based picking")
            # #         # a = [np.argmax(prob)]
            # #         # a = [np.random.choice(s[-2].size, p=prob.flatten())]
            # #         # pick all cuts with probability > 0.4
            # #         a = np.where(prob > 0.4)[0]
            # #         if not len(a):
            # #             a = [np.argmax(prob)]
            # # else:
            # #     # for testing case, only sample action
            # #     a = [np.random.choice(s[-2].size, p=prob.flatten())]
            # # get index of largest element in option_rewards
            # rewards, costs = zip(*opt_rewards)
            # rewards = np.array(rewards)
            # costs = np.array(costs)
            #
            # # min_cost = np.inf
            # # max_reward = -np.inf
            # # index = None
            # #
            # # for i, (reward, cost) in enumerate(zip(rewards, costs)):
            # #     if cost < min_cost and reward > max_reward:
            # #         min_cost = cost
            # #         max_reward = reward
            # #         index = i
            #
            # # Normalize the rewards and costs
            # # Normalize the rewards and costs between 0 and 1
            # rewards_min = np.min(rewards)
            # rewards_range = np.max(rewards) - rewards_min
            # if rewards_range == 0:
            #     normalized_rewards = np.zeros_like(rewards)
            # else:
            #     normalized_rewards = (rewards - rewards_min) / rewards_range
            #
            # costs_min = np.min(costs)
            # costs_range = np.max(costs) - costs_min
            # if costs_range == 0:
            #     normalized_costs = np.zeros_like(costs)
            # else:
            #     normalized_costs = (costs - costs_min) / costs_range
            #
            # # Calculate the reward-to-cost ratio
            # ratios = normalized_rewards / normalized_costs
            #
            # # Get the index of the highest reward-to-cost ratio
            # index = np.nanargmax(ratios)
            #
            # a = [a[index]]

            # find the maximum option_reward

            new_state, r, d, _ = env.step(list(a))
            if d != 0:
                remaining_vars = d
            print("Action taken: ", index, "Reward: ", r, "Remaining vars: ", d)
            # print('episode', e, 'step', t, 'reward', r, 'action space size', new_state[-1].size, 'action', a)
            # a = np.random.randint(0, s[-2].size,
            #                       1)  # s[-1].size shows the number of actions, i.e., cuts available at state s
            A, b, c0, cuts_a, cuts_b, x_LP = new_state

            og_r = r
            # if option_rewards[a] < 0:
            #     # penalize choosing a cut that makes the problem bigger
            #     r = r + option_rewards[a] * 10
            # else:
            #     r = r + option_rewards[a]

            obss_constraint.append(curr_constraints)
            obss_cuts.append(available_cuts)
            acts.append(a)
            rews.append(r)
            s = new_state
            if r < 1e-2:
                repisode += -10
            else:
                repisode += r
            # *100 - d
            og_repisode += og_r

            if i % 10 == 0:
                # save numpy array to file
                save_episode_np = np.concatenate((A, b[:, None]), axis=1)[411:]
                if not os.path.isdir(f"new_session{session_type}_{i}"):
                    os.makedirs(f"new_session{session_type}_{i}")
                np.save(f"new_session{session_type}_{i}/ab_{e}.npy", save_episode_np)
                # append r to a file called reward_{i}.txt
                with open(f"new_session{session_type}_{i}/reward.txt", "a") as f:
                    f.write(str(e) + "\t" + str(og_repisode) + "\t" + str(remaining_vars) + "\n")

            # if repisode < 0:
            #     # penalize not solving the problem
            #     repisode = repisode - 1000
            #     d = True

        # with open('option_rewards_dict.json', 'w') as f:
        #     json.dump(option_rewards_dict, f)
        # record rewards and print out to track performance
        rrecord.append(np.sum(rews))
        # returns = discounted_rewards(rews, gamma)
        # # we only use one trajectory so only one-variate gaussian used here.
        # Js = returns + np.random.normal(0, 1, len(returns)) / sigma
        print("episode: ", e)
        print("sum reward: ", repisode)
        # print(x_LP)

        # # PG update and save best model so far
        # if training:
        #     if repisode >= best_rew:
        #         best_rew = repisode
        #         torch.save(actor, PATH)
        #
        #     loss = actor.train(obss_constraint, obss_cuts, acts, Js)
        #     print("Loss: ", loss)
