from builtins import *

import random
import sys
import os
import math
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from functools import partial

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from .base_tabular_learner import Agent, StationaryAgent
import importlib
import maci.utils as utils
from copy import deepcopy


class BaseQAgent(Agent):
    def __init__(self, name, id_, action_num, env, alpha_decay_steps=10000., alpha=0.01, gamma=0.95, episilon=0.1, verbose=True, **kwargs):
        super().__init__(name, id_, action_num, env, **kwargs)
        self.episilon = episilon
        self.alpha_decay_steps = alpha_decay_steps
        self.gamma = gamma
        self.alpha = alpha
        self.epoch = 0
        self.Q = None
        self.pi = defaultdict(partial(np.random.dirichlet, [1.0] * self.action_num))
        self.record = defaultdict(list)
        self.verbose = verbose
        self.pi_history = [deepcopy(self.pi)]

    def done(self, env):
        if self.verbose:
            utils.pv('self.full_name(game)')
            utils.pv('self.Q')
            utils.pv('self.pi')

        numplots = env.numplots if env.numplots >= 0 else len(self.record)
        for s, record in sorted(
                self.record.items(), key=lambda x: -len(x[1]))[:numplots]:
            self.plot_record(s, record, env)
        self.record.clear()

    # learning rate decay
    def step_decay(self):
        # drop = 0.5
        # epochs_drop = 10000
        # decay_alpha = self.alpha * math.pow(drop, math.floor((1 + self.epoch) / epochs_drop))
        # return 1 / (1 / self.alpha + self.epoch * 1e-4)
        return self.alpha_decay_steps / (self.alpha_decay_steps + self.epoch)
        # return decay_alpha
    # def alpha(self, t):
    #     return self.alpha_decay_steps / (self.alpha_decay_steps + t)

    def act(self, s, exploration, game):
        if exploration and random.random() < self.episilon:
            return random.randint(0, self.action_num - 1)
        else:
            if self.verbose:
                for s in self.Q.keys():
                    print('{}--------------'.format(self.id_))
                    print('Q of agent {}: state {}: {}'.format(self.id_, s, str(self.Q[s])))
                    # print('QAof agent {}: state {}: {}'.format(self.id_, s, str(self.Q_A[s])))
                    # self.Q_A
                    print('pi of agent {}: state {}: {}'.format(self.id_, s, self.pi[s]))
                    # print('pi of opponent agent {}: state{}: {}'.format(self.id_, s, self.opponent_best_pi[s]))
                    print('{}--------------'.format(self.id_))
            # print()
            return StationaryAgent.sample(self.pi[s])

    @abstractmethod
    def update(self, s, a, o, r, s2, env, done=False):
        pass

    @abstractmethod
    def update_policy(self, s, a, env):
        pass

    def plot_record(self, s, record, env):
        os.makedirs('policy/', exist_ok=True)
        fig = plt.figure(figsize=(18, 10))
        n = self.action_num
        for a in range(n):
            plt.subplot(n, 1, a + 1)
            plt.tight_layout()
            plt.gca().set_ylim([-0.05, 1.05])
            plt.gca().set_xlim([1.0, env.t + 1.0])
            plt.title('player: {}: state: {}, action: {}'.format(self.full_name(env), s, a))
            plt.xlabel('step')
            plt.ylabel('pi[a]')
            plt.grid()
            x, y = list(zip(*((t, pi[a]) for t, pi in record)))
            x, y = list(x) + [env.t + 1.0], list(y) + [y[-1]]
            plt.plot(x, y, 'r-')
        fig.savefig('policy/{}_{}.pdf'.format(self.full_name(env), s))
        plt.close(fig)

    def record_policy(self, s, env):
        pass
        # if env.numplots != 0:
        #     if s in self.record:
        #         self.record[s].append((env.t - 0.01, self.record[s][-1][1]))
        #     self.record[s].append((env.t, np.copy(self.pi[s])))


class QAgent(BaseQAgent):
    def __init__(self, id_, action_num, env, **kwargs):
        super().__init__('q', id_, action_num, env, **kwargs)
        self.Q = defaultdict(partial(np.random.rand, self.action_num))
        self.R = defaultdict(partial(np.zeros, self.action_num))
        self.count_R = defaultdict(partial(np.zeros, self.action_num))

    def done(self, env):
        self.R.clear()
        self.count_R.clear()
        super().done(env)

    def update(self, s, a, o, r, s2, env, done=False):
        self.count_R[s][a] += 1.0
        self.R[s][a] += (r - self.R[s][a]) / self.count_R[s][a]
        Q = self.Q[s]
        V = self.val(s2)
        decay_alpha = self.step_decay()
        if done:
            Q[a] = Q[a] + decay_alpha * (r - Q[a])
        else:
            Q[a] = Q[a] + decay_alpha * (r + self.gamma * V - Q[a])
        print(self.epoch)
        self.update_policy(s, a, env)
        self.record_policy(s, env)
        self.epoch += 1

    def val(self, s):
        return np.max(self.Q[s])

    def update_policy(self, s, a, env):
        Q = self.Q[s]
        self.pi[s] = (Q == np.max(Q)).astype(np.double)


class PGAAPPAgent(QAgent):
    def __init__(self, id_, action_num, env, eta=0.01, **kwargs):
        super().__init__(id_, action_num, env, **kwargs)
        self.name = 'pha-app'
        self.eta = eta
        self.pi_history = [deepcopy(self.pi)]

    def update_policy(self, s, a, game):
        V = np.dot(self.pi[s], self.Q[s])
        delta_hat_A = np.zeros(self.action_num)
        delta_A = np.zeros(self.action_num)
        for ai in range(self.action_num):
            if self.pi[s][ai] == 1:
                delta_hat_A[ai]= self.Q[s][ai] - V
            else:
                delta_hat_A[ai] = (self.Q[s][ai] - V) / (1 - self.pi[s][ai])
            delta_A[ai] = delta_hat_A[ai] - self.gamma * abs(delta_hat_A[ai]) *self.pi[s][ai]
        self.pi[s] += self.eta * delta_A
        StationaryAgent.normalize(self.pi[s])
        self.pi_history.append(deepcopy(self.pi))


class GIGAWoLFAgent(QAgent):
    def __init__(self, id_, action_num, env, eta=0.01, **kwargs):
        super().__init__(id_, action_num, env, **kwargs)
        self.name = 'giga-wolf'
        self.eta = eta
        self.pi_history = [deepcopy(self.pi)]

    def update_policy(self, s, a, game):
        V = np.dot(self.pi[s], self.Q[s])
        delta_hat_A = np.zeros(self.action_num)
        delta_A = np.zeros(self.action_num)
        for ai in range(self.action_num):
            if self.pi[s][ai] == 1:
                delta_hat_A[ai]= self.Q[s][ai] - V
            else:
                delta_hat_A[ai] = (self.Q[s][ai] - V) / (1 - self.pi[s][ai])
            delta_A[ai] = delta_hat_A[ai] - self.gamma * abs(delta_hat_A[ai]) *self.pi[s][ai]
        self.pi[s] += self.eta * delta_A
        StationaryAgent.normalize(self.pi[s])
        self.pi_history.append(deepcopy(self.pi))


class EMAQAgent(QAgent):
    def __init__(self, id_, action_num, env, delta1=0.001, delta2=0.002, **kwargs):
        super().__init__(id_, action_num, env, **kwargs)
        self.name = 'emaq'
        self.delta1 = delta1
        self.delta2 = delta2
        self.pi_history = [deepcopy(self.pi)]

    def update_policy(self, s, a, game):
        if a == np.argmax(self.Q[s]):
            delta = self.delta1
            vi = np.zeros(self.action_num)
            vi[a] = 1.
        else:
            delta = self.delta2
            vi = np.zeros(self.action_num)
            vi[a] = 0.

        self.pi[s] = (1 - delta) * self.pi[s] + delta * vi
        StationaryAgent.normalize(self.pi[s])
        self.pi_history.append(deepcopy(self.pi))


class OMQAgent(QAgent):
    def __init__(self, id_, action_num, env, **kwargs):
        super().__init__(id_, action_num, env, **kwargs)
        self.name = 'omq'
        self.count_SO = defaultdict(partial(np.zeros, self.action_num))
        self.opponent_pi = defaultdict(partial(np.random.dirichlet, [1.0] * self.action_num))
        self.pi_history = [deepcopy(self.pi)]
        self.opponent_pi_history = [deepcopy(self.opponent_pi)]
        self.Q = defaultdict(partial(np.random.rand, *(self.action_num, self.action_num)))
        self.R = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))
        self.count_R = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))


    def update(self, s, a, o, r, s2, env, done=False):
        self.count_SO[s][o] += 1.
        self.opponent_pi[s] = self.count_SO[s] / np.sum(self.count_SO[s])
        self.count_R[s][a][o] += 1.0
        self.R[s][a][o] += (r - self.R[s][a][o]) / self.count_R[s][a][o]
        Q = self.Q[s]
        V = self.val(s2)
        decay_alpha = self.step_decay()
        if done:
            Q[a][o] = Q[a][o] + decay_alpha * (r - Q[a])
        else:
            Q[a][o] = Q[a][o] + decay_alpha * (r + self.gamma * V - Q[a][o])
        print(self.epoch)
        self.update_policy(s, a, env)
        self.record_policy(s, env)
        self.epoch += 1

    def val(self, s):
        return np.max(np.dot(self.Q[s], self.opponent_pi[s]))

    def update_policy(self, s, a, game):
        # print('Qs {}'.format(self.Q[s]))
        # print('OPI {}'.format(self.opponent_best_pi[s]))
        # print('pis: ' + str(np.dot(self.Q[s], self.opponent_best_pi[s])))
        self.pi[s] = utils.softmax(np.dot(self.Q[s], self.opponent_pi[s]))

        # print('pis: ' + str(np.sum(np.dot(self.Q[s], self.opponent_best_pi[s]))))
        self.pi_history.append(deepcopy(self.pi))
        self.opponent_pi_history.append(deepcopy(self.opponent_pi))
        print('opponent pi of {}: {}'.format(self.id_, self.opponent_pi[s]))


class RRQAgent(QAgent):
    def __init__(self, id_, action_num, env, phi_type='count', a_policy='softmax', **kwargs):
        super().__init__(id_, action_num, env, **kwargs)
        self.name = 'RR2Q'
        self.phi_type = phi_type
        self.a_policy = a_policy
        self.count_AOS = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))
        self.count_OS = defaultdict(partial(np.zeros, (self.action_num, )))
        self.opponent_best_pi = defaultdict(partial(np.random.dirichlet, [1.0] * self.action_num))
        self.pi_history = [deepcopy(self.pi)]
        self.opponent_best_pi_history = [deepcopy(self.opponent_best_pi)]
        self.Q = defaultdict(partial(np.random.rand, *(self.action_num, self.action_num)))
        self.Q_A = defaultdict(partial(np.random.rand, self.action_num))
        self.R = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))
        self.count_R = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))


    def update(self, s, a, o, r, s2, env, done=False, tau=0.5):
        self.count_AOS[s][a][o] += 1.0
        self.count_OS[s][o] += 1.
        decay_alpha = self.step_decay()
        if self.phi_type == 'count':
            count_sum = np.reshape(np.repeat(np.sum(self.count_AOS[s], 1), self.action_num), (self.action_num, self.action_num))
            self.opponent_best_pi[s] = self.count_AOS[s] / (count_sum + 0.1)
            self.opponent_best_pi[s] = self.opponent_best_pi[s] / (np.sum(self.opponent_best_pi[s]) + 0.1)
        elif self.phi_type == 'norm-exp':
            self.Q_A_reshaped = np.reshape(np.repeat(self.Q_A[s], self.action_num), (self.action_num, self.action_num))
            self.opponent_best_pi[s] = np.log(np.exp((self.Q[s] - self.Q_A_reshaped)))
            self.opponent_best_pi[s] = self.opponent_best_pi[s] / np.reshape(
                np.repeat(np.sum(self.opponent_best_pi[s], 1), self.action_num), (self.action_num, self.action_num))

        self.count_R[s][a][o] += 1.0
        self.R[s][a][o] += (r - self.R[s][a][o]) / self.count_R[s][a][o]
        Q = self.Q[s]
        V = self.val(s2)
        if done:
            Q[a][o] = Q[a][o] + decay_alpha * (r - Q[a][o])
            self.Q_A[s][a] = self.Q_A[s][a] + decay_alpha * (r - self.Q_A[s][a])
        else:
            Q[a][o] = Q[a][o] + decay_alpha * (r + self.gamma * V - Q[a][o])
            self.Q_A[s][a] = self.Q_A[s][a] + decay_alpha * (r + self.gamma * V - self.Q_A[s][a])
        print(self.epoch)
        self.update_policy(s, a, env)
        self.record_policy(s, env)
        self.epoch += 1

    def val(self, s):
        return np.max(np.sum(np.multiply(self.Q[s], self.opponent_best_pi[s]), 1))

    def update_policy(self, s, a, game):
        if self.a_policy == 'softmax':
            self.pi[s] = utils.softmax(np.sum(np.multiply(self.Q[s], self.opponent_best_pi[s]), 1))
        else:
            Q = np.sum(np.multiply(self.Q[s], self.opponent_best_pi[s]), 1)
            self.pi[s] = (Q == np.max(Q)).astype(np.double)
        self.pi_history.append(deepcopy(self.pi))
        self.opponent_best_pi_history.append(deepcopy(self.opponent_best_pi))
        print('opponent pi of {}: {}'.format(self.id_, self.opponent_best_pi))



class GRRQAgent(QAgent):
    def __init__(self, id_, action_num, env, k=0, phi_type='count', a_policy='softmax', **kwargs):
        super().__init__(id_, action_num, env, **kwargs)
        self.name = 'GRRQ'
        self.k = k
        self.phi_type = phi_type
        self.a_policy = a_policy
        self.count_AOS = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))
        self.opponent_best_pi = defaultdict(partial(np.random.dirichlet, [1.0] * self.action_num))
        self.pi_history = [deepcopy(self.pi)]
        self.opponent_best_pi_history = [deepcopy(self.opponent_best_pi)]
        self.Q = defaultdict(partial(np.random.rand, *(self.action_num, self.action_num)))
        self.Q_A = defaultdict(partial(np.random.rand, self.action_num))
        self.R = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))
        self.count_R = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))





    def update(self, s, a, o, r, s2, env, done=False):
        self.count_AOS[s][a][o] += 1.0
        decay_alpha = self.step_decay()
        if self.phi_type == 'count':
            count_sum = np.reshape(np.repeat(np.sum(self.count_AOS[s], 1), self.action_num), (self.action_num, self.action_num))
            self.opponent_best_pi[s] = self.count_AOS[s] / (count_sum + 0.1)
        elif self.phi_type == 'norm-exp':
            self.Q_A_reshaped = np.reshape(np.repeat(self.Q_A[s], self.action_num), (self.action_num, self.action_num))
            self.opponent_best_pi[s] = np.log(np.exp(self.Q[s] - self.Q_A_reshaped))
            self.opponent_best_pi[s] = self.opponent_best_pi[s] / np.reshape(
                np.repeat(np.sum(self.opponent_best_pi[s], 1), self.action_num), (self.action_num, self.action_num))

        self.count_R[s][a][o] += 1.0
        self.R[s][a][o] += (r - self.R[s][a][o]) / self.count_R[s][a][o]
        Q = self.Q[s]
        V = self.val(s2)
        if done:
            Q[a][o] = Q[a][o] + decay_alpha * (r - Q[a][o])
            self.Q_A[s][a] = self.Q_A[s][a] + decay_alpha * (r - self.Q_A[s][a])
        else:
            Q[a][o] = Q[a][o] + decay_alpha * (r + self.gamma * V - Q[a][o])
            self.Q_A[s][a] = self.Q_A[s][a] + decay_alpha * (r + self.gamma * V - self.Q_A[s][a])
        print(self.epoch)
        self.update_policy(s, a, env)
        self.record_policy(s, env)
        self.epoch += 1

    def val(self, s):
        return np.max(np.sum(np.multiply(self.Q[s], self.opponent_best_pi[s]), 1))

    def update_policy(self, s, a, game):
        if self.a_policy == 'softmax':
            self.pi[s] = utils.softmax(np.sum(np.multiply(self.Q[s], self.opponent_best_pi[s]), 1))
        else:
            Q = np.sum(np.multiply(self.Q[s], self.opponent_best_pi[s]), 1)
            self.pi[s] = (Q == np.max(Q)).astype(np.double)
        self.pi_history.append(deepcopy(self.pi))
        self.opponent_best_pi_history.append(deepcopy(self.opponent_best_pi))
        print('opponent pi of {}: {}'.format(self.id_, self.opponent_best_pi))



class MinimaxQAgent(BaseQAgent):
    def __init__(self, id_, action_num, env, opp_action_num):
        super().__init__('minimax', id_, action_num, env)
        self.solvers = []
        self.opp_action_num = opp_action_num
        self.pi_history = [deepcopy(self.pi)]
        self.Q = defaultdict(partial(np.random.rand, self.action_num, self.opp_action_num))

    def done(self, env):
        self.solvers.clear()
        super().done(env)

    def val(self, s):
        Q = self.Q[s]
        pi = self.pi[s]
        return min(np.dot(pi, Q[:, o]) for o in range(self.opp_action_num))

    def update(self, s, a, o, r, s2, env):
        Q = self.Q[s]
        V = self.val(s2)
        decay_alpha = self.step_decay()
        Q[a, o] = Q[a, o] + decay_alpha * (r + self.gamma * V - Q[a, o])
        self.update_policy(s, a, env)
        self.record_policy(s, env)

    def update_policy(self, s, a, env):
        self.initialize_solvers()
        for solver, lib in self.solvers:
            try:
                self.pi[s] = self.lp_solve(self.Q[s], solver, lib)
                StationaryAgent.normalize(self.pi[s])
                self.pi_history.append(deepcopy(self.pi))
            except Exception as e:
                print('optimization using {} failed: {}'.format(solver, e))
                continue
            else: break

    def initialize_solvers(self):
        if not self.solvers:
            for lib in ['gurobipy', 'scipy.optimize', 'pulp']:
                try: self.solvers.append((lib, importlib.import_module(lib)))
                except: pass

    def lp_solve(self, Q, solver, lib):
        ret = None

        if solver == 'scipy.optimize':
            c = np.append(np.zeros(self.action_num), -1.0)
            A_ub = np.c_[-Q.T, np.ones(self.opp_action_num)]
            b_ub = np.zeros(self.opp_action_num)
            A_eq = np.array([np.append(np.ones(self.action_num), 0.0)])
            b_eq = np.array([1.0])
            bounds = [(0.0, 1.0) for _ in range(self.action_num)] + [(-np.inf, np.inf)]
            res = lib.linprog(
                c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            ret = res.x[:-1]
        elif solver == 'gurobipy':
            m = lib.Model('LP')
            m.setParam('OutputFlag', 0)
            m.setParam('LogFile', '')
            m.setParam('LogToConsole', 0)
            v = m.addVar(name='v')
            pi = {}
            for a in range(self.action_num):
                pi[a] = m.addVar(lb=0.0, ub=1.0, name='pi_{}'.format(a))
            m.update()
            m.setObjective(v, sense=lib.GRB.MAXIMIZE)
            for o in range(self.opp_action_num):
                m.addConstr(
                    lib.quicksum(pi[a] * Q[a, o] for a in range(self.action_num)) >= v,
                    name='c_o{}'.format(o))
            m.addConstr(lib.quicksum(pi[a] for a in range(self.action_num)) == 1, name='c_pi')
            m.optimize()
            ret = np.array([pi[a].X for a in range(self.action_num)])
        elif solver == 'pulp':
            v = lib.LpVariable('v')
            pi = lib.LpVariable.dicts('pi', list(range(self.action_num)), 0, 1)
            prob = lib.LpProblem('LP', lib.LpMaximize)
            prob += v
            for o in range(self.opp_action_num):
                prob += lib.lpSum(pi[a] * Q[a, o] for a in range(self.action_num)) >= v
            prob += lib.lpSum(pi[a] for a in range(self.action_num)) == 1
            prob.solve(lib.GLPK_CMD(msg=0))
            ret = np.array([lib.value(pi[a]) for a in range(self.action_num)])

        if not (ret >= 0.0).all():
            raise Exception('{} - negative probability error: {}'.format(solver, ret))

        return ret


class MetaControlAgent(Agent):
    def __init__(self, id_, action_num, env, opp_action_num):
        super().__init__('metacontrol', id_, action_num, env)
        self.agents = [QAgent(id_, action_num, env), MinimaxQAgent(id_, action_num, env, opp_action_num)]
        self.n = np.zeros(len(self.agents))
        self.controller = None

    def act(self, s, exploration, env):
        print([self.val(i, s) for i in range(len(self.agents))])
        self.controller = np.argmax([self.val(i, s) for i in range(len(self.agents))])
        return self.agents[self.controller].act(s, exploration, env)

    def done(self, env):
        for agent in self.agents:
            agent.done(env)

    def val(self, i, s):
        return self.agents[i].val(s)

    def update(self, s, a, o, r, s2, env):
        for agent in self.agents:
            agent.update(s, a, o, r, s2, env)

        self.n[self.controller] += 1
        print('id: {}, n: {} ({}%)'.format(self.id_, self.n, 100.0 * self.n / np.sum(self.n)))
