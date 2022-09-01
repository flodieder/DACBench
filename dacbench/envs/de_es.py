"""
DE-ES environment adapted from TODO
Original author: probably me?
"""

import numpy as np
from scipy._lib._util import check_random_state
from collections import deque
from cma import bbobbenchmarks as bn
import threading
import warnings
from dacbench import AbstractEnv
import resource
import sys

resource.setrlimit(resource.RLIMIT_STACK, (2 ** 35, -1))
sys.setrecursionlimit(10 ** 9)

warnings.filterwarnings("ignore")


class DEEnv(AbstractEnv):
    """
    Environment to control the step size of DE-ES
    """

    def __init__(self, config):
        """
        Initialize DE Env

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super(DEEnv, self).__init__(config)
        self.lower_bound = self.instance[6]
        self.upper_bound = self.instance[7]
        self.dist_max = np.linalg.norm(self.lower_bound - self.upper_bound)

        self.action_dim = len(self.config['action_space_args'])//2
        self.f_gamma = 0
        self.crossp = self.instance[5]

        # State information
        self.fbest = 0
        self.fbest_old = 0
        self.idxbest = 0
        self.f_bsf = 0
        self.f_bsf_old = 0
        self.f_wsf = 0
        self.f_wsf_old = 0
        self.stag_counter = 0

        # Deques that store the information for the state
        self.history_len = config.hist_length
        self.past_obj_value_deltas = deque(maxlen=self.history_len)
        self.past_mut_deltas = deque(maxlen=self.history_len)

        self.solutions = None
        self.fitness = []
        self.traj = []

        self.popsize = self.instance[2]
        self.mutation_strat = self.instance[3]
        self.crossover_strat = self.instance[4]
        self.mut = [0] * self.popsize
        self.mut_old = [0] * self.popsize

        if "reward_function" in config.keys():
            self.get_reward = config["reward_function"]
        else:
            self.get_reward = self.get_default_reward

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        else:
            self.get_state = self.get_default_state

    def step(self, action):
        """
        Execute environment step

        Parameters
        ----------
        action : list
            action to execute

        Returns
        -------
        np.array, float, bool, dict
            state, reward, done, info
        """
        done = super(DEEnv, self).step_()
        if not done:
            """Moves forward in time one step"""
            self.fbest_old = self.fbest
            self.f_bsf_old = self.f_bsf
            self.f_wsf_old = self.f_wsf
            self.last_action = action
            self.mut_old = self.mut[self.idxbest]
            f_mu = max(0, action[0])
            f_gamma = 0.1 if self.action_dim == 1 else max(0, action[1])
            # Check for 0 gamma to save compute
            if f_gamma == 0:
                self.mut = [f_mu] * self.popsize
            else:
                self.mut = np.clip(self._cauchy(mu=f_mu, gamma=f_gamma),
                                   a_min=0,
                                   a_max=1)

            # Generate trial vectors and evaluate them
            for j in range(self.popsize):
                ui = self._evolve(j)
                u_mixed = self._scale(ui)
                fit = self.fcn(u_mixed)

                # Perform Selection
                if fit < self.fitness[j]:
                    if fit < self.fitness[self.idxbest]:
                        self.idxbest = j
                    self.fitness[j] = fit
                    self.pop[j] = ui
                    self.successes[j] = True
                elif fit > self.fitness[j] and fit > self.f_wsf:
                    self.f_wsf = fit

                self.traj.append(self.fitness[self.idxbest])

            # Temp Computations for the state features
            a = np.ones(self.popsize) * self.f_wsf
            a[:self.popsize//2] = self.f_bsf
            std_max = np.std(a)

            idxs = [idx for idx in range(self.popsize)]
            r1, r2, r3 = self.pop[self.np_random.choice(
                idxs, 3, replace=False)]

            if self.fbest != self.fitness[self.idxbest]:
                self.stag_counter = 0
                self.f_bsf = self.fitness[self.idxbest]
                self.fbest = self.fitness[self.idxbest]
            else:
                self.stag_counter += 1

            # print("Fbest: %s  Iteration: %s" % (self.fbest, self.c_step),
                  # flush=True)

            normalized_fbest_delta = (self.fbest_old - self.fbest)/self.fbest_old if self.fbest_old != 0 else self.fbest_old - self.fbest
            self.past_obj_value_deltas.appendleft(normalized_fbest_delta)
            normalized_mut = (self.mut_old - self.mut[self.idxbest]) / self.mut_old if self.mut_old != 0 else 0.0
            self.past_mut_deltas.appendleft(normalized_mut)

        return self.get_state(self), self.get_reward(self), done, {
            'mutations': self.mut,
            'successes': self.successes
        }

    def reset(self):
        """
        Reset environment

        Returns
        -------
        np.array
            Environment state
        """
        super(DEEnv, self).reset_()
        self.past_obj_value_deltas.extend([0] * self.history_len)
        self.past_mut_deltas.extend([0] * self.history_len)
        self.pop = np.array(self.instance[8])
        self.dim = self.instance[1]
        self.fcn = bn.instantiate(self.instance[0])[0]

        self.fitness = []
        temp = self.pop[0]
        for j in range(self.popsize):
            temp = self._scale(self._boundary_checking(self.pop[j]))
            f_temp = self.fcn(temp)
            self.fitness.append(f_temp)
            if j == 0:
                self.fbest = f_temp
                self.idxbest = 0
                self.f_bsf = f_temp
                self.f_wsf = f_temp
            else:
                if f_temp < self.fbest:
                    self.fbest = f_temp
                    self.idxbest = j
                if f_temp > self.f_wsf:
                    self.f_wsf = f_temp
        self.f_bsf = self.fbest
        self.mut = [0] * self.popsize
        self.mut_old = [0] * self.popsize
        self.successes = np.full(self.popsize, False)
        self.stag_counter = 0
        self.last_action = np.zeros(shape=(1))

        return self.get_state(self)

    def close(self):
        """
        No additional cleanup necessary

        Returns
        -------
        bool
            Cleanup flag
        """
        return True

    def render(self, mode: str = "human"):
        """
        Render env in human mode

        Parameters
        ----------
        mode : str
            Execution mode
        """
        if mode != "human":
            raise NotImplementedError

        pass

    def get_default_reward(self, _):
        """
        Compute reward

        Returns
        -------
        float
            Reward

        """
        reward = min(self.reward_range[1], max(self.reward_range[0], -self.fbest))
        return reward

    def get_default_state(self, _):
        """
        Gather state description

        Returns
        -------
        dict
            Environment state

        """
        state = np.concatenate((self.past_mut_deltas, self.mut, self.successes, self.last_action))
        return state

    def _boundary_checking(self, x):
        for index in np.where((x < 0) | (x > 1))[0]:
            x[index] = self.np_random.rand()

        return x


    def _evolve(self, j: int):
        """
        Returns a new trial vector based on the mutation and crossover strategy with individual j as a parent

        Args:
            j: index of the parent in the current population
        """
        best_idv = self.pop[self.idxbest]
        current_idv = self.pop[j]

        # perform mutation operation
        if self.mutation_strat == "rand1":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2, r3 = self.pop[self.np_random.choice(
                idxs, 3, replace=False)]
            temp = r1 + self.mut[j] * (r2 - r3)
            # print("Temp: %s" % temp, flush=True)

        elif self.mutation_strat == "best1":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2 = self.pop[self.np_random.choice(
                idxs, 2, replace=False)]
            temp = best_idv + self.mut[j] * (r1 - r2)

        elif self.mutation_strat == "rand2":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2, r3, r4, r5 = self.pop[self.np_random.choice(
                idxs, 5, replace=False)]
            temp = r1 + self.mut[j] * (r1 - r2) + self.mut[j] * (r3 - r4)

        elif self.mutation_strat == "best2":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2, r3, r4 = self.pop[self.np_random.choice(
                idxs, 4, replace=False)]
            temp = best_idv + self.mut[j] * (r1 - r2) + self.mut[j] * (r3 - r4)

        elif self.mutation_strat == "currenttobest1":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2 = self.pop[self.np_random.choice(
                idxs, 2, replace=False)]
            temp = current_idv + self.mut[j] * (
                best_idv - current_idv) + self.mut[j] * (r1 - r2)

        elif self.mutation_strat == "randtobest1":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2, r3 = self.pop[self.np_random.choice(
                idxs, 3, replace=False)]
            temp = r1 + self.mut[j] * (best_idv - r1) + self.mut[j] * (r2 - r3)

        else:
            print('No Mutation strategy selected')
            return

        #bound checking
        vi = self._boundary_checking(temp)

        # perform crossover operation
        if self.crossover_strat == 'bin':
            cross_points = self.np_random.rand(
                self.dim) < self.crossp

            if not np.any(cross_points):
                cross_points[self.np_random.randint(
                    0, self.dim)] = True
            ui = np.where(cross_points, vi, current_idv)

        else:
            # Untested
            i = 0
            fill_point = self.np_random.randint(0, self.dim)
            while (i < self.dim and self.np_random.rand(0, 1) < self.crossp):
                ui[fill_point] = vi[fill_point]
                fill_point = (fill_point + 1) % self.dim
                i += 1

        return ui

    def _scale(self, x):
        """
        Scales the individual from [0, 1] to [lower_bound, upper_bound]

        Args:
            x: The individual to scale
        """
        temp = self.lower_bound + x * np.abs(self.lower_bound -
                                             self.upper_bound)
        return temp

    def _cauchy(self, mu=0.5, gamma=0.1):

        value = mu + gamma * np.tan(
            np.pi * (self.np_random.rand(self.popsize) - 0.5))

        for i in range(len(value)):
            while value[i] <= 0:
                value[i] = mu + gamma * np.tan(
                    np.pi * (self.np_random.rand() - 0.5))

        maxx = np.ones(self.popsize)
        value = np.minimum(maxx, value)

        return list(value)
