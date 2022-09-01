from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import DEEnv
from gym import spaces
import numpy as np
import os
import csv
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

HISTORY_LENGTH = 40
INPUT_DIM = 10
POP_SIZE = 18

DEFAULT_CFG_SPACE = CS.ConfigurationSpace()
F_MU = CSH.UniformFloatHyperparameter(name='f_mu', lower=0, upper=1)
DEFAULT_CFG_SPACE.add_hyperparameter(F_MU)

INFO = {
    "identifier": "DE",
    "name": "Cauchy Adaption in DE",
    "reward": "Negative best function value",
    "state_description": [
        "Loc",
        "Past Deltas",
        "Population Size",
        "History Deltas",
        "Past Sigma Deltas",
    ],
}

DE_DEFAULTS = objdict(
    {
        "action_space_class": "Box",
        "action_space_args": [np.array([0]), np.array([1])],
        "config_space": DEFAULT_CFG_SPACE,
        "observation_space_class": "Box",
        "observation_space_type": np.float64,
        "observation_space_args": [
                np.array([-np.inf for _ in range(HISTORY_LENGTH + 1 + 2 * POP_SIZE)]),
                np.array([np.inf for _ in range(1 + HISTORY_LENGTH + 2 * POP_SIZE)]),
        ],
        "reward_range": (-(10 ** 9), 0),
        "cutoff": 100,
        "hist_length": HISTORY_LENGTH,
        "seed": 0,
        "instance_set_path": "../instance_sets/de/de_train.csv",
        "test_set_path": "../instance_sets/de/de_test.csv",
        "benchmark_info": INFO,
    }
)


class DEBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for DE-ES
    """

    def __init__(self, config_path=None, config=None):
        """
        Initialize DE Benchmark

        Parameters
        -------
        config_path : str
            Path to confg file (optional)
        """
        super(DEBenchmark, self).__init__(config_path, config)
        if not self.config:
            self.config = objdict(DE_DEFAULTS.copy())

        for key in DE_DEFAULTS:
            if key not in self.config:
                self.config[key] = DE_DEFAULTS[key]

    def get_environment(self):
        """
        Return DEEnv env with current configuration

        Returns
        -------
        DEEnv
            DE environment
        """
        if "instance_set" not in self.config.keys():
            self.read_instance_set()

        #Read test set if path is specified
        if "test_set" not in self.config.keys() and "test_set_path" in self.config.keys():
            self.read_instance_set(test=True)

        env = DEEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        return env

    def read_instance_set(self, test=False):
        """
        Read path of instances from config into list
        """
        if test:
            path = (
                os.path.dirname(os.path.abspath(__file__))
                + "/"
                + self.config.test_set_path
            )
            keyword = "test_set"
        else:
            path = (
                os.path.dirname(os.path.abspath(__file__))
                + "/"
                + self.config.instance_set_path
            )
            keyword = "instance_set"

        self.config[keyword] = {}
        with open(path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                init_locs = [np.fromstring(row[f"init_loc{i}"].strip('[]'), dtype=float, sep=' ') for i in range(int(row["pop_size"]))]
                instance = [
                    int(row["fcn_index"]),
                    int(row["dim"]),
                    int(row["pop_size"]),
                    row["mutation_strat"],
                    row["crossover_strat"],
                    float(row["crossp"]),
                    float(row["lower_bound"]),
                    float(row["upper_bound"]),
                    init_locs,
                ]
                self.config[keyword][int(row["ID"])] = instance

    def get_benchmark(self, seed=0):
        """
        Get benchmark

        Parameters
        -------
        seed : int
            Environment seed

        Returns
        -------
        env : DEEnv
            DE environment
        """
        self.config = objdict(DE_DEFAULTS.copy())
        self.config.seed = seed
        self.read_instance_set()
        self.read_instance_set(test=True)
        return DEEnv(self.config)
