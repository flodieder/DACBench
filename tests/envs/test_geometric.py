import unittest

import numpy as np
from dacbench import AbstractEnv
from dacbench.envs import GeometricEnv
from dacbench.benchmarks.geometric_benchmark import GEOMETRIC_DEFAULTS


class TestSigmoidEnv(unittest.TestCase):
    def make_env(self):
        config = GEOMETRIC_DEFAULTS
        config["instance_set"] = {20: [20, "linear", 0.4, 0.2]}
        env = GeometricEnv(config)
        return env

    def test_setup(self):
        # TODO
        env = self.make_env()
        self.assertTrue(issubclass(type(env), AbstractEnv))
        self.assertFalse(env.np_random is None)
        self.assertTrue(
            np.array_equal(
                env.shifts, 5 * np.ones(len(GEOMETRIC_DEFAULTS["action_values"]))
            )
        )
        self.assertTrue(
            np.array_equal(
                env.slopes, -1 * np.ones(len(GEOMETRIC_DEFAULTS["action_values"]))
            )
        )
        self.assertTrue(env.n_actions == len(GEOMETRIC_DEFAULTS["action_values"]))
        self.assertTrue(env.slope_multiplier == GEOMETRIC_DEFAULTS["slope_multiplier"])
        self.assertTrue(env.action_vals == GEOMETRIC_DEFAULTS["action_values"])

    def test_reset(self):
        # TODO
        env = self.make_env()
        state = env.reset()
        self.assertTrue(np.array_equal(env.shifts, [0, 1]))
        self.assertTrue(np.array_equal(env.slopes, [2, 3]))
        self.assertTrue(state[0] == GEOMETRIC_DEFAULTS["cutoff"])
        self.assertTrue(np.array_equal([state[1], state[3]], env.shifts))
        self.assertTrue(np.array_equal([state[2], state[4]], env.slopes))
        self.assertTrue(np.array_equal(state[5:], -1 * np.ones(2)))

    def test_step(self):
        # TODO
        env = self.make_env()
        env.reset()
        state, reward, done, meta = env.step(1)
        self.assertTrue(reward >= env.reward_range[0])
        self.assertTrue(reward <= env.reward_range[1])
        self.assertTrue(state[0] == 9)
        self.assertTrue(np.array_equal([state[1], state[3]], env.shifts))
        self.assertTrue(np.array_equal([state[2], state[4]], env.slopes))
        self.assertTrue(len(state) == 7)
        self.assertFalse(done)
        self.assertTrue(len(meta.keys()) == 0)

    def test_close(self):
        env = self.make_env()
        self.assertTrue(env.close())
