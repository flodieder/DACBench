import unittest
import json
import os

from dacbench.benchmarks import GeometricBenchmark
from dacbench.envs import GeometricEnv


class TestSigmoidBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = GeometricBenchmark()
        env = bench.get_environment()
        self.assertTrue(issubclass(type(env), GeometricEnv))

    def test_setup(self):
        bench = GeometricBenchmark()
        self.assertTrue(bench.config is not None)

        config = {"dummy": 0}
        with open("test_conf.json", "w+") as fp:
            json.dump(config, fp)
        bench = GeometricBenchmark("test_conf.json")
        self.assertTrue(bench.config.dummy == 0)
        os.remove("test_conf.json")

    def test_save_conf(self):
        bench = GeometricBenchmark()
        bench.save_config("test_conf.json")
        with open("test_conf.json", "r") as fp:
            recovered = json.load(fp)
        for k in bench.config.keys():
            self.assertTrue(k in recovered.keys())
        os.remove("test_conf.json")

    def test_read_instances(self):
        # TODO check
        bench = GeometricBenchmark()
        bench.read_instance_set()
        self.assertTrue(len(bench.config.instance_set.keys()) == 300)
        self.assertTrue(len(bench.config.instance_set[0]) == 4)
        first_inst = bench.config.instance_set[0]

        bench2 = GeometricBenchmark()
        env = bench2.get_environment()
        self.assertTrue(len(env.instance_set[0]) == 4)
        self.assertTrue(env.instance_set[0] == first_inst)
        self.assertTrue(len(env.instance_set.keys()) == 300)

    def test_action_value_setting(self):
        # TODO check
        bench = GeometricBenchmark()
        bench.set_action_values([1, 2, 3])
        self.assertTrue(bench.config.action_values == [1, 2, 3])
        self.assertTrue(bench.config.action_space_args == [6])
        self.assertTrue(len(bench.config.observation_space_args[0]) == 10)

    def test_benchmark_env(self):
        bench = GeometricBenchmark()
        env = bench.get_benchmark()
        self.assertTrue(issubclass(type(env), GeometricEnv))
