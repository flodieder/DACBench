import ConfigSpace
import gym
import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from ConfigSpace.read_and_write import json


class Space(ConfigurationSpace, gym.Space):
    def sample(self):
        return self.sample_configuration().get_array()

    def contains(self, x):
        if isinstance(x, dict):
            config = ConfigSpace.Configuration(configuration_space=self, values=x)
        elif isinstance(x, (np.ndarray, list)):
            config = ConfigSpace.Configuration(configuration_space=self, vector=x)
        else:
            raise ValueError("x must be either a dict or a numpy array")

        try:
            config.is_valid_configuration()
            return True
        except Exception:
            return False

    def seed(self, seed):
        ConfigurationSpace.seed(self, seed)

    @property
    def np_random(self):
        raise NotImplementedError("Not available in this implementation")


if __name__ == "__main__":
    space = Space()

    space.add_hyperparameter(UniformFloatHyperparameter(name="alpha", lower=0, upper=1))
    print(space.sample())
    print(space.sample() in space)
    print([-2] in space)
    print({"alpha": 0.3} in space)
    with open("test.json", "w") as f:
        f.write(json.write(space))
