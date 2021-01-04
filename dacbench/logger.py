import json
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from datetime import datetime
from functools import reduce
from itertools import chain
from pathlib import Path
from typing import Union, Dict, Any, Tuple, List

import pandas as pd

from dacbench import AbstractEnv, AbstractBenchmark
from dacbench.abstract_agent import AbstractDACBenchAgent


def split(predicate, iterable) -> Tuple[List, List]:
    """
    Splits the iterable into two list depending on the result of predicate.
    :param predicate:
    :param iterable:
    :return: positives, negatives
    """
    positives, negatives = [], []

    for item in iterable:
        (positives if predicate(item) else negatives).append(item)

    return positives, negatives


def flatten_log_entry(log_entry: Dict) -> List[Dict]:
    """
    Transforms a log entry of format like
    {
        'step': 0,
        'episode': 2,
        'some_value': {
            'values' : [34, 45],
            'times':['28-12-20 16:20:53', '28-12-20 16:21:30'],
        }
    }
    into
    [
        { 'step': 0,'episode': 2, 'value': 34, 'time': '28-12-20 16:20:53'},
        { 'step': 0,'episode': 2, 'value': 45, 'time': '28-12-20 16:21:30'}
    ]

    :param log_entry:
    :return:
    """
    dict_entries, top_level_entries = split(
        lambda item: isinstance(item[1], dict), log_entry.items()
    )
    rows = []
    for value_name, value_dict in dict_entries:
        current_rows = (
            dict(
                top_level_entries
                + [("value", value), ("time", time), ("name", value_name)]
            )
            for value, time in zip(value_dict["values"], value_dict["times"])
        )

        rows.extend(map(dict, current_rows))

    return rows


def log2dataframe(logs: List[dict], wide=False) -> pd.DataFrame:
    """
    Converts a list of log entries to a pandas dataframe.
    :param logs:
    :param wide: determines the format of the dataframe.
    wide=False (default) produces a dataframe with columns (episode, step, time, name, value)
    wide=True returns a dataframe (episode, step, time, name_1, name_2, ...) if the variable name_n has not been logged
    at (episode, step, time) name_n is NaN.
    :return:
    """
    flat_logs = map(flatten_log_entry, logs)
    rows = reduce(lambda l1, l2: l1 + l2, flat_logs)

    dataframe = pd.DataFrame(rows)

    if wide:
        primary_index_columns = ["episode", "step", "time"]
        field_id_column = "name"
        additional_columns = list(
            set(dataframe.columns)
            - set(primary_index_columns + ["value", field_id_column])
        )
        index_columns = primary_index_columns + additional_columns + [field_id_column]
        dataframe = dataframe.set_index(index_columns)
        dataframe = dataframe.unstack()
        dataframe.reset_index(inplace=True)
        dataframe.columns = [a if b == "" else b for a, b in dataframe.columns]

    return dataframe


class AbstractLogger(metaclass=ABCMeta):
    valid_types = {
        "recursive": [dict, list, tuple],
        "primitive": [str, int, float, bool],
    }

    def __init__(
        self,
        experiment_name: str,
        output_path: Path,
        step_write_frequency: int = None,
        episode_write_frequency: int = 1,
    ):
        """
        :param experiment_name:
        :param output_path:
        :param step_write_frequency: number of steps after which the loggers writes to file.
        If None only the data is only written to file if  write is called.
        :param episode_write_frequency: see step_write_frequency
        """
        self.experiment_name = experiment_name
        self.output_path = output_path
        self.log_dir = self._init_logging_dir(self.output_path / self.experiment_name)
        self.step_write_frequency = step_write_frequency
        self.episode_write_frequency = episode_write_frequency
        self.additional_info = {"seed": None, "instance": None}

    def _pretty_valid_types(self) -> str:
        valid_types = chain(
            self.valid_types["recursive"], self.valid_types["primitive"]
        )
        return ", ".join(map(lambda type_: type_.__name__, valid_types))

    @staticmethod
    def _init_logging_dir(log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def is_of_valid_type(self, value: Any) -> bool:
        f"""
        Check if the value of any type in {self._pretty_valid_types()}
        :param value:
        :return:
        """
        if any(isinstance(value, type) for type in self.valid_types["primitive"]):
            return True

        elif any(isinstance(value, type) for type in self.valid_types["recursive"]):
            value = value.keys() if isinstance(value, dict) else value
            return all(self.is_of_valid_type(sub_value) for sub_value in value)

        else:
            return False

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def next_step(self):
        pass

    @abstractmethod
    def next_episode(self):
        pass

    @abstractmethod
    def write(self):
        pass

    @abstractmethod
    def log(self, key, value):
        f"""
        Write value to list of values for key.
        :param key:
        :param value: the value must of of a type that is
        json serializable. Currently only {self._pretty_valid_types()} and recursive types of these are
        valid
        :return:
        """
        pass


class ModuleLogger(AbstractLogger):
    def __init__(
        self,
        output_path: Path,
        experiment_name: str,
        module: str,
        step_write_frequency: int = None,
        episode_write_frequency: int = 1,
    ) -> None:
        """
        All results are placed under 'output_path / experiment_name'
        :param output_path: the path where logged information should be stored
        :param experiment_name: name of the experiment.
        :param the module (mostly name of the wrapper), each wrapper get's its own file
        :param step_write_frequency: number of steps after which the loggers writes to file.
        If None only the data is only written to file if  write is called.
        :param episode_write_frequency: see step_write_frequency
        """
        super(ModuleLogger, self).__init__(
            experiment_name, output_path, step_write_frequency, episode_write_frequency
        )

        self.log_file = open(self.log_dir / f"{module}.jsonl", "a+")

        self.step = 0
        self.episode = 0
        self.buffer = []
        self.current_step = self.__init_dict()

    def close(self):
        if not self.log_file.closed:
            self.write()
            self.log_file.close()

    def __del__(self):
        if not self.log_file.closed:
            self.close()

    def __end_step(self):
        if self.current_step:
            self.current_step["step"] = self.step
            self.current_step["episode"] = self.episode
            self.current_step.update(self.additional_info)
            self.buffer.append(json.dumps(self.current_step))
        self.current_step = self.__init_dict()

    @staticmethod
    def __init_dict():
        return defaultdict(lambda: {"times": [], "values": []})

    def reset_episode(self):
        self.__end_step()
        self.episode = 0
        self.step = 0

    def __reset_step(self):
        self.__end_step()
        self.step = 0

    def next_step(self):
        self.__end_step()
        if (
            self.step_write_frequency is not None
            and self.step % self.step_write_frequency == 0
        ):
            self.write()
        self.step += 1

    def next_episode(self):
        self.__reset_step()
        if (
            self.episode_write_frequency is not None
            and self.episode % self.episode_write_frequency == 0
        ):
            self.write()
        self.episode += 1

    def write(self):
        self.__buffer_to_file()

    def __buffer_to_file(self):
        self.log_file.write("\n".join(self.buffer))
        self.log_file.write("\n")
        self.buffer.clear()
        self.log_file.flush()

    def set_additional_info(self, **kwargs):
        self.additional_info.update(kwargs)

    def log(
        self, key: str, value: Union[Dict, List, Tuple, str, int, float, bool]
    ) -> None:
        f"""
        Write value to list of values for key.
        :param key:
        :param value: the value must of of a type that is
        json serializable. Currently only {self._pretty_valid_types()} and recursive types of these are
        valid
        :return:
        """
        # TODO add numpy support
        if not self.is_of_valid_type(value):
            valid_types = self._pretty_valid_types()
            raise ValueError(
                f"value {type(value)} is not of valid type or a recursive composition of valid types ({valid_types})"
            )
        self.current_step[key]["times"].append(
            datetime.now().strftime("%d-%m-%y %H:%M:%S")
        )
        self.current_step[key]["values"].append(value)


class Logger(AbstractLogger):
    def __init__(
        self,
        experiment_name: str,
        output_path: Path,
        step_write_frequency: int = None,
        episode_write_frequency: int = 1,
    ) -> None:
        """
        :param experiment_name:
        :param output_path:
        :param step_write_frequency: number of steps after which the loggers writes to file.
        If None only the data is only written to file if  write is called.
        :param episode_write_frequency: see step_write_frequency
        """
        super(Logger, self).__init__(
            experiment_name, output_path, step_write_frequency, episode_write_frequency
        )
        self.module_logger: Dict[str, ModuleLogger] = dict()

    def close(self):
        for _, module_logger in self.module_logger.items():
            module_logger.close()

    def __del__(self):
        self.close()

    def next_step(self):
        for _, module_logger in self.module_logger.items():
            module_logger.next_step()

    def next_episode(self):
        for _, module_logger in self.module_logger.items():
            module_logger.next_episode()

    def reset_episode(self):
        for _, module_logger in self.module_logger.items():
            module_logger.reset_episode()

    def write(self):
        for _, module_logger in self.module_logger.items():
            module_logger.write()

    def add_module(self, module: Union[str, type]) -> ModuleLogger:
        module = module if isinstance(module, str) else module.__class__
        if module in self.module_logger:
            raise ValueError(f"Module {module} already registered")
        else:
            self.module_logger[module] = ModuleLogger(
                self.output_path,
                self.experiment_name,
                module,
                self.step_write_frequency,
                self.episode_write_frequency,
            )

        return self.module_logger[module]

    def add_agent(self, agent: AbstractDACBenchAgent):
        """
        Writes information about the agent
        :param agent:
        :return:
        """
        agent_config = {"type": agent.__class__}
        with open(self.output_path / "agent.json") as f:
            json.dump(agent_config, f)

    def set_env(self, env: AbstractEnv):
        """
        Writes information about the env
        :param env:
        :return:
        """
        raise NotImplementedError()

    def add_benchmark(self, benchmark: AbstractBenchmark) -> None:
        """
        Writes the config to the experiment path
        :param benchmark:
        :return:
        """
        benchmark.save_config(self.output_path)

    def set_additional_info(self, **kwargs):
        for _, module_logger in self.module_logger.items():
            module_logger.set_additional_info(**kwargs)

    def log(self, key, value, module):
        if module not in self.module_logger:
            raise ValueError(f"Module {module} not registered yet")
        self.module_logger.log(key, value)
