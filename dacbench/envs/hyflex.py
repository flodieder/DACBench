"""
Python wrapper classes for HyFlex (should be a separate file, or even project)
"""

from enum import Enum

from typing import List
from collections import deque

import numpy as np
import requests

HeuristicType = Enum('HeuristicType', 'CROSSOVER LOCAL_SEARCH MUTATION OTHER RUIN_RECREATE')
H_TYPE = HeuristicType
HISTORY_LENGTH = 5
MAXINT = 1e8


class ToyProblemDomain:
    """
    This class is a toy domain

    Example code for creating a toy environment:
        bench = HyFlexBenchmark(config_path='dacbench/additional_configs/hyflex/toy.json')
        env = bench.get_environment()
    """

    def _diff(self, l, exclude):
        return [i for i in l if i not in exclude]

    def __init__(self, seed: int, exclude={}):
        """
        Creates a new problem domain and creates a new random number generator using the seed provided. If
        the seed takes the value -1, the seed is generated taking the current System time. The random number generator
        is used for all stochastic operations, so the problem will be initialised in the same way if the seed is the
        same. Sets the solution memory size to 2.

        :param domain: the unqualified class name of the HyFlex domain to be wrapped, e.g., SAT, BinPacking, etc.
        :param seed: a random seed
        """
        # raise NotImplementedError
        h = [lambda x: x - 2,
             lambda x: x + 1,
             lambda x: x + 2,
             lambda x: x / 2 if x % 2 == 0 else 2 * x]
        self.heuristics = [h[i] for i in range(len(h)) if i not in exclude]
        self.heuristics_of_type = {H_TYPE.CROSSOVER: [],
                                   H_TYPE.LOCAL_SEARCH: self._diff([0], exclude),
                                   H_TYPE.MUTATION: self._diff([1, 2], exclude),
                                   H_TYPE.OTHER: [],
                                   H_TYPE.RUIN_RECREATE: self._diff([3], exclude)
                                   }
        self.mem_size = 2
        self.mem = None
        self.base = None
        
    def getHeuristicDescription(heuristic_id, current_f_inc):
        if heuristic_id==-1:
            return " x"
        if heuristic_id==0:
            return "-2"
        if heuristic_id==1:
            return "+1"
        if heuristic_id==2:
            return "+2"
        if current_f_inc%2==0:
            return "/2"
        return "*2"

    def getHeuristicsOfType(self, heuristicType: HeuristicType) -> List[int]:
        """
        Gets an array of heuristicIDs of the type specified by heuristicType.

        :param heuristicType: the heuristic type.
        :return: An list containing the indices of the heuristics of the type specified. If there are no heuristics of
            this type it returns None.
        """
        return self.heuristics_of_type[heuristicType]

    def loadInstance(self, instanceID: int) -> None:
        """
        Loads the instance specified by instanceID.

        :param instanceID: Specifies the instance to load. The ID's start at zero.
        :return: None
        """
        self.base = 1024 + 2 ** instanceID
        self.mem = [-1] * self.mem_size

    def setMemorySize(self, size: int) -> None:
        """
        Sets the size of the array where the solutions are stored. The default size is 2.

        :param size: The new size of the solution array.
        :return: None
        """
        self.mem_size = size
        self.mem = [-1] * self.mem_size

    def initialiseSolution(self, index: int) -> None:
        """
        Create an initial solution at a specified position in the memory array. The method of initialising the solution
        depends on the specific problem domain, but it is a random process, which will produce a different solution
        each time. The initialisation process may randomise all of the elements of the problem, or it may use a
        constructive heuristic with a randomised input.

        :param index: The index of the memory array at which the solution should be initialised.
        :return: None
        """
        self.mem[index] = self.base

    def getNumberOfHeuristics(self) -> int:
        """
        Gets the number of heuristics available in this problem domain

        :return: The number of heuristics available in this problem domain
        """
        return len(self.heuristics)

    def applyHeuristicUnary(self, heuristicID: int, solutionSourceIndex: int, solutionDestinationIndex: int) -> float:
        """
        Applies the heuristic specified by heuristicID to the solution at position solutionSourceIndex and places the
        resulting solution at position solutionDestinationIndex in the solution array. If the heuristic is a
        CROSSOVER type then the solution at solutionSourceIndex is just copied to solutionDestinationIndex.

        :param heuristicID: The ID of the heuristic to apply (starts at zero)
        :param solutionSourceIndex: The index of the solution in the memory array to which to apply the heuristic
        :param solutionDestinationIndex: The index in the memory array at which to store the resulting solution
        :return: the objective function value of the solution created by applying the heuristic
        """
        s = self.heuristics[heuristicID](self.mem[solutionSourceIndex])
        self.mem[solutionDestinationIndex] = s if s >= 0 else self.mem[solutionSourceIndex]
        return self.mem[solutionDestinationIndex]

    def copySolution(self, solutionSourceIndex: int, solutionDestinationIndex: int) -> None:
        """
        Copies a solution from one position in the solution array to another

        :param solutionSourceIndex: The position of the solution to copy
        :param solutionDestinationIndex: The position in the array to copy the solution to.
        :return: None
        """
        self.mem[solutionDestinationIndex] = self.mem[solutionSourceIndex]

    def toString(self) -> str:
        """
        Gets the name of the problem domain. For example, "Bin Packing"

        :return: the name of the ProblemDomain
        """
        return "Toy"

    def getFunctionValue(self, solutionIndex: int) -> float:
        """
        Gets the objective function value of the solution at index solutionIndex

        :param solutionIndex: The index of the solution from which the objective function is required
        :return: A double value of the solution's objective function value.
        """
        return self.mem[solutionIndex]


class HyflexProblemDomain:
    """
    This class implements a generic python wrapper for HyFlex problem domains.
    
    Example code for creating a Hyflex environment:
        bench = HyFlexBenchmark() # use the HYFLEX_DEFAULTS default configuration in dacbench/benchmarks/hyflex_benchmark.py
        env = bench.get_environment()
    """

    def __init__(self, domain: str, seed: int, host: str = "http://127.0.0.1:8080"):
        """
        Creates a new problem domain and creates a new random number generator using the seed provided. If
        the seed takes the value -1, the seed is generated taking the current System time. The random number generator
        is used for all stochastic operations, so the problem will be initialised in the same way if the seed is the
        same. Sets the solution memory size to 2.

        :param domain: the unqualified class name of the HyFlex domain to be wrapped, e.g., SAT, BinPacking, etc.
        :param seed: a random seed
        """
        self.domain = domain
        self.seed = seed
        self.host = host
        self.session = requests.Session()
        self.token = self.session.put(self.host + "/instantiate/" + domain + "/" + str(seed)).text

    def getHeuristicCallRecord(self) -> List[int]:
        """
        Shows how many times each low level heuristic has been called.

        :return: A list which contains an integer value for each low level heuristic, representing the number of times
            that heuristic has been called by the HyperHeuristic object.
        """
        return self.session.get(self.host + "/heuristic/record/call/" + self.token).json()

    def getHeuristicCallTimeRecord(self) -> List[int]:
        """
        Shows the total time that each low level heuristic has been operating on the problem.

        :return: A list which contains an integer value representing the total number of milliseconds used by each low
            level heuristic.
        """
        return self.session.get(self.host + "/heuristic/record/callTime/" + self.token).json()

    def setDepthOfSearch(self, depthOfSearch: float) -> None:
        """
        Sets the parameter specifying the extent to which a local search heuristic will modify the solution.
        This parameter is related to the number of improving steps to be completed by the local search heuristics.

        :param depthOfSearch: must be in the range 0 to 1. The initial value of 0.1 represents the default operation of
            the low level heuristic.
        :return: None
        """
        self.session.post(self.host + "/search/depth/" + self.token + "/" + str(depthOfSearch))

    def setIntensityOfMutation(self, intensityOfMutation: float) -> None:
        """
        Sets the parameter specifying the extent to which a mutation or ruin-recreate low level heuristic will mutate
        the solution. For a mutation heuristic, this could mean the range of new values that a variable can take,
        in relation to its current value. It could mean how many variables are changed by one call to the heuristic.
        For a ruin-recreate heuristic, it could mean the percentage of the solution that is destroyed and rebuilt.
        For example, a value of 0.5 may indicate that half the solution will be rebuilt by a RUIN_RECREATE heuristic.
        The meaning of this variable is intentionally vaguely stated, as it depends on the heuristic in question,
        and the problem domain in question.

        :param intensityOfMutation: must be in the range 0 to 1. The initial value of 0.1 represents the default
            operation of the low level heuristic.
        :return: None
        """
        self.session.post(self.host + "/mutationIntensity/" + self.token + "/" + str(intensityOfMutation))

    def getDepthOfSearch(self) -> float:
        """
        Gets the current intensity of mutation parameter.

        :return: the current value of the intensity of mutation parameter.
        """
        return self.session.get(self.host + "/search/depth/" + self.token).text

    def getIntensityOfMutation(self) -> float:
        """
        Gets the current intensity of mutation parameter.

        :return: the current value of the intensity of mutation parameter.
        """
        return float(self.session.get(self.host + "/mutationIntensity/" + self.token).text)

    def getHeuristicsOfType(self, heuristicType: HeuristicType) -> List[int]:
        """
        Gets an array of heuristicIDs of the type specified by heuristicType.

        :param heuristicType: the heuristic type.
        :return: A list containing the indices of the heuristics of the type specified. If there are no heuristics of
            this type it returns None.
        """
        return list(self.session.get(self.host + "/heuristic/" + self.token + "/" + str(heuristicType.name)).json())

    def getHeuristicsThatUseIntensityOfMutation(self) -> List[int]:
        """
        Gets an array of heuristicIDs that use the intensityOfMutation parameter

        :param heuristicType: the heuristic type.
        :return: An array containing the indexes of the heuristics that use the intensityOfMutation parameter, or None
            if there are no heuristics of this type.
        """
        return self.session.get(self.host + "/heuristic/mutationIntensity/" + self.token).json()

    def getHeuristicsThatUseDepthOfSearch(self) -> List[int]:
        """
        Gets an array of heuristicIDs that use the depthOfSearch parameter

        :param heuristicType: the heuristic type.
        :return: An array containing the indexes of the heuristics that use the depthOfSearch parameter, or None if
            there are no heuristics of this type.
        """
        return self.session.get(self.host + "/heuristic/depth/" + self.token).json()

    def loadInstance(self, instanceID: int) -> None:
        """
        Loads the instance specified by instanceID.

        :param instanceID: Specifies the instance to load. The ID's start at zero.
        :return: None
        """
        self.session.post(self.host + "/instance/" + self.token + "/" + str(instanceID))

    def setMemorySize(self, size: int) -> None:
        """
        Sets the size of the array where the solutions are stored. The default size is 2.

        :param size: The new size of the solution array.
        :return: None
        """
        self.session.post(self.host + "/memorySize/" + self.token + "/" + str(size))

    def initialiseSolution(self, index: int) -> None:
        """
        Create an initial solution at a specified position in the memory array. The method of initialising the solution
        depends on the specific problem domain, but it is a random process, which will produce a different solution
        each time. The initialisation process may randomise all of the elements of the problem, or it may use a
        constructive heuristic with a randomised input.

        :param index: The index of the memory array at which the solution should be initialised.
        :return: None
        """
        # raise NotImplementedError
        self.session.put(self.host + "/solution/init/" + self.token + "/" + str(index))

    def getNumberOfHeuristics(self) -> None:
        """
        Gets the number of heuristics available in this problem domain

        :return: The number of heuristics available in this problem domain
        """
        return int(self.session.get(self.host + "/heuristic/num/" + self.token).text)

    def applyHeuristicUnary(self, heuristicID: int, solutionSourceIndex: int, solutionDestinationIndex: int) -> float:
        """
        Applies the heuristic specified by heuristicID to the solution at position solutionSourceIndex and places the
        resulting solution at position solutionDestinationIndex in the solution array. If the heuristic is a
        CROSSOVER type then the solution at solutionSourceIndex is just copied to solutionDestinationIndex.

        :param heuristicID: The ID of the heuristic to apply (starts at zero)
        :param solutionSourceIndex: The index of the solution in the memory array to which to apply the heuristic
        :param solutionDestinationIndex: The index in the memory array at which to store the resulting solution
        :return: the objective function value of the solution created by applying the heuristic
        """
        return float(self.session.post(self.host + "/heuristic/apply/" + self.token + "/" + str(heuristicID) + "/" + str(
            solutionSourceIndex) + "/" + str(solutionDestinationIndex)).text)

    def applyHeuristicBinary(self, heuristicID: int, solutionSourceIndex1: int, solutionSourceIndex2: int,
                             solutionDestinationIndex: int) -> float:
        """
        Apply the heuristic specified by heuristicID to the solutions at position solutionSourceIndex1 and position
        solutionSourceIndex2 and put the resulting solution at position solutionDestinationIndex. The heuristic can
        be of any type (including CROSSOVER).

        :param heuristicID: The ID of the heuristic to apply (starts at zero)
        :param solutionSourceIndex1: The index of the first solution in the memory array to which to apply the heuristic
        :param solutionSourceIndex2: The index of the second solution in the memory array to which to apply the heuristic
        :param solutionDestinationIndex: The index in the memory array at which to store the resulting solution
        :return: the objective function value of the solution created by applying the heuristic
        """
        return float(self.session.post(self.host + "/heuristic/apply/" + self.token + "/" + str(heuristicID) + "/" + str(
            solutionSourceIndex1) + "/" + str(solutionSourceIndex2) + "/" + str(solutionDestinationIndex)).text)

    def copySolution(self, solutionSourceIndex: int, solutionDestinationIndex: int) -> None:
        """
        Copies a solution from one position in the solution array to another

        :param solutionSourceIndex: The position of the solution to copy
        :param solutionDestinationIndex: The position in the array to copy the solution to.
        :return: None
        """
        self.session.post(self.host + "/solution/copy/" + self.token + "/" + str(solutionSourceIndex) + "/" + str(
            solutionDestinationIndex))

    def toString(self) -> str:
        """
        Gets the name of the problem domain. For example, "Bin Packing"

        :return: the name of the ProblemDomain
        """
        return self.session.get(self.host + "/toString/" + self.token).text

    def getNumberOfInstances(self) -> int:
        """
        Gets the number of instances available in this problem domain

        :return: the number of instances available
        """
        return int(self.session.get(self.host + "/instances/" + self.token).text)

    def bestSolutionToString(self) -> str:
        """
        Returns the objective function value of the best solution found so far by the HyperHeuristic.

        :return: The objective function value of the best solution.
        """
        return self.session.get(self.host + "/solution/best/toString/" + self.token).text

    def getBestSolutionValue(self) -> float:
        """
        Returns the objective function value of the best solution found so far by the HyperHeuristic.

        :return: The objective function value of the best solution.
        """
        return float(self.session.get(self.host + "/solution/best/value/" + self.token).text)

    def solutionToString(self, solutionIndex: int) -> str:
        """
        Gets a String representation of a given solution in memory

        :param solutionIndex: The index of the solution of which a String representation is required
        :return: A String representation of the solution at solutionIndex in the solution memory
        """
        return self.session.get(self.host + "/solution/toString/" + self.token + "/" + str(solutionIndex)).text

    def getFunctionValue(self, solutionIndex: int) -> float:
        """
        Gets the objective function value of the solution at index solutionIndex

        :param solutionIndex: The index of the solution from which the objective function is required
        :return: A double value of the solution's objective function value.
        """
        # raise NotImplementedError
        return float(self.session.get(self.host + "/solution/functionValue/" + self.token + "/" + str(solutionIndex)).text)

    def compareSolutions(self, solutionIndex1: int, solutionIndex2: int) -> bool:
        """
        Compares the two solutions on their structure (i.e. in the solution space, not in the objective/fitness
        function space).

        :param solutionIndex1: The index of the first solution in the comparison
        :param solutionIndex2: The index of the second solution in the comparison
        :return: true if the solutions are identical, false otherwise.
        """
        return bool(self.session.get(
            self.host + "/solution/compare/" + self.token + "/" + str(solutionIndex1) + "/" + str(solutionIndex2)).text)


"""
Gym Environment for HyFlex
"""
from dacbench import AbstractEnv


class HyFlexEnv(AbstractEnv):
    """
    Environment to control the step size of CMA-ES
    """

    def __init__(self, config):
        """
        Initialize CMA Env

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super(HyFlexEnv, self).__init__(config)
        self.seed(config["seed"])

        # some useful constants
        # solution memory indices
        self.mem_size = 3
        self.s_best = 0  # mem pos for best
        self.s_inc = 1  # mem pos for incumbent
        self.s_prop = 2  # mem pos for proposal
        # actions
        self.reject = 0  # action corresponding to reject
        self.accept = 1  # action corresponding to accept

        # The following variables are (re)set at reset
        self.problem = None  # HyFlex ProblemDomain object ~ current DAC instance
        self.unary_heuristics = None  # indices for unary heuristics
        self.binary_heuristics = None  # indices for binary heuristics
        self.heuristic_indices = None  # indices for all heuristics
        self.heuristic_arities = None  # the # solutions the heuristic with index i takes as input
        self.f_best = None  # fitness of current best
        self.f_prop = None  # fitness of proposal
        self.f_inc = None  # fitness of incumbent
        self.h = None # index of the currently proposed heuristic
        self.f_delta = None # self.f_inc - self.f_prop

        # action parser
        def value_of(action):
            try:
                return action[0]
            except (TypeError, IndexError):
                return action
        if config["learn_select"] and config["learn_accept"]:
            self._parse_action = lambda action: (action[0], action[1]-1) # (nguyen) heuristic index starting from -1, while gym.spaces.Discrete starts from 0
        elif config["learn_select"]:
            self._parse_action = lambda action: (None, value_of(action)-1) # (nguyen) heuristic index starting from -1, while gym.spaces.Discrete starts from 0
        elif config["learn_accept"]:
            self._parse_action = lambda action: (value_of(action), None)
        else:
            raise Exception("No learning target: Either selection and/or acceptance must be learned!")

        # set reward function
        if "reward_function" in config.keys():
            if isinstance(config["reward_function"],str): # reward_function is the name of a function in HyFlexEnv      
                self.get_reward = getattr(HyFlexEnv, config["reward_function"])
            else: # reward function is a function pointer
                self.get_reward = config["reward_function"] 
        else:
            self.get_reward = self.get_default_reward

        # set state method
        if "state_method" in config.keys(): 
            if isinstance(config["state_method"], str) and (config["state_method"].strip()!=""): 
                try:
                    # state method is just the name of a function in HyFlexEnv
                    self.get_state = getattr(HyFlexEnv, config["state_method"])
                except:
                    # state method is described by a string listing names of features in the state. In this case we'll construct self.get_state based on the description
                    # Examples:
                    #   a state space with information of the current step only: "f_best, f_delta, f_prop, f_inc, h"
                    #   a state space with information of multiple time steps: "f_best, f_best_{t-1}, f_best_{t-4}, f_prop, f_prop_{t-1}, f_prop_{t-4}"
                    #   a state space with information of multiple time steps: "f_best_{t-0..4}, f_prop_{t-0..4}"
                    state_features = [s.strip() for s in config["state_method"].split(',')]
                    self._state_feat_funcs = self._get_state_function_from_feature_names(state_features)
                    self.get_state = self.get_state_from_features
            else: # state method is a function pointer
                self.get_state = config["state_method"]            
        else:
            self.get_state = self.get_default_state

    def _get_state_function_from_feature_names(self, features):
        state_feat_funcs = []
        for feat in features:            
            if feat in ['f_best','f_delta','f_prop','f_inc','h']: # get values at current iteration
                state_feat_funcs.append(lambda history='history_'+feat: [vars(self)[history][-1]])
            elif "_{t-" in feat: # get values from histories, e.g., "f_prop_{t-1}" will return self.history_f_best[-2], and "f_prop_{t-0..4}" will return self.history_f_best[-1:-5]
                ids = feat.split("_{t-")[1]
                name = feat.split("_{t-")[0]
                if '..' in ids: # it's a range
                    start = int(ids.split('..')[0]) + 1
                    end = int(ids.split('..')[1][:-1]) + 1
                    assert (start>=1 and start<=end and end<=HISTORY_LENGTH), "Error: " + feat + ": invalid range (must be within 0.." + str(HISTORY_LENGTH-1) + ")"
                    state_feat_funcs.append(lambda his='history_'+name: [vars(self)[his][k] for k in range(-end,-start+1)])
                else: # it's just one value
                    k = int(feat.split("_{t-")[1][:-1]) + 1                    
                    assert (k>=1 and k<=HISTORY_LENGTH), "Error: " + feat + ": invalid range (must be within 0.." + str(HISTORY_LENGTH-1) + ")"
                    assert k<=HISTORY_LENGTH, "Error: " + feat + ": index out of range history limit (" + str(HISTORY_LENGTH) + ") exceeded, please increase HISTORY_LENGTH"
                    state_feat_funcs.append(lambda his='history_'+name: [vars(self)[his][-k]])          
            else:
                raise Exception("Error: invalid state features: " + feat)
        return state_feat_funcs
        
    # TODO: to be removed
    def get_state_from_features(self):
        state_vals = np.asarray([val for f in self._state_feat_funcs for val in f()])
        if len(state_vals.shape)>1: # flatten the state if necessary
            state_vals = np.concatenate(state_vals)
        return state_vals

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
        done = super(HyFlexEnv, self).step_()

        prev_accept_action, next_select_action = self._parse_action(action)

        if prev_accept_action is None or prev_accept_action == self.accept:
            # accept previous proposal as new incumbent
            self.problem.copySolution(self.s_prop, self.s_inc)
            self.f_inc = self.f_prop

        # generate a new proposal
        self.f_prop, (nary, h)  = self._generate_proposal(next_select_action)

        # calculate reward (note: assumes f_best is not yet updated!)
        reward = self.get_reward(self)

        # update best
        self._update_best()
        
        # update self.h and self.f_delta
        self.f_delta = self.f_inc - self.f_prop
        self.h = h
        
        # update histories
        acceptance = prev_accept_action
        if acceptance is None:
            acceptance = self.accept
        self.history_acceptance.append(acceptance)
        self.history_f_best.append(self.f_best)
        self.history_f_inc.append(self.f_inc)
        self.history_f_prop.append(self.f_prop)
        self.history_f_delta.append(self.f_delta)
        self.history_h.append(h)

        # update logs
        self.log_acceptance.append(acceptance)
        self.log_f_best.append(self.f_best)
        self.log_f_inc.append(self.f_inc)
        self.log_f_prop.append(self.f_prop)
        self.log_reward.append(reward)
        self.log_h.append((nary,h))
        if isinstance(self.problem, ToyProblemDomain):
            self.log_h_str.append(ToyProblemDomain.getHeuristicDescription(h,self.f_inc))

        # info
        if done:
            info = {'f_best': self.f_best,
                    'log_acceptance': self.log_acceptance,
                    'log_f_best': self.log_f_best,
                    'log_f_inc': self.log_f_inc,
                    'log_f_prop': self.log_f_prop,
                    'log_reward': self.log_reward,                  
                    'log_h': self.log_h,
                    'log_h_str': self.log_h_str,
                    'instance': self.instance}
        else:
            info = {'f_best': self.f_best}

        return self.get_state(), reward, done, info

    def reset(self):
        """
        Reset environment

        Returns
        -------
        np.array
            Environment state
        """
        super(HyFlexEnv, self).reset_()

        domain, instance_index, seed, self.n_steps = self.instance
        # create problem domain
        if domain == "Toy":
            self.problem = ToyProblemDomain(seed)
        else:
            self.problem = HyflexProblemDomain(domain, seed)

        self.np_random = np.random.RandomState(seed)

        # classify heuristics as unary/binary
        self.unary_heuristics = self.problem.getHeuristicsOfType(H_TYPE.LOCAL_SEARCH)
        self.unary_heuristics += self.problem.getHeuristicsOfType(H_TYPE.MUTATION)
        self.unary_heuristics += self.problem.getHeuristicsOfType(H_TYPE.RUIN_RECREATE)
        self.unary_heuristics += self.problem.getHeuristicsOfType(H_TYPE.OTHER)
        self.binary_heuristics = self.problem.getHeuristicsOfType(H_TYPE.CROSSOVER)
        self.heuristic_indices = list(range(-1, len(self.unary_heuristics) + len(self.binary_heuristics)))
        self.heuristic_arities = {**{-1: 0}, 
                                  **{h: 1 for h in self.unary_heuristics},
                                  **{h: 2 for h in self.binary_heuristics}}
        # load instance
        self.problem.loadInstance(instance_index)
        # initialise solution memory
        self.problem.setMemorySize(self.mem_size)
        self.problem.initialiseSolution(self.s_inc)
        self.problem.copySolution(self.s_inc, self.s_best)
        # initialise fitness best/inc
        self.f_best = self.problem.getFunctionValue(self.s_best)
        self.f_inc = self.f_best
        # generate a proposal
        self.f_prop, (nary,h) = self._generate_proposal()
        self.h = h
                
        # update best
        self._update_best()
        
        # reset histories
        self.history_acceptance = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_f_best =  deque([self.f_best]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)  # used to be MAXINT
        self.history_f_inc =  deque([self.f_inc]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)   # used to be MAXINT
        self.history_f_prop =  deque([self.f_prop]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)  # used to be MAXINT
        self.history_f_delta = deque([0]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_h = deque([-2]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH) # history of heuristic ids being proposed, starting from -1

        # for logging
        self.log_acceptance = []
        self.log_f_best = []
        self.log_f_inc = []
        self.log_f_prop = []
        self.log_reward = []
        self.log_h = []
        self.log_h_str = []
        
        # update histories
        self.history_f_best.append(self.f_best)
        self.history_f_inc.append(self.f_inc)
        self.history_f_prop.append(self.f_prop)
        self.history_f_delta.append(self.f_inc-self.f_prop)
        self.history_h.append(h)

        # update logs
        self.log_f_best.append(self.f_best)
        self.log_f_inc.append(self.f_inc)
        self.log_f_prop.append(self.f_prop)        
        self.log_h.append((nary,h))                
        if isinstance(self.problem, ToyProblemDomain):
            self.log_h_str.append(ToyProblemDomain.getHeuristicDescription(h,self.f_inc))

        return self.get_state()

    def _generate_proposal(self, select_action=None):
        if select_action is None:
            select_action = self.np_random.choice(self.heuristic_indices)
        if self.heuristic_arities[select_action] == 0:
            self.problem.initialiseSolution(self.s_prop)
            f_prop = self.problem.getFunctionValue(self.s_prop)
        elif self.heuristic_arities[select_action] == 1:
            f_prop = self.problem.applyHeuristicUnary(select_action, self.s_inc, self.s_prop)
        else:
            # note: the best solution found thus far is used as 2nd argument for crossover
            f_prop = self.problem.applyHeuristicBinary(select_action, self.s_inc, self.s_best, self.s_prop)
        return f_prop, (self.heuristic_arities[select_action], select_action)

    def _update_best(self):
        if self.f_prop < self.f_best:
            self.problem.copySolution(self.s_prop, self.s_best)
            self.f_best = self.f_prop

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

        print("incumbent: {} \t proposed: {} \t best: {}".format(self.f_inc, self.f_prop, self.f_best))

    def get_default_reward(self, _):
        """
        Compute reward

        Returns
        -------
        float
            Reward

        """
        return max(self.f_best - self.f_prop, 0)

    def get_auc_reward(self):
        return -min(self.f_best, self.f_prop)

    def get_default_state(self, _):
        """
        Gather state description

        Returns
        -------
        dict
            Environment state

        """
        return {"f_delta": self.f_inc - self.f_prop}
