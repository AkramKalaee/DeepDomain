import time

class sw(object):

    def __init__(self):
        self.__Positions = []
        self.__Gbest = []
        self.__F = []

    def _set_Gbest(self, Gbest):
        self.__Gbest = Gbest

    def _points(self, agents):
        self.__Positions.append([list(i) for i in agents])

    def _out(self, F):
        self.__F.append(F)

    def get_agents(self):
        """Returns a history of all agents of the algorithm (return type:
        list)"""

        return self.__Positions

    def get_F(self):
        return self.__F[-1]

    def get_Gbest(self):
        """Return the best position of algorithm (return type: list)"""

        return list(self.__Gbest)

    def get_history(self):
        return self.__Positions, self.__F

class SwarmTerminationCondition:
    def __init__(self, max_iterations, timeout, stagnation_threshold, num_stagnation_iterations, save_path):
        self.max_iterations = max_iterations
        self.timeout = timeout
        # self.min_acceptable_fitness = min_acceptable_fitness
        self.stagnation_threshold = stagnation_threshold
        self.num_stagnation_iterations = num_stagnation_iterations
        self.start_time = time.time()
        self.best_fitness = None
        self.last_improvement = 0
        self.save_path = save_path

    def check(self, current_iteration, current_fitness):
        if self.best_fitness is not None:
            delta = current_fitness - self.best_fitness
            with open(f"{self.save_path}/metric_generation.txt", "a") as file:
                file.write(str(delta) + ",")
                file.close()

        # # Check if maximum number of iterations has been reached
        # if current_iteration >= self.max_iterations:
        #     return True

        # Check if timeout period has been reached
        if time.time() - self.start_time > self.timeout:
            return True

        # # Check if minimum acceptable fitness level has been reached
        # if current_fitness >= self.min_acceptable_fitness:
        #     return True

        # Check for stagnation in improvement
        if self.best_fitness is not None and delta < self.stagnation_threshold:
            self.last_improvement += 1
            if self.last_improvement >= self.num_stagnation_iterations:
                return True
        else:
            self.best_fitness = current_fitness
            self.last_improvement = 0

        return False

class RL(object):
    pass

class RLTerminationCondition:
    def __init__(self, max_iterations, timeout, stagnation_threshold, num_stagnation_iterations, save_path):
        self.max_iterations = max_iterations
        self.timeout = timeout
        # self.min_acceptable_fitness = min_acceptable_fitness
        self.stagnation_threshold = stagnation_threshold
        self.num_stagnation_iterations = num_stagnation_iterations
        self.start_time = time.time()
        self.best_fitness = None
        self.last_improvement = 0
        self.save_path = save_path

    def check(self, current_iteration, current_fitness):
        if self.best_fitness is not None:
            delta = current_fitness - self.best_fitness
            with open(f"{self.save_path}/metric_generation.txt", "a") as file:
                file.write(str(delta) + ",")
                file.close()

        # # Check if maximum number of iterations has been reached
        # if current_iteration >= self.max_iterations:
        #     return True

        # Check if timeout period has been reached
        if time.time() - self.start_time > self.timeout:
            return True

        # # Check if minimum acceptable fitness level has been reached
        # if current_fitness >= self.min_acceptable_fitness:
        #     return True

        # Check for stagnation in improvement
        if self.best_fitness is not None and delta < self.stagnation_threshold:
            self.last_improvement += 1
            if self.last_improvement >= self.num_stagnation_iterations:
                return True
        else:
            self.best_fitness = current_fitness
            self.last_improvement = 0

        return False