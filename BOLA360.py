import numpy as np


class BOLA360:
    def __init__(self, video, gamma, v_coeff, buffer_capacity, beta, eta=None, BA=None):
        self.video = video
        self.gamma = gamma
        self.buffer = 0
        self.buffer_capacity = buffer_capacity
        self.D = video.D
        self.M = len(video.values)

        self.eta = 1 # for practical adjustments
        self.buffer_adjust = 1 # for practical adjustments

        self.beta = beta

        self.V = v_coeff * (video.buffer_size - self.D) / (video.values[-1] + gamma * video.delta - beta)
        self.all_sols = self.get_all_solutions(self.D)
        self.downloaded_segments = [0 for _ in range(self.video.N)]
        self.last_finished_segments = -1

    def get_idle_time(self, buffer_array, downloaded_tiles, required_space, delta):
        time_remaining = [buffer_array[i] * delta / downloaded_tiles[i] if downloaded_tiles[i] > 0 else 0 for i in
                          range(len(downloaded_tiles))]

        freed_segment = 0
        required_time = 0
        for i in range(len(downloaded_tiles)):
            if freed_segment >= required_space:
                return required_time + 0.01

            if time_remaining[i] > 0:
                freed_segment += downloaded_tiles[i]
                required_time += time_remaining[i]

    def get_action(self, probs, segment, buffer_array, downloaded_tiles, wait_time, delta, previous_history={}):
        """
        :param probs: array size D, showing the probability of watching tiles
        previous_history: includes information regarding previous actions like probabilities and bitrates downloaded
        :return: array size D, showing the selected bitrates of each tile
        """

        all_sols = self.all_sols
        solution = None
        max_rho = -1
        max_n = 0
        for sol in all_sols:
            rho = self.calc_rho(sol, probs)
            n_sols = 0
            for m in sol:
                if m > 0:
                    n_sols += 1
            if rho > max_rho:
                max_rho = rho
                solution = sol
                max_n = n_sols
            if rho == max_rho and n_sols > max_n:
                max_rho = rho
                solution = sol
                max_n = n_sols
        n = 0
        for a in solution:
            if a > 0:
                n += 1

        if n > 0 and self.buffer_capacity - self.buffer >= n:
            type = "new"
            new_segments = n
            action_segment = segment
            return solution, type, action_segment, new_segments, 0
        else:
            type = "wait"
            new_segments = 0
            action_segment = -1
            required_time = max(n - (self.buffer_capacity - self.buffer), 0)
            if required_time > 0:
                idle_time = self.get_idle_time(buffer_array, downloaded_tiles, required_time, delta)
                return [0 for _ in range(self.D)], type, action_segment, new_segments, idle_time
            else:
                return [0 for _ in range(self.D)], type, action_segment, new_segments, wait_time

    def set_buffer(self, buffer):
        self.buffer = buffer

    def take_action(self, solution, n, time):
        number_of_downloaded_segments = 0
        for v in solution:
            if v > 0:
                number_of_downloaded_segments += 1
        # self.buffer += number_of_downloaded_segments
        self.downloaded_segments[n] = number_of_downloaded_segments

    def calc_rho(self, solution, probs):
        """

        :param solution: array size D, showing the selected bitrates of each tile
        :param probs: array size D, showing the probability of watching tiles
        :return:
        """
        if np.sum(solution) == 0:
            return 0
        rho = 0
        for d in range(self.D):
            m = solution[d]
            v = self.video.values[m]
            if m > 0:
                p = min(1, self.eta * probs[d])
                rho += (self.V * (
                        p * v + self.gamma * self.video.delta - self.beta) - self.buffer * self.buffer_adjust *
                        probs[d]) / self.video.sizes[m]
                # rho += (self.V * (
                #         p * v + self.gamma * self.video.delta - self.beta) - self.buffer * self.buffer_adjust) / self.video.sizes[m]
        return rho

    def get_all_solutions(self, D):
        if D == 0:
            return [[]]
        sub_sol = self.get_all_solutions(D - 1)
        solution = []
        for v in sub_sol:
            max_last = self.M
            if len(v) > 0:
                max_last = v[-1] + 1
            for m in range(max_last):
                new_v = v.copy()
                new_v.append(m)
                solution.append(new_v)
        return solution
