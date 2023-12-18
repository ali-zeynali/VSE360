import numpy as np


class Bola3d:
    def __init__(self, video, gamma, v_coeff, buffer_capacity):
        self.video = video
        # self.head_move = head_move
        self.gamma = gamma
        self.buffer = 0
        self.buffer_capacity = buffer_capacity
        self.D = video.D
        self.M = len(video.values)
        self.V = v_coeff * (video.buffer_size - self.D) / (video.values[-1] + gamma * video.delta)
        self.all_sols = self.get_all_solutions(self.D)
        self.downloaded_segments = [0 for _ in range(self.video.N)]
        self.last_finished_segments = -1

    def get_action(self, probs):
        """
        :param probs: array size D, showing the probability of watching tiles
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
        if self.buffer_capacity - self.buffer >= n:
            return solution
        else:
            return [0 for _ in range(self.D)]

    def set_buffer(self, buffer):
        self.buffer = buffer
    def take_action(self, solution, n, time):
        # finished_segments = int(time / self.video.delta)
        # finished_segments = min(finished_segments, n) # consider rebuff
        # for i in range(self.last_finished_segments, min(finished_segments, n + 1)):
        #     if i >= 0:
        #         self.buffer -= self.downloaded_segments[i]
        # self.last_finished_segments = finished_segments
        # self.buffer = max(self.buffer, 0)
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
        rho2 = 0
        for d in range(self.D):
            m = solution[d]
            v = self.video.values[m]
            a = 0
            if m > 0:
                a = 1
            rho += probs[d] * a * (self.V * (v + self.gamma * self.video.delta) - self.buffer)
            rho2 += self.video.sizes[m] * a
        return rho / rho2

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
