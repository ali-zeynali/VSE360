import numpy as np


class ProbDash:
    """
    Implementation of 360ProbDash algorithm
    """
    def __init__(self, video, params):
        self.video = video
        self.buffer = 0
        self.D = video.D
        self.M = len(video.values)
        self.downloaded_segments = [0 for _ in range(self.video.N)]
        self.last_finished_segments = -1
        self.buffer_capacity = params['buffer_capacity']
        self.bitrates = params['bitrates']
        self.target_buffer = params['target_buffer']
        self.eta = params['eta']
        self.all_sols = self.get_all_solutions(self.D)
        self.distorion = self.calc_distortion()

    def calculate_aggregate_rate(self, bandwidth):
        target_rate = bandwidth * (
                self.buffer * self.video.delta - self.target_buffer + self.video.delta) / self.video.delta

        target_rate = max(target_rate, self.bitrates[1] * self.D)

        return target_rate

    def calc_distortion(self):
        return [-v for v in self.video.values]

    def calc_phi(self, solution, probs):

        phi = 0
        for i in range(self.D):
            phi += probs[i] * self.distorion[solution[i]]
        return phi / self.D

    def calc_psi(self, solution, probs):
        s = 4 * np.pi / self.D
        psi = 0
        phi = self.calc_phi(solution, probs)
        for i in range(self.D):
            psi += probs[i] * s * (self.distorion[solution[i]] - phi) ** 2
        return psi / self.D

    def calc_aggregate(self, solution):
        bitrate = 0
        for i in range(self.D):
            bitrate += self.bitrates[solution[i]]
        return bitrate

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

    def get_action(self, probs, bandwidth_capacity, segment, buffer_array, downloaded_tiles, wait_time, delta):
        """
        :param probs: array size D, showing the probability of watching tiles
        :return: array size D, showing the selected bitrates of each tile
        """

        target_rate = self.calculate_aggregate_rate(bandwidth_capacity)

        min_obj = float('inf')
        best_solution = None
        for solution in self.all_sols:
            aggregate_rate = self.calc_aggregate(solution)
            if aggregate_rate > target_rate:
                continue
            phi = self.calc_phi(solution, probs)
            psi = self.calc_psi(solution, probs)
            obj = phi + self.eta * psi
            if obj < min_obj:
                min_obj = obj
                best_solution = solution

        n = 0
        for a in best_solution:
            if a > 0:
                n += 1

        if n > 0 and self.buffer_capacity - self.buffer >= n:
            type = "new"
            new_segments = n
            action_segment = segment
            return best_solution, type, action_segment, new_segments, 0
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

    def set_buffer(self, buffer):
        self.buffer = buffer

    def take_action(self, solution, n, time):

        number_of_downloaded_segments = 0
        for v in solution:
            if v > 0:
                number_of_downloaded_segments += 1

        self.downloaded_segments[n] = number_of_downloaded_segments
