import numpy as np


class Mosaic:
    """
    Implementation of Mosaic algorithm
    """
    def __init__(self, video, params):
        self.video = video
        self.buffer = 0
        self.D = video.D
        self.M = len(video.values)
        self.downloaded_segments = [0 for _ in range(self.video.N)]
        self.last_finished_segments = -1
        self.buffer_capacity = params['buffer_capacity']
        self.mu1 = params['mu1']
        self.mu2 = params['mu2']
        self.mu3 = params['mu3']
        self.mu4 = params['mu4']
        self.last_Q = None

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

        return required_time + 0.01

    def get_Bj_Sj_Ej(self, solution, bitrates, probs):
        selected_rates = [bitrates[a] for a in solution]
        B = np.array(selected_rates).dot(probs)
        S = np.sum([0 if solution[i] > 0 else probs[i] for i in range(self.D)]) * bitrates[1]

        E = np.std(np.array(selected_rates) * probs)

        return B, S, E

    def get_Ti(self, solution, sizes, bandwidth_capacity, time_rem):
        size = np.sum([sizes[a] for a in solution])
        download_time = size / bandwidth_capacity

        if download_time < time_rem:
            return 0
        else:
            return download_time - time_rem

    def get_QoE(self, solution, sizes, delta, bandwidth_capacity, probs, time_rem):
        B, S, E = self.get_Bj_Sj_Ej(solution, np.array(sizes) / delta, probs)

        Q = 0.9 * B - 0.3 * S

        T = self.get_Ti(solution, sizes, bandwidth_capacity, time_rem)

        G = abs(Q - self.last_Q) if self.last_Q is not None else 0
        QoE = self.mu1 * Q - self.mu2 * T - self.mu3 * G * self.mu4 * E

        return QoE

    def get_action(self, probs, bandwidth_capacity, segment, buffer_array, downloaded_tiles, wait_time, delta):
        """
        :param probs: array size D, showing the probability of watching tiles
        :return: array size D, showing the selected bitrates of each tile
        """
        time_remaining = [buffer_array[i] * delta / downloaded_tiles[i] if downloaded_tiles[i] > 0 else 0 for i in
                          range(len(downloaded_tiles))]
        time_remaining = np.sum(time_remaining)

        max_QoE = float("-inf")
        solution = [0 for _ in range(self.D)]
        C = bandwidth_capacity * max(delta, time_remaining * 0.5)
        for sol in self.get_all_solutions(self.D):
            QoE = self.get_QoE(sol, self.video.sizes, delta, bandwidth_capacity, probs, time_remaining)
            sol_size = np.sum([self.video.sizes[a] for a in solution])
            if QoE > max_QoE and sol_size < C:
                solution = sol
                max_QoE = QoE

        n = np.sum([1 if a > 0 else 0 for a in solution])

        if n == 0:
            type = "new"
            new_segments = 1
            action_segment = segment
            solution = [0 for _ in range(self.D)]
            solution[0] = 1
            self.last_Q = self.get_QoE(solution, self.video.sizes, delta, bandwidth_capacity, probs, time_remaining)
            return solution, type, action_segment, new_segments, 0

        elif self.buffer_capacity - self.buffer >= n:
            type = "new"
            new_segments = n
            action_segment = segment
            self.last_Q = max_QoE
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

    def get_all_solutions(self, D):
        if D == 0:
            return [[]]
        sub_sol = self.get_all_solutions(D - 1)
        solution = []
        for v in sub_sol:
            max_last = self.M
            if len(v) > 0:
                max_last = v[-1] + 1
            for m in range(1, max_last):
                new_v = v.copy()
                new_v.append(m)
                solution.append(new_v)
        return solution

    def take_action(self, solution, n, time):

        number_of_downloaded_segments = 0
        for v in solution:
            if v > 0:
                number_of_downloaded_segments += 1

        self.downloaded_segments[n] = number_of_downloaded_segments
