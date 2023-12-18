import numpy as np


class Flare:
    def __init__(self, video, params):
        self.video = video
        self.buffer = 0
        self.D = video.D
        self.M = len(video.values)
        self.downloaded_segments = [0 for _ in range(self.video.N)]
        self.last_finished_segments = -1
        self.buffer_capacity = params['buffer_capacity']

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

    def get_action(self, probs, bandwidth_capacity, segment, buffer_array, downloaded_tiles, wait_time, delta):
        """
        :param probs: array size D, showing the probability of watching tiles
        :return: array size D, showing the selected bitrates of each tile
        """
        o = self.buffer / ((self.video.N - segment) * self.D)
        ksi = o * (0.6 - 0.3) + 0.3

        time_remaining = [buffer_array[i] * delta / downloaded_tiles[i] if downloaded_tiles[i] > 0 else 0 for i in
                          range(len(downloaded_tiles))]
        time_remaining = np.sum(time_remaining)

        max_size = time_remaining * bandwidth_capacity * ksi

        if max_size <= 0:
            solution = [1 for _ in range(self.D)]
            n = self.D
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

        sizes = self.video.sizes

        ######################
        solution = [0 for _ in range(self.D)]
        all_sols = self.get_all_solutions(self.D)
        sols = []
        bitrates = []
        for sol in all_sols:
            size = np.array([sizes[a] for a in sol])
            if np.sum(size) <= max_size:
                sols.append(sol)
                bitrates.append(size/delta)

        if len(sols) > 0:
            utilities = np.array(bitrates).dot(probs)
            solution = sols[np.argmax(utilities)]










        #######################
        n = np.sum([1 if solution[i] > 0 else 0 for i in range(self.D)])
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


    def evaluate_sol(self, sol, max_size, probs):
        size = np.sum([self.video.sizes[a] for a in sol])
        #satisfying the constraint
        if size > max_size:
            return float('-inf')

        utility = [self.video.sizes[sol[i]] * probs[i] for i in range(len(sol))]
        return np.sum(utility)


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
