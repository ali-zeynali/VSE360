from Bandwidth import *


class DDPOnline:
    def __init__(self, video, params, t_0=1):
        self.video = video
        self.buffer_size = params['buffer_size'] * self.video.delta
        self.D = video.D
        self.bandwidth = params['bandwidth']
        self.t_0 = t_0
        self.solutions = None
        self.M = len(video.values)
        self.gamma = params['gamma']
        self.all_sols = self.get_all_solutions(self.D)
        self.buffer_capacity = params['buffer_size']
        self.buffer = 0
        self.downloaded_segments = [0 for _ in range(self.video.N)]
        self.beta = params['beta']

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

    def get_total_size(self, sol):
        total_size = 0
        n = 0
        for m in sol:
            total_size += self.video.sizes[m]
            if m > 0:
                n += 1
        return total_size, n

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

    def get_action(self, probs, t, segment, buffer_array, downloaded_tiles, wait_time, delta):
        self.solutions = None
        all_sols = self.all_sols
        max_r = 0
        for m in all_sols:
            if np.sum(m) == 0:
                continue
            total_size, n = self.get_total_size(m)
            x = self.bandwidth.expected_download_time(total_size, t)
            # xp = max(x, b + n * self.video.delta - self.buffer_size)
            # x0 = np.ceil(xp / self.t_0) * self.t_0
            # y = max(x0 - b, 0)
            # tp = t + x0
            # bp = b - x0 + y + n * self.video.delta
            rp = 0
            for i in range(self.D):
                if m[i] > 0:
                    t_ratio = self.video.sizes[m[i]] / total_size
                    time_i = t_ratio * x
                    rp += self.gamma * n * self.video.delta / time_i
                    rp += (self.video.values[m[i]] * probs[i] - self.beta) / time_i

            temp = 0
            if rp > max_r:
                self.solutions = m
                max_r = rp

        if self.solutions == None:
            type = "wait"
            new_segments = 0
            action_segment = -1
            return [0 for _ in range(self.D)], type, action_segment, new_segments
        else:
            n = 0
            for a in self.solutions:
                if a > 0:
                    n += 1
            if n > 0 and self.buffer_capacity - self.buffer >= n:
                type = "new"
                new_segments = n
                action_segment = segment
                return self.solutions, type, action_segment, new_segments, 0
            else:
                type = "wait"
                new_segments = 0
                action_segment = -1
                required_time = max(n - (self.buffer_capacity - self.buffer), 0)
                if required_time > 0 :
                    idle_time = self.get_idle_time(buffer_array, downloaded_tiles,
                                               required_time, delta)
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
