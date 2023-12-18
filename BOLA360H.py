import numpy as np


class BOLA360H:
    def __init__(self, video, gamma, v_coeff, buffer_capacity, beta, heuristic="BASIC"):
        self.video = video
        # self.head_move = head_move
        self.gamma = gamma
        self.buffer = 0
        self.buffer_capacity = buffer_capacity
        self.D = video.D
        self.M = len(video.values)

        # adjustments
        self.eta = 1  # was 3
        self.buffer_adjust = 1

        self.beta = beta

        # self.V = v_coeff * (video.buffer_size - self.D / 2) / (video.values[-1] * self.eta + gamma * video.delta)
        # self.V = max(self.V, (video.buffer_size - self.D) / (self.gamma * video.delta * self.buffer_adjust))
        self.V = v_coeff * (buffer_capacity - self.D) / (video.values[-1] + gamma * video.delta - beta)
        self.all_sols = self.get_all_solutions(self.D)
        self.downloaded_segments = [0 for _ in range(self.video.N)]
        self.last_finished_segments = -1
        self.heuristic = heuristic

    def get_action(self, probs, segment, buffer_array, downloaded_tiles, wait_time, delta, expected_bandwidth, params,
                   history={}):
        if self.heuristic == "BASIC":
            return self.get_bola_action(probs, segment, buffer_array, downloaded_tiles, wait_time, delta,
                                        previous_history=history)
        elif self.heuristic == "REP":
            return self.get_replacement_action(probs, segment, buffer_array, downloaded_tiles, wait_time, delta,
                                               previous_history=history)
        elif self.heuristic == "PL":
            return self.get_pl_action(probs, segment, buffer_array, downloaded_tiles, wait_time, delta,
                                      expected_bandwidth)
        elif self.heuristic == "PLL":
            return self.get_pl_limited_action(probs, segment, buffer_array, downloaded_tiles, wait_time, delta,
                                              expected_bandwidth)
        elif self.heuristic == "PA":
            return self.get_pa_action(probs, segment, buffer_array, downloaded_tiles, wait_time, delta,
                                      expected_bandwidth)
        elif self.heuristic == "BE":
            if 'BE_Expansion' in params:
                expanstion = params['BE_Expansion']
            else:
                expanstion = None
            return self.get_be_action(probs, segment, buffer_array, downloaded_tiles, wait_time, delta,
                                      expected_bandwidth,
                                      expansion=expanstion)
        elif self.heuristic == "THR":
            if 'THR_threshold_buffer' in params:
                threshold_buffer = params['THR_threshold_buffer']
            else:
                threshold_buffer = None
            return self.get_throuput_action(probs, segment, buffer_array, downloaded_tiles, wait_time, delta,
                                            expected_bandwidth, threshold_buffer=threshold_buffer)

        else:
            return self.get_bola_action(probs, segment, buffer_array, downloaded_tiles, wait_time, delta,
                                        previous_history=history)

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

    def throuput_based(self, probs, expected_bandwidth):
        action = []
        for i in range(self.D):
            size_portion = expected_bandwidth * probs * self.video.delta / 4
            a = np.argmax(np.array(self.video.sizes) > size_portion)
            a = max(a, 1)
            a = int(a)
            action.append(a)
        return action

    def start_or_seek_happened(self):
        buffer_threshold = self.buffer_capacity * 0.65
        if 1 < self.buffer < buffer_threshold:
            return True
        return False

    # def get_pl_action(self, probs, segment, buffer_array, downloaded_tiles, wait_time, delta, expected_bandwidth):
    #     # Place Holder
    #
    #     if not self.start_or_seek_happened():
    #         return self.get_bola_action(probs, segment, buffer_array, downloaded_tiles, wait_time, delta)
    #
    #     actual_buffer = self.buffer
    #
    #     self.buffer = self.buffer_capacity / 2
    #     action, type, action_segment, new_segments, waiting = self.get_bola_action(probs, segment, buffer_array,
    #                                                                                downloaded_tiles, wait_time,
    #                                                                                delta)
    #     self.buffer = actual_buffer
    #     return action, type, action_segment, new_segments, waiting

    def get_pa_action(self, probs, segment, buffer_array, downloaded_tiles, wait_time, delta, expected_bandwidth):

        safe_download_size = expected_bandwidth * 0.5 * self.buffer * self.video.delta

        prob_adjust_factor = 1

        while True:
            solution, type, action_segment, new_segments, wait_time = self.get_bola_action(probs, segment, buffer_array,
                                                                                           downloaded_tiles, wait_time,
                                                                                           delta)
            size = np.sum([self.video.sizes[a] for a in solution])

            if size >= safe_download_size or new_segments == 0 or prob_adjust_factor > 3:
                return solution, type, action_segment, new_segments, wait_time

            prob_adjust_factor += 0.2
            probs = [min(1, p * prob_adjust_factor) for p in probs]

    def get_pl_action(self, probs, segment, buffer_array, downloaded_tiles, wait_time, delta, expected_bandwidth):
        # Place Holder
        pl_delta = 0.5

        if not self.start_or_seek_happened():
            return self.get_bola_action(probs, segment, buffer_array, downloaded_tiles, wait_time, delta)

        actual_buffer = self.buffer

        pl_threshold = expected_bandwidth / 3
        pl_threshold = max(pl_threshold, self.video.sizes[1] * self.D / self.video.delta)
        while True:
            action, type, action_segment, new_segments, waiting = self.get_bola_action(probs, segment, buffer_array,
                                                                                       downloaded_tiles, wait_time,
                                                                                       delta)

            aggregate_bitrate = np.sum([self.video.sizes[a] for a in action]) / self.video.delta
            if aggregate_bitrate > pl_threshold or self.buffer / self.buffer_capacity > 0.85 or new_segments == 0:
                self.buffer = actual_buffer
                return action, type, action_segment, new_segments, waiting
            self.buffer += pl_delta

    def get_pl_limited_action(self, probs, segment, buffer_array, downloaded_tiles, wait_time, delta,
                              expected_bandwidth):
        # Place Holder with limit on bitrate
        pl_delta = 0.5

        if not self.start_or_seek_happened():
            return self.get_bola_action(probs, segment, buffer_array, downloaded_tiles, wait_time, delta)

        safe_download_size = expected_bandwidth * 0.5 * self.buffer * self.video.delta
        safe_download_size = max(safe_download_size, self.D * self.video.sizes[1])

        actual_buffer = self.buffer

        pl_threshold = expected_bandwidth / 3
        pl_threshold = max(pl_threshold, self.video.sizes[1] * self.D / self.video.delta)

        pl_threshold = max(pl_threshold,
                           safe_download_size)  # limiting the maximum aggregate bitrate according to expected bandwidth
        while True:
            action, type, action_segment, new_segments, waiting = self.get_bola_action(probs, segment, buffer_array,
                                                                                       downloaded_tiles, wait_time,
                                                                                       delta)

            aggregate_bitrate = np.sum([self.video.sizes[a] for a in action]) / self.video.delta
            if aggregate_bitrate > pl_threshold or self.buffer / self.buffer_capacity > 0.85 or new_segments == 0:
                self.buffer = actual_buffer
                return action, type, action_segment, new_segments, waiting
            self.buffer += pl_delta

    def get_be_action(self, probs, segment, buffer_array, downloaded_tiles, wait_time, delta, expected_bandwidth,
                      expansion=None):
        # Buffer Expanded

        safe_download_size = expected_bandwidth * 0.5 * self.buffer * self.video.delta
        safe_download_size = max(safe_download_size, self.D * self.video.sizes[1])

        if expansion is None:
            expansion = self.video.D * 8

        self.buffer += expansion
        self.buffer_capacity += expansion

        self.V *= (self.buffer_capacity - self.D) / (self.buffer_capacity - expansion - self.D)

        action, type, action_segment, new_segments, waiting = self.get_bola_action(probs, segment, buffer_array,
                                                                                   downloaded_tiles, wait_time, delta)

        self.V *= (self.buffer_capacity - expansion - self.D) / (self.buffer_capacity - self.D)

        self.buffer -= expansion
        self.buffer_capacity -= expansion

        size = np.sum([self.video.sizes[a] for a in action])

        n = 0
        for a in action:
            if a > 0:
                n += 1

        if size > safe_download_size:

            required_space = max(n - (self.buffer_capacity - self.buffer), 0)

            if required_space > 0:
                type = "wait"
                new_segments = 0
                action_segment = -1
                idle_time = self.get_idle_time(buffer_array, downloaded_tiles, required_space, delta)
                return [0 for _ in range(self.D)], type, action_segment, new_segments, idle_time
            else:
                return [1 for _ in range(self.D)], type, action_segment, new_segments, 0
        else:
            return action, type, action_segment, new_segments, waiting

    def get_replacement_action(self, probs, segment, buffer_array, downloaded_tiles, wait_time, delta,
                               previous_history={}):
        danger_zone = 2 * self.video.delta

        time_remaining = [buffer_array[i] * delta / downloaded_tiles[i] if downloaded_tiles[i] > 0 else 0 for i in
                          range(len(downloaded_tiles))]

        if np.sum(time_remaining) < danger_zone:
            return self.get_bola_action(probs, segment, buffer_array, downloaded_tiles, wait_time, delta)

        aggregate_buffer = [time_remaining[0]]
        for i in range(1, len(time_remaining)):
            aggregate_buffer.append(time_remaining[i] + aggregate_buffer[i - 1])

        safe_segment_index = np.argmax(np.array(aggregate_buffer) >= danger_zone)

        for seg in range(safe_segment_index, segment):
            probs = previous_history[seg]['probs']
            bitrates = previous_history[seg]['bitrate']
            for tile in range(self.D):
                tile_sol = self.get_tile_sol(probs[tile])
                if bitrates[tile] < tile_sol - 1:
                    type = "replacement"
                    new_segments = 1
                    action_segment = seg
                    action = [0 for _ in range(self.D)]
                    action[tile] = tile_sol
                    return action, type, action_segment, new_segments, 0
        return self.get_bola_action(probs, segment, buffer_array, downloaded_tiles, wait_time, delta)

    def get_throuput_action(self, probs, segment, buffer_array, downloaded_tiles, wait_time, delta, expected_bandwidth,
                            threshold_buffer=None):
        if threshold_buffer is None:
            threshold_buffer = self.D * 1.5
        if 1 < self.buffer < threshold_buffer:
            action = self.throuput_based(probs, expected_bandwidth)
            n = 0
            for a in action:
                if a > 0:
                    n += 1
            type = "new"
            new_segments = n
            action_segment = segment
            return action, type, action_segment, new_segments, 0

        action, type, action_segment, new_segments, waiting = self.get_bola_action(probs, segment, buffer_array,
                                                                                   downloaded_tiles, wait_time, delta)

        size = np.sum([self.video.sizes[a] for a in action])

        safe_size = self.video.delta * expected_bandwidth / 3

        if size < safe_size:
            action = self.throuput_based(probs, expected_bandwidth)
            n = 0
            for a in action:
                if a > 0:
                    n += 1

            if n > 0 and self.buffer_capacity - self.buffer >= n:

                type = "new"
                new_segments = n
                action_segment = segment
                return action, type, action_segment, new_segments, 0
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
        else:
            return action, type, action_segment, new_segments, waiting

    def get_bola_action(self, probs, segment, buffer_array, downloaded_tiles, wait_time, delta, previous_history={}):
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

    def get_lazy_bola_action(self, probs, segment, buffer_array, downloaded_tiles, wait_time, delta, recent_actions,
                             previous_history={}, lazyness=0.1):
        """
        :param probs: array size D, showing the probability of watching tiles
        previous_history: includes information regarding previous actions like probabilities and bitrates downloaded
        :return: array size D, showing the selected bitrates of each tile
        """

        solution, type, action_segment, new_segments, wait_time = self.get_bola_action(probs, segment, buffer_array,
                                                                                       downloaded_tiles, wait_time,
                                                                                       delta,
                                                                                       previous_history=previous_history)
        if type != "new":
            return solution, type, action_segment, new_segments, wait_time
        actual_buffer = self.buffer
        self.buffer = max(0, self.buffer * (1 - lazyness))
        lower_solution, lower_type, lower_action_segment, lower_new_segments, lower_wait_time = self.get_bola_action(
            probs, segment, buffer_array, downloaded_tiles, wait_time, delta,
            previous_history=previous_history)

        self.buffer = min(self.buffer_capacity, self.buffer * (1 + lazyness))
        higher_solution, higher_type, higher_action_segment, higher_new_segments, higher_wait_time = self.get_bola_action(
            probs, segment, buffer_array, downloaded_tiles, wait_time, delta,
            previous_history=previous_history)
        self.buffer = actual_buffer
        final_actions = []
        for i in range(len(solution)):
            if recent_actions[i] == solution[i]:
                final_actions.append(solution[i])
            elif recent_actions[i] < solution[i]:
                if lower_solution[i] == recent_actions[i]:
                    final_actions.append(lower_solution[i])
                else:
                    final_actions.append(solution[i])
            else:
                if higher_solution[i] == recent_actions[i]:
                    final_actions.append(higher_solution[i])
                else:
                    final_actions.append(solution[i])

        return final_actions, type, action_segment, new_segments, wait_time



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
        return rho

    def get_tile_sol(self, prob):
        max_rho = 0
        best_sol = 0
        for sol in range(self.M):
            v = self.video.values[sol]
            rho = (self.V * (
                    prob * v + self.gamma * self.video.delta - self.beta) - self.buffer * self.buffer_adjust *
                   prob) / self.video.sizes[sol]
            if rho > max_rho:
                max_rho = rho
                best_sol = sol

        return sol

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
