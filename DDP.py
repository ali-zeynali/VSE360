from Bandwidth import *


class DDP:
    def __init__(self, video, buffer_size, bandwidth, gamma, beta, t_0=1):
        self.video = video
        self.buffer_size = buffer_size * self.video.delta
        # self.head_moves = head_moves
        self.D = video.D
        self.bandwidth = bandwidth
        self.t_0 = t_0
        self.T0_max = 20
        self.Nb_max = 50
        self.N = video.N
        self.rewards = {}
        for i in range(self.N + 1):
            self.rewards[i] = {}
        self.rewards[0][0] = {}
        self.rewards[0][0][0] = 0


        self.beta = beta
        self.solutions = [None for _ in range(self.N)]
        self.Time = [float('inf') for _ in range(self.N)]
        self.M = len(video.values)
        self.gamma = gamma
        self.all_sols = self.get_all_solutions(self.D)

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

    def get_total_size(self, sizes):
        # fixed
        total_size = 0
        n = 0
        for m in sizes:
            total_size += self.video.sizes[m]
            if m > 0:
                n += 1
        return total_size, n

    def save_info(self, path="results/DDP_1.json"):
        data = {}
        data['solution'] = self.solutions
        data['time'] = self.Time
        data['reward'] = self.get_optimal_reward()
        with open(path, 'w') as writer:
            json.dump(data, writer)

    def train(self, headmovement):
        for n in range(1, self.N + 1):
            print("Calculating Optimal solution for: n = {1} / {0}".format(self.N, n))
            for tp in self.rewards[n - 1]:
                buffers = self.rewards[n - 1][tp].keys()
                if len(buffers) == 0:
                    print("No buffer found for n{0}, t: {1}".format(n, tp))
                for bp in self.rewards[n - 1][tp]:
                    all_sols = self.all_sols
                    for m in all_sols:
                        if np.sum(m) == 0:
                            continue
                        total_size, download_n = self.get_total_size(m)
                        x = self.bandwidth.download_time(total_size, tp * self.t_0)
                        xp = max(x, bp * self.t_0 + download_n * self.video.delta - self.buffer_size * self.video.delta)
                        # x0 = int(xp / self.t_0) * self.t_0

                        y = max(xp - bp * self.t_0, 0)
                        t = tp * self.t_0 + xp

                        b = max(bp * self.t_0 - xp + y + download_n * self.video.delta, 0)

                        rp = self.rewards[n - 1][tp][bp]
                        probs = headmovement.get_pose_probs(n-1)
                        for i in range(self.D):
                            if m[i] > 0:
                                t_ratio = self.video.sizes[m[i]] / total_size
                                time_i = t_ratio * x
                                rp += (self.video.values[m[i]] * probs[i] - self.beta + self.gamma * self.video.delta) / time_i

                        t_index = int(t / self.t_0) * self.t_0
                        b_index = int(b / self.t_0) * self.t_0
                        if not self.has_values(n, t_index, b_index) or rp > self.rewards[n][t_index][b_index]:
                            self.solutions[n - 1] = m
                            self.Time[n - 1] = t
                            self.update_val(n, t_index, b_index, rp)

    def has_values(self, n, t, b):
        if t in self.rewards[n]:
            if b in self.rewards[n][t]:
                return True
        return False

    def update_val(self, n, t_index, b_index, r):
        if t_index not in self.rewards[n]:
            self.rewards[n][t_index] = {}
            self.rewards[n][t_index][b_index] = r
        else:
            self.rewards[n][t_index][b_index] = r

    def get_optimal_solutions(self):
        return self.solutions

    def get_optimal_reward(self):
        max_rev = -1
        for t in self.rewards[self.N]:
            buffers = self.rewards[self.N][t].keys()
            for b in buffers:
                if self.rewards[self.N][t][b] > max_rev:
                    max_rev = self.rewards[self.N][t][b]
        return max_rev
