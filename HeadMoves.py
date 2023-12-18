import json

import numpy as np


class HeadMoves:
    """
    This class simulates the user's head movements using the provided navigation graph dataset

    """
    def __init__(self, N, D, path=None):
        self.N = N #number of segments
        self.D = D #number of tiles
        self.all_probs = []
        self.path = path
        self.actual_views = []
        self.initialize_probs()

    def nav_graph(self, path):
        D = self.D
        all_probs = []
        if path is not None:
            with open(path) as file:
                raw_navigation_graph = json.load(file)

            previous_views = [self.binary_view("0001")]
            for segment_entry in raw_navigation_graph:
                f1_graph = np.zeros((D, D))
                views = [self.binary_view(hex_view) for hex_view in segment_entry['views']]

                transitions = np.array(segment_entry['transitions'], dtype=float)
                (tr_h, tr_w) = transitions.shape
                for i in range(tr_h):  # over current views
                    view = views[i]
                    for j in range(tr_w):  # over previous views
                        if transitions[i][j] > 0:
                            pr_view = previous_views[j]
                            for current_tile in range(D):  # for each tile inside current tiles
                                if view[current_tile] == "1":
                                    for pr_tile in range(D):  # for each tile inside previous tiles
                                        if pr_view[pr_tile] == "1":
                                            f1_graph[pr_tile][current_tile] += transitions[i][j]
                previous_views = views
                for i in range(D):
                    count = np.sum(f1_graph[i])
                    if count == 0:
                        f1_graph[i] = np.ones(D) / D
                    else:
                        f1_graph[i] = f1_graph[i] / count
                all_probs.append(f1_graph)
        self.all_probs = all_probs

    def binary_view(self, hex_view):
        zeros = 0
        for c in hex_view:
            if c == "0":
                zeros += 1
            else:
                break

        b_view = bin(int(hex_view, 16))[2:].zfill(8)
        return "0000" * zeros + str(b_view)

    def initialize_probs(self):
        self.nav_graph(self.path)

    def random_probs(self):
        probs = []
        for n in range(self.N):
            ps = []
            for d in range(self.D):
                ps.append(np.random.random())
            ps = np.array(ps) / np.sum(ps)
            probs.append(ps)
        self.all_probs = probs

    def bin_probs(self):
        self.random_probs()

        new_probs = []
        for prob in self.all_probs:
            n = 0
            new_ps = []
            for p in prob:
                if p >= 1 / self.D:
                    n += 1
            for p in prob:
                if p >= 1 / self.D:
                    new_ps.append(1 / n)
                else:
                    new_ps.append(0)
            new_probs.append(new_ps)
        self.all_probs = new_probs

    def set_actual_views(self, actual_views):
        self.actual_views = actual_views

    def get_sample_actual_view(self):
        act_views = []
        for n in range(self.N):
            if n == 0:
                act_view = np.random.choice(range(self.D), 1)[0]
                act_views.append(act_view)
            else:
                last_view = act_views[-1]
                probs = self.reorder_probs(self.all_probs[n][last_view])
                act_view = np.random.choice(range(self.D), 1, p=probs)[0]
                act_views.append(act_view)
        return act_views


    def reorder_probs(self, probs):
        new_probs = list(probs.copy())
        new_probs.sort(reverse=True)
        return new_probs

    def get_all_probs(self):
        all_probs = []
        n = 0
        for tile in self.actual_views:
            all_probs.append(self.reorder_probs(self.all_probs[n][tile]))
            n += 1
        return all_probs

    def get_pose_probs(self, segment_number):
        current_tile = 0 if segment_number == 0 else self.actual_views[segment_number - 1]
        return self.all_probs[segment_number][current_tile]

    def get_pose_probs_updates(self, segment_number):
        current_tile = 0 if segment_number == 0 else self.actual_views[segment_number - 1]
        probs = {}
        for i in range(segment_number + 1):
            probs[i] = self.all_probs[i][current_tile]
        return probs

    def get_pose_probs_expected(self, segment_number, playing_segment):
        current_tile = 0 if segment_number == 0 else self.actual_views[segment_number - 1]
        probs = {}
        for i in range(segment_number + 1):
            if i <= playing_segment:
                probs[i] = self.all_probs[i][current_tile]
            else:
                uniform_dist = np.array([1/self.D for _ in range(self.D)])
                err = (i - playing_segment) * 0.1
                err = min(err, 1)
                expected_probs = np.array(self.all_probs[i][current_tile]) * (1-err) + err * uniform_dist
                expected_probs = list(expected_probs)
                probs[i] = expected_probs

        return probs

    def activate_hetrogen_model(self, hetro, positive_tiles):
        if hetro < 0 or hetro > 1:
            print("Invalid hetro parameter!")
            raise Exception
        single_tile = False
        if positive_tiles <= 1:
            single_tile = True
        p_min = 0.05
        p_max = 0.95
        all_probs = []
        delta_p = (p_max - p_min) / (positive_tiles - 1) if positive_tiles > 1 else 0
        for i in range(self.N):
            probs = []
            for tile in range(self.D):
                tile_prob = []
                p_hetro = p_max
                for j in range(self.D):
                    if j > positive_tiles:
                        tile_prob.append(0)
                        continue
                    if not single_tile:
                        p = p_hetro * hetro + (1 - hetro) / (positive_tiles)
                        tile_prob.append(p)
                        p_hetro -= delta_p
                        p_hetro = max(p_hetro, 0)
                    else:
                        if j == 0:
                            tile_prob.append(1)
                        else:
                            tile_prob.append(0)
                tile_prob = np.array(tile_prob) / np.sum(tile_prob)
                probs.append(tile_prob)
            all_probs.append(probs)
        self.all_probs = all_probs

