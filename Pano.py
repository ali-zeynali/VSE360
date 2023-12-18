import numpy as np


class Pano:
    """
    Implementation of Pano algoerithm
    """
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
        time_remaining = [buffer_array[i] * delta / downloaded_tiles[i] if downloaded_tiles[i] > 0 else 0 for i in
                          range(len(downloaded_tiles))]
        time_remaining = np.sum(time_remaining)

        max_size = time_remaining * bandwidth_capacity

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

        solution = [1 for _ in range(self.D)]
        size = sizes[0] * self.D
        diff_size = [sizes[a+1] - sizes[a] if a < len(sizes) - 1 else float('inf') for a in solution]
        M_qs = [(sizes[-1] / delta - sizes[a] / delta) ** 2 for a in range(len(sizes))]
        while size <= max_size - min(diff_size):
            M_changes = [sizes[a+1]*M_qs[a+1] - sizes[a]*M_qs[a] if a < len(sizes) - 1 else 0 for a in solution]
            min_q = float('inf')
            best_tile = -1
            for tile in range(self.D):
                if M_changes[tile] < min_q and size + diff_size[tile] <= max_size:
                    min_q = M_changes[tile]
                    best_tile = tile
            if best_tile == -1:
                break
            solution[best_tile] += 1
            size += sizes[solution[best_tile]] - sizes[solution[best_tile] - 1]
            diff_size = [sizes[a + 1] - sizes[a] if a < len(sizes) - 1 else float('inf') for a in solution]


        n = np.sum([1 if a > 0 else 0 for a in solution])
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

        self.downloaded_segments[n] = number_of_downloaded_segments
