import numpy as np


class Naive:
    """
    A Naive algorithm for 360ABR problem.
    It equally splits the entire expected bandidth among fixed number of tiles.
    """
    def __init__(self, video, params):
        self.video = video
        self.tile_to_download = params['tile_to_download']
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

    def get_action(self, probs, bandwidth_capacity, segment, buffer_array, downloaded_tiles, wait_time, delta):
        """
        :param probs: array size D, showing the probability of watching tiles
        :return: array size D, showing the selected bitrates of each tile
        """
        sorted_probs_indexes = len(probs) - 1 - np.argsort(probs)
        target_size = self.video.delta * bandwidth_capacity / self.tile_to_download
        sizes = self.video.sizes

        bitrate = 0
        for i in range(len(sizes) - 1):
            if sizes[i] <= target_size < sizes[i + 1]:
                bitrate = i
                break
        bitrate = max(bitrate, 1)
        solution = [0 for _ in range(self.D)]
        for i in range(self.D):
            if sorted_probs_indexes[i] < self.tile_to_download:
                solution[i] = bitrate
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
