import json

from ChromeDownloader import *


class Bandwidth:
    """
    This class is simulating the bandwidth network simulation using the provided input format.

    Please update this file based on format of your dataset.
    """

    def __init__(self, path_dir, error_rate, target_average=9, series=1, __real_downloader__=False):

        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.delta = 0.01
        self.error_rate = error_rate
        self.current_tp = np.average([0, 0])
        self.throuput_changes = []  # (time, val)
        self.last_updated_time = 0
        self.avg_thr = 0

        if series == 2:
            adjust = self.read_dataset_series2(path_dir, adjust=0)
            self.read_dataset_series2(path_dir, adjust=adjust)
        elif series == 3:
            self.set_constant_bandwidth(target_average)
        else:
            self.read_dataset(path_dir)
        self.calculated_download_time = {}
        self.check_avg(target_avg=target_average)
        if __real_downloader__:
            self.chrome_downloader = ChromeDownloader()  # using chrome Downloader for real downloading simulation.

    def check_avg(self, target_avg=9):
        for i in range(len(self.throuput_changes)):
            self.throuput_changes[i] = (
                self.throuput_changes[i][0], self.throuput_changes[i][1] * target_avg / self.avg_thr)

    def set_constant_bandwidth(self, target_value):
        self.throuput_changes.append((0, target_value))
        self.avg_thr = target_value
        self.last_updated_time = 0

    def read_dataset(self, path_dir):
        with open(path_dir, 'r') as reader:
            dataset = json.load(reader)
        time = 0
        aggr_thr = 0
        for tuple in dataset['results']:
            dlrate = tuple['dlRate'] / 1000
            self.throuput_changes.append((time, dlrate))
            aggr_thr += dlrate
            self.last_updated_time = max(self.last_updated_time, time)
            time += tuple['dlDuration']
            if self.min_val > dlrate:
                self.min_val = dlrate
            if self.max_val < dlrate:
                self.min_val = dlrate
        self.current_tp = self.throuput_changes[0]
        self.avg_thr = aggr_thr / len(self.throuput_changes)

    def read_dataset_series2(self, path, adjust=0):
        with open(path, 'r') as reader:
            lines = reader.readlines()
            times = []
            thrs = []
            time = 0
            aggr_thr = 0
            for line in lines:
                line.replace("\n", "")
                line = line.split(" ")
                times.append(time)
                time += float(line[-1])
                thr = float(line[-2]) * 8 / (10 ** 3 * float(line[-1]))
                thrs.append(thr)  # convert Bps to Mbps
                aggr_thr += thr
            times = np.array(times) - times[0]
            self.last_updated_time = times[-1] / 1000 + adjust
            for i in range(len(times)):
                self.throuput_changes.append((times[i] / 1000 + adjust, thrs[i] / 3))
            self.avg_thr = aggr_thr / len(self.throuput_changes)
        return times[-1] / 1000

    def get_thr(self, time):
        if time > self.last_updated_time:
            return self.throuput_changes[-1][1]
        for (t, thr) in self.throuput_changes:
            if time <= t:  # TODO: need to update this, thr must update after thrigger not before it
                return thr
        return self.throuput_changes[-1][1]

    def activate_synthetic_thr_1(self):
        """
        A sample of synthetic network throughput
        :return:
        """
        thrs = []
        tiles = 6
        alpha = 10
        thrs.append((20, 0.4 * tiles * alpha))
        thrs.append((50, 0.9 * tiles * alpha))
        thrs.append((100, 0.7 * tiles * alpha))
        thrs.append((120, 0.6 * tiles * alpha))
        thrs.append((150, 1 * tiles * alpha))
        thrs.append((170, 0.7 * tiles * alpha))
        thrs.append((200, 0.5 * tiles * alpha))
        thrs.append((220, 0.4 * tiles * alpha))
        thrs.append((250, 0.8 * tiles * alpha))
        thrs.append((300, 0.6 * tiles * alpha))
        thrs.append((400, 0.8 * tiles * alpha))
        thrs.append((10000, 0.9 * tiles * alpha))

        self.last_updated_time = 10000
        self.throuput_changes = thrs

    def activate_synthetic_thr_2(self):
        """
                A sample of synthetic network throughput
                :return:
        """

        thrs = []
        tiles = 6

        thrs.append((20, 0.4 * tiles))
        thrs.append((50, 0.9 * tiles))
        thrs.append((100, 0.7 * tiles))
        thrs.append((120, 0.6 * tiles))
        thrs.append((150, 1 * tiles))
        thrs.append((170, 0.7 * tiles))
        thrs.append((200, 0.5 * tiles))
        thrs.append((220, 0.4 * tiles))
        thrs.append((250, 0.8 * tiles))
        thrs.append((300, 0.6 * tiles))
        thrs.append((400, 0.8 * tiles))
        thrs.append((10000, 0.9 * tiles))
        self.last_updated_time = 10000
        self.throuput_changes = thrs

    def integral_of_bandwidth(self, begin, end):

        n = int((end - begin) / self.delta)
        result = 0
        for i in range(n):
            t1 = begin + i * self.delta
            t2 = t1 + self.delta
            result += self.delta * (self.get_thr(t1) + self.get_thr(t2)) / 2
        return result

    def get_finish_time(self, size, start):
        remaining = size
        time = start
        # try:
        while remaining > 0:
            remaining -= self.get_thr(time) * self.delta
            time += self.delta
        return time
        # except Exception as e:
        #     print(remaining)
        #     exit(0)

    def expected_download_time_old(self, segment_size, start_time):
        down_time = -1
        while down_time <= 0:
            dt = self.download_time(segment_size, start_time)
            rnd = np.random.random() * 2 * self.error_rate
            down_time = dt * (1 - self.error_rate + rnd)
        return down_time

    def average_bandwidth(self, time, window=5, dt=0.2):
        actual_bandwidths = []
        t = max(0, time - window)
        while t < time:
            thr = self.get_thr(t)
            actual_bandwidths.append(thr)
            t += dt
        if len(actual_bandwidths) == 0:
            return max(self.get_thr(0), 2)
        else:
            return max(np.average(actual_bandwidths), 2)

    def get_estimated_bandwidth(self, time, dt=0.2):
        actual_bandwidths = []
        t = max(0, time - 5)
        while t < time:
            thr = self.get_thr(t)
            actual_bandwidths.append(thr)
            t += dt
        if len(actual_bandwidths) == 0:
            return max(self.get_thr(0), 2)
        else:
            return max(np.average(actual_bandwidths), 2)

    def expected_download_time(self, segment_size, start_time):
        expected_thr = self.get_estimated_bandwidth(start_time)
        return segment_size / expected_thr

    def download_time_simulated(self, total_size, start_time):
        if start_time in self.calculated_download_time:
            if total_size in self.calculated_download_time[start_time]:
                return self.calculated_download_time[start_time][total_size]
            else:
                calculated_time = self.get_finish_time(total_size, start_time) - start_time
                self.calculated_download_time[start_time][total_size] = calculated_time

                return calculated_time
        else:
            self.calculated_download_time[start_time] = {}

            calculated_time = self.get_finish_time(total_size, start_time) - start_time
            self.calculated_download_time[start_time][total_size] = calculated_time

            return calculated_time

    def download_time_real(self, total_size, start_time):
        bandwidth_capacity = self.get_thr(start_time) * 10 ** 6
        download_time = self.chrome_downloader.download_file(total_size * 10 ** 6, bandwidth_capacity)
        return download_time

    def download_time(self, total_size, start_time, __real_downloader__=False):
        if __real_downloader__:
            return self.download_time_real(total_size, start_time)
        else:
            return self.download_time_simulated(total_size, start_time)
