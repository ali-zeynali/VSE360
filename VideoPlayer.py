from BOLA360 import *
from BOLA360H import *
from Buffer import *
from DDPOnline import *
from Flare import *
from HeadMoves import *
from Mosaic import *
from Naive import *
from Pano import *
from ProbDash import *
from Request import *
from SalientVR import *
from VAware import *
from Video import *

"""
        This file perform the streaming control for the implemented algorithms.
        
        You can use the implemented algorithms for your streaming evaluation or implement your own algorithm and control
        it in a new function in this class.
        
        Implemented algorithms are:
            BOLA360
            DDP-Online: A greedy algorithm
            Flare
            Mosaic
            Naive: A naive solution to 360ABR
            Pano
            360ProbDash
            SalientVR
            Vieport-Aware
            
"""


class VideoPlayer:
    def __init__(self):
        pass

    def get_available_video_in_buffer(self, buffer_array, downloaded_segments, delta):
        available_time = 0
        for i in range(len(buffer_array)):
            n = downloaded_segments[i]
            if n == 0:
                continue
            b = buffer_array[i] / n
            available_time += np.max(b * delta, 0)
        return available_time

    def remove_played_segments(self, buffer_array, time, delta):
        played_segment = time / delta
        played_segment_int = int(played_segment)
        if played_segment_int > len(buffer_array):
            played_segment_int = len(buffer_array)
        try:
            for i in range(played_segment_int):
                buffer_array[i] = 0
            if played_segment_int < len(buffer_array) - 1:
                buffer_array[played_segment_int] *= (1 - played_segment + played_segment_int)
        except Exception as e:
            print(e)
            exit(0)
        return buffer_array, played_segment

    def get_buffer_value(self, buffer_array):
        pass

    @staticmethod
    def get_download_times_static(solutions, times):
        download_times = []
        for i in range(len(solutions)):
            if np.sum(solutions[i]) > 0:
                download_times.append(times[i + 1])
        return download_times

    def get_download_times(self, solutions, times):
        download_times = []
        for i in range(len(solutions)):
            if np.sum(solutions[i]) > 0:
                download_times.append(times[i + 1])
        return download_times

    def update_rendered_times(self, play_time1, play_time2, rendered_times, delta, clock_time1):
        played_index1 = int(play_time1 / delta)
        played_index2 = int(play_time2 / delta)
        portion_played = play_time1 - played_index1 * delta
        cnt = 0
        for i in range(played_index1, played_index2):
            rendered_times[i] = clock_time1 + delta * cnt - portion_played
            cnt += 1
        return rendered_times

    def finalize_rendered_times(self, rendered_times, delta):
        t = 0
        cnt = 1
        for i in range(len(rendered_times)):
            if rendered_times[i] < 0:
                rendered_times[i] = t + cnt * delta
                cnt += 1
            else:
                t = rendered_times[i]
        return rendered_times

    def evaluation(self, solution, segment, new_segment_downloaded, delta, beta, probs, video, gamma, buffer_array,
                   time, time_passed, rebuffer, play_time1, downloaded_segments, time_from_previous_download,
                   rendered_times):

        available_time_in_buffer = self.get_available_video_in_buffer(buffer_array, downloaded_segments, delta)

        rebuff2 = time_passed - available_time_in_buffer
        rebuff2 = np.maximum(rebuff2, 0)

        play_time2 = time - rebuffer - rebuff2

        if rebuff2 > time:
            raise RuntimeError("Invalid rebuffering time!")

        if play_time2 < 0:
            raise RuntimeError("Negative value for time happened")
        buffer_array, played_segment = self.remove_played_segments(buffer_array, play_time1 + time_passed, delta)
        rendered_times = self.update_rendered_times(play_time1, play_time2, rendered_times, delta, time - time_passed)

        if segment > int(played_segment) - 1:
            buffer_array[segment] += new_segment_downloaded

        buffer = np.sum(buffer_array) * delta

        reward_utility = 0
        reward_smooth = 0
        total_size = self.get_total_size(solution, video)[0]
        expected_vals = 0
        for i in range(len(solution)):
            if solution[i] > 0:
                expected_vals += probs[i] * video.values[solution[i]]
                r_utility = probs[i] * video.values[solution[i]] - beta
                # r_utility /= (time_from_previous_download * video.sizes[solution[i]] / total_size)
                reward_utility += r_utility

                r_smooth = gamma * delta
                # r_smooth /= (time_from_previous_download * video.sizes[
                #     solution[i]] / total_size)  # version 1: using the difference time from previous download
                # r_smooth /= (time_passed * video.sizes[solution[i]] / total_size) # version 2: using the download time of this segment
                reward_smooth += r_smooth

        return reward_utility, reward_smooth, expected_vals, buffer, rebuff2, buffer_array, play_time2, played_segment, rendered_times

    def calc_reward(self, solution, segment, delta, beta, probs, video, gamma, buffer_array, available_time, time,
                    diff_time,
                    rebuffer):
        """

        :param solution:
        :param buffer:
        :param download_time:
        :param delta:
        :param probs:
        :param video:
        :param gamma:
        :return:
            r: expected reward of that download
            buffer: buffer level after downloading ends
            y: rebuffering time
        """
        ddp_n = 0
        for m in solution:
            if m > 0:
                ddp_n += 1

        played_time = time - rebuffer
        available_buffer = max(segment * delta, 0)
        played_time = min(played_time, available_buffer)
        last_played_segment = played_time / delta
        last_played_segment_int = int(last_played_segment)
        for i in range(last_played_segment_int):
            buffer_array[i] = 0
        buffer_array[last_played_segment_int] *= last_played_segment - last_played_segment_int  # remove the portion of
        # recently played segment from the buffer

        last_available = 0 - delta
        reb = None
        for i in range(segment):
            if available_time[i] > 0:
                last_available = available_time[i]

            if buffer_array[i] > 0:
                reb = 0
                break

        if reb is None:
            # if time > last_available + delta:
            reb = time - last_available - delta

        if reb < 0:
            print("OMG")
        buffer = np.sum(buffer_array) * delta
        reward = 0
        total_size = self.get_total_size(solution, video)[0]
        expected_vals = 0
        for i in range(len(solution)):
            if solution[i] > 0:
                expected_vals += probs[i] * video.values[solution[i]]
                r = probs[i] * video.values[solution[i]] + gamma * delta - beta
                r /= (diff_time * video.sizes[solution[i]] / total_size)
                reward += r

        if ddp_n > 0:
            return reward, expected_vals, buffer, reb, buffer_array
        else:
            return 0, 0, buffer, reb, buffer_array

    def read_json(self, path):
        with open(path) as reader:
            data = json.load(reader)
        return data

    def convert_action_to_rates(self, actions, sizes, delta):
        rates = []
        for action in actions:
            rates.append(sizes[action] / delta)

        return rates

    def get_total_size(self, set_of_actions, video):
        """

        :param set_of_actions: array size D, showing the bitrates of tiles which we are going to download
        :param video:
        :return:
            total_size: size of downloading segments in bits
            n: number of selected segments (non empty tiles) to download
        """
        total_size = 0
        n = 0
        for a in set_of_actions:
            total_size += video.sizes[a]
            if a > 0:
                n += 1
        return total_size, n

    def convert_action_to_dic(self, action, D):
        bitrate_dic = {}
        for d in range(D):
            if action[d] > 0:
                bitrate_dic[d] = action[d]
        return bitrate_dic

    def clean_history(self, history_parameters, current_playing_indx):
        for i in range(current_playing_indx):
            if i in history_parameters:
                history_parameters.pop(i)
        return history_parameters

    def update_history_probs(self, all_probs, history_parameters):
        for n in history_parameters:
            history_parameters[n]['probs'] = all_probs[n]
        return history_parameters

    def get_average_dl_rate(self, download_time_bitrates):
        bitrates = []
        for tuple in download_time_bitrates:
            btrs = tuple[1]
            for bitrate in btrs:
                bitrates.append(bitrate)
        return np.mean(bitrates)

    def run_BOLA360H(self, name, batch, path_to_meta, path_to_bandwidth, path_to_navGraph, result_path,
                     heuristic, given_gamma=-1, bandwidth_series=1, __print__=True, __start_time__=0,
                     __print_QoE__=False):
        if __print__:
            print("Algorithm starts work for batch {0}".format(batch))
        meta = self.read_json(path_to_meta)
        N = meta['N']
        D = meta['D']
        buffer_size = meta['b_max']
        delta = meta['delta']
        gamma = meta['gamma']
        beta = meta['beta']
        t_0 = meta['t_0']
        wait_time = meta['wait']
        sizes = np.array(meta['sizes'])
        bandwidth_error = meta['b_error']
        v_coeff = meta['V']
        actual_movement = meta['view']
        values = meta['values']
        hetro = meta['hetro']
        positive_tiles = meta['pos_tiles']
        bandwidth_target = meta['bandwidth_target']

        video = Video(N, delta, D, values, sizes, buffer_size)
        if bandwidth_series == 2:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, series=2,
                                  target_average=bandwidth_target)
        elif bandwidth_series == 3:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, series=3,
                                  target_average=bandwidth_target)
        else:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, target_average=bandwidth_target)

        headMovements = HeadMoves(N, D, path=path_to_navGraph)
        if hetro >= 0:
            headMovements.activate_hetrogen_model(hetro, positive_tiles)
        headMovements.set_actual_views(actual_movement)

        model = BOLA360H(video, gamma, v_coeff, buffer_size, beta, heuristic)

        params = {}
        if given_gamma > 0:
            gamma = given_gamma

        model_solutions = []

        available_content = [-1 for _ in range(N)]
        buffer_array_model = [0 for _ in range(N)]
        time_levels_model = []

        model_play_rates = []
        time_model = __start_time__
        buffer_model = 0  # unit: time
        buffer_values = Buffer()
        reward_utility = 0
        reward_smooth = 0
        rebuffer_model = 0
        total_reward_utility = 0
        total_reward_smooth = 0
        all_reward_smooth = []
        all_reward_utility = []
        total_reward_val_model = 0
        played_time_model = 0
        downloaded_tiles = [0 for _ in range(N)]

        played_segment_index = -1
        history_parameters = {}

        # algorithm
        n = 0
        recent_time_model = 0
        requests = []
        time_from_previous_download = 0
        rendered_times = [-1 for _ in range(N)]
        while True:
            if n >= N:
                break

            time_levels_model.append(time_model)

            # print("Alg n: ", n)
            all_probs = headMovements.get_pose_probs_updates(n)

            history_parameters = self.update_history_probs(all_probs, history_parameters)
            bandwidth_capacity = bandwidth.get_thr(time_model)
            model_action, action_type, action_segment, new_segments, idle_time = model.get_action(all_probs[n], n,
                                                                                                  buffer_array_model,
                                                                                                  downloaded_tiles,
                                                                                                  wait_time, delta,
                                                                                                  bandwidth_capacity,
                                                                                                  params,
                                                                                                  history=history_parameters)
            if __print__:
                print(
                    "{5}, Taking action for n: {0}, action_segment: {1}, new_segments: {2}, buffer: {3}, idle: {4}".format(
                        n, action_segment, new_segments, buffer_model, idle_time, name))
            if new_segments > 0:
                if action_segment not in history_parameters:
                    history_parameters[action_segment] = {}
                    history_parameters[action_segment][
                        'bitrate'] = model_action
                else:
                    for i in range(len(model_action)):
                        if model_action[i] > 0:
                            history_parameters[action_segment]['bitrate'][i] = model_action[i]
                history_parameters[action_segment]['probs'] = all_probs[action_segment]

            # print("Bola action: {0}".format(bola_action))
            model_solutions.append(model_action)
            # print("Action for n:{0}, \t{1}".format(n, bola_action))
            # print("getting DP action")

            if np.sum(model_action) > 0:
                total_size, download_n = self.get_total_size(model_action, video)
                download_time_model = bandwidth.download_time(total_size, time_model)
                tmp_line = 0

                if action_type == "new":
                    downloaded_tiles[action_segment] = download_n
                    # model_solutions.append(model_action)
                    # new_segment_downloaded = download_n
                if action_type == "replacement":
                    # downloaded_tiles[action_segment] += new_segments
                    pass
                    # new_segment_downloaded = new_segments
                model.take_action(model_action, action_segment, time_model)
                model_play_rates.append(
                    (time_model + download_time_model, self.convert_action_to_rates(model_action, sizes, delta)))
                # print("Downloaded : {0} and took {1} seconds".format(bola_action, download_time_bola))
            else:
                download_n = 0
                download_time_model = idle_time
                model.take_action(model_action, action_segment, time_model)

            time_taken = max(download_time_model,
                             (download_n - buffer_size) * delta + buffer_model - download_time_model)

            time_model += time_taken
            if np.sum(model_action) > 0:
                available_content[n] = time_model

            prob = all_probs[action_segment] if action_segment >= 0 else [0 for _ in range(D)]
            reward_utility, reward_smooth, reward_val_model, buffer_model, reb, buffer_array_model, played_time_model, played_segment, rendered_times = self.evaluation(
                model_action, n, new_segments, delta, beta, prob, video, gamma,
                buffer_array_model,
                time_model, time_model - recent_time_model, rebuffer_model, played_time_model, downloaded_tiles,
                            time_model - time_from_previous_download, rendered_times)

            self.clean_history(history_parameters, int(played_segment))
            recent_time_model = time_model
            if np.sum(model_action) > 0:
                n += 1
                time_from_previous_download = time_model
                bitrate_dic = self.convert_action_to_dic(model_action, D)
                request = Request("new", n - 1, time_model - time_taken, time_model, bitrate_dic)
                all_reward_smooth.append(reward_smooth)
                all_reward_utility.append(reward_utility)

            else:
                request = Request("wait", -1, time_model - time_taken, time_model, {})
            requests.append(request)

            # if n >= N:
            #     time_levels_bola.append(time_bola)
            #     buffer_levels_bola.append(buffer_bola)

            model.set_buffer(buffer_model / delta)

            rebuffer_model += reb
            total_reward_utility += reward_utility
            total_reward_smooth += reward_smooth
            total_reward_val_model += reward_val_model

            buffer_values.log_buffer(time_model, buffer_model)
        time_levels_model.append(time_model)
        self.finalize_rendered_times(rendered_times, delta)
        result = {}
        result['solution'] = model_solutions
        result['time'] = time_levels_model

        result['buffer'] = buffer_values.buffer_logs
        result['reward_utility'] = float(total_reward_utility / time_model)
        result['reward_smooth'] = float(total_reward_smooth / time_model)
        result['all_reward_utility'] = [v / time_model for v in all_reward_utility]
        result['all_reward_smooth'] = [v / time_model for v in all_reward_smooth]
        result['final_time'] = float(time_model)
        result['playing_br'] = model_play_rates
        result['rebuff'] = float(rebuffer_model)
        result['reward_val'] = float(total_reward_val_model)
        result['rendered_times'] = rendered_times
        result['requests'] = [x.__getArray__() for x in requests]

        with open(result_path, 'w') as writer:
            json.dump(result, writer)
        if __print_QoE__:
            print("QoE: {0}".format(total_reward_utility + total_reward_smooth))
            print("Rebuffering: {0}%".format(rebuffer_model * 100 / time_model))
            print("Average downloaded bitrate: {0}".format(self.get_average_dl_rate(model_play_rates)))

    def run_BOLA360(self, batch, path_to_meta, path_to_bandwidth, path_to_navGraph, result_path,
                    given_gamma=-1, bandwidth_series=1,
                    __print__=True, __start_time__=0, __print_QoE__=False):
        if __print__:
            print("Bola starts work for batch {0}".format(batch))
        meta = self.read_json(path_to_meta)
        N = meta['N']
        D = meta['D']
        buffer_size = meta['b_max']
        delta = meta['delta']
        gamma = meta['gamma']
        beta = meta['beta']
        t_0 = meta['t_0']
        wait_time = meta['wait']
        sizes = np.array(meta['sizes'])
        bandwidth_error = meta['b_error']
        v_coeff = meta['V']
        actual_movement = meta['view']
        values = meta['values']
        hetro = meta['hetro']
        positive_tiles = meta['pos_tiles']
        bandwidth_target = meta['bandwidth_target']

        video = Video(N, delta, D, values, sizes, buffer_size)
        if bandwidth_series == 2:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, series=2,
                                  target_average=bandwidth_target)
        elif bandwidth_series == "synthetic_1":
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error)
            bandwidth.activate_synthetic_thr_1()
        else:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, target_average=bandwidth_target)

        headMovements = HeadMoves(N, D, path=path_to_navGraph)
        if hetro >= 0:
            headMovements.activate_hetrogen_model(hetro, positive_tiles)
        headMovements.set_actual_views(actual_movement)

        if given_gamma > 0:
            gamma = given_gamma

        bola_solutions = []

        available_content = [-1 for _ in range(N)]
        buffer_array_bola = [0 for _ in range(N)]
        time_levels_bola = []

        bola_play_rates = []
        time_bola = __start_time__
        buffer_bola = 0  # unit: time
        buffer_values = Buffer()

        rebuffer_bola = 0
        total_reward_utility = 0
        total_reward_smooth = 0
        all_reward_smooth = []
        all_reward_utility = []
        total_reward_val_bola = 0
        played_time_model = 0
        downloaded_tiles = [0 for _ in range(N)]

        bola3d = BOLA360(video, gamma, v_coeff, buffer_size, beta)

        history_parameters = {}

        # bola algorithm
        n = 0
        recent_time_model = 0
        requests = []
        time_from_previous_download = 0
        rendered_times = [-1 for _ in range(N)]
        while True:
            if n >= N:
                break

            time_levels_bola.append(time_bola)

            all_probs = headMovements.get_pose_probs_updates(n)

            history_parameters = self.update_history_probs(all_probs, history_parameters)

            # print("getting bola action")
            bola_action, action_type, action_segment, new_segments, idle_time = bola3d.get_action(all_probs[n], n,
                                                                                                  buffer_array_bola,
                                                                                                  downloaded_tiles,
                                                                                                  wait_time, delta,
                                                                                                  previous_history=history_parameters)

            if new_segments > 0:
                if action_segment not in history_parameters:
                    history_parameters[action_segment] = {}
                    history_parameters[action_segment][
                        'bitrate'] = bola_action
                else:
                    for i in range(len(bola_action)):
                        if bola_action[i] > 0:
                            history_parameters[action_segment]['bitrate'][i] = bola_action[i]
                history_parameters[action_segment]['probs'] = all_probs[action_segment]

            bola_solutions.append(bola_action)

            if np.sum(bola_action) > 0:
                total_size, download_n = self.get_total_size(bola_action, video)
                download_time_bola = bandwidth.download_time(total_size, time_bola)
                if __print__:
                    print(
                        "[BOLA]: download finished for segment: {0} of batch {1}, buffer = {2}, action = {3}".format(n,
                                                                                                                     batch,
                                                                                                                     buffer_bola,
                                                                                                                     bola_action))
                tmp_line = 0

                if action_type == "new":
                    downloaded_tiles[action_segment] = download_n
                    # new_segment_downloaded = download_n
                if action_type == "replacement":
                    downloaded_tiles[action_segment] += new_segments
                    # new_segment_downloaded = new_segments
                bola3d.take_action(bola_action, action_segment, time_bola)
                bola_play_rates.append(
                    (time_bola + download_time_bola, self.convert_action_to_rates(bola_action, sizes, delta)))

            else:
                download_n = 0
                download_time_bola = idle_time
                bola3d.take_action(bola_action, action_segment, time_bola)

            time_taken = max(download_time_bola, (download_n - buffer_size) * delta + buffer_bola - download_time_bola)

            time_bola += time_taken
            if np.sum(bola_action) > 0:
                available_content[n] = time_bola

            prob = all_probs[action_segment] if action_segment >= 0 else [0 for _ in range(D)]
            reward_utility, reward_smooth, reward_val_bola, buffer_bola, reb, buffer_array_bola, played_time_model, played_segment, rendered_times = self.evaluation(
                bola_action, n, new_segments, delta, beta, prob, video, gamma,
                buffer_array_bola,
                time_bola, time_bola - recent_time_model, rebuffer_bola, played_time_model, downloaded_tiles,
                           time_bola - time_from_previous_download, rendered_times)

            self.clean_history(history_parameters, int(played_segment))
            recent_time_model = time_bola
            if np.sum(bola_action) > 0:
                n += 1
                time_from_previous_download = time_bola
                bitrate_dic = self.convert_action_to_dic(bola_action, D)
                request = Request("new", n - 1, time_bola - time_taken, time_bola, bitrate_dic)
                all_reward_smooth.append(reward_smooth)
                all_reward_utility.append(reward_utility)

            else:
                request = Request("wait", -1, time_bola - time_taken, time_bola, {})
            requests.append(request)

            bola3d.set_buffer(buffer_bola / delta)

            rebuffer_bola += reb
            total_reward_utility += reward_utility
            total_reward_smooth += reward_smooth
            total_reward_val_bola += reward_val_bola

            buffer_values.log_buffer(time_bola, buffer_bola)
        time_levels_bola.append(time_bola)
        self.finalize_rendered_times(rendered_times, delta)
        result = {}
        result['solution'] = bola_solutions
        result['time'] = time_levels_bola

        result['buffer'] = buffer_values.buffer_logs
        result['reward_utility'] = float(total_reward_utility / time_bola)
        result['reward_smooth'] = float(total_reward_smooth / time_bola)
        result['all_reward_utility'] = [v / time_bola for v in all_reward_utility]
        result['all_reward_smooth'] = [v / time_bola for v in all_reward_smooth]
        result['final_time'] = float(time_bola)
        result['playing_br'] = bola_play_rates
        result['rebuff'] = float(rebuffer_bola)
        result['reward_val'] = total_reward_val_bola
        result['rendered_times'] = rendered_times
        result['requests'] = [x.__getArray__() for x in requests]

        with open(result_path, 'w') as writer:
            json.dump(result, writer)
        if __print_QoE__:
            print("QoE: {0}".format(total_reward_utility + total_reward_smooth))
            print("Rebuffering: {0}%".format(rebuffer_bola * 100 / time_bola))
            print("Average downloaded bitrate: {0}".format(self.get_average_dl_rate(bola_play_rates)))

    def run_DDP(self, batch, path_to_meta, path_to_bandwidth, path_to_navGraph, result_path,
                result_path_base, bandwidth_series=1, base_comparision=False,
                __print__=True, __start_time__=0, __print_QoE__=False):
        if __print__:
            print("DDP starts work for batch {0}".format(batch))
        if base_comparision and batch >= 1:
            with open(result_path_base, 'r') as reader:
                data = json.load(reader)
            with open(result_path, 'w') as writer:
                json.dump(data, writer)
            return
        meta = self.read_json(path_to_meta)
        N = meta['N']
        D = meta['D']
        buffer_size = meta['b_max']
        delta = meta['delta']
        gamma = meta['gamma']
        beta = meta['beta']
        t_0 = meta['t_0']
        wait_time = meta['wait']
        sizes = np.array(meta['sizes'])
        bandwidth_error = meta['b_error']
        actual_movement = meta['view']
        values = meta['values']
        hetro = meta['hetro']
        positive_tiles = meta['pos_tiles']
        bandwidth_target = meta['bandwidth_target']

        video = Video(N, delta, D, values, sizes, buffer_size)
        if bandwidth_series == 2:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, series=2,
                                  target_average=bandwidth_target)
        else:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, target_average=bandwidth_target)

        headMovements = HeadMoves(N, D, path=path_to_navGraph)
        if hetro >= 0:
            headMovements.activate_hetrogen_model(hetro, positive_tiles)
        headMovements.set_actual_views(actual_movement)

        # model_performance = []
        model_solutions = []
        # buffer_levels_model = []
        available_content = [-1 for _ in range(N)]
        buffer_array_model = [0 for _ in range(N)]
        time_levels_model = []

        model_play_rates = []
        time_model = __start_time__
        buffer_model = 0  # unit: time
        buffer_values = Buffer()
        reward_model = 0
        rebuffer_model = 0
        total_reward_utility = 0
        total_reward_smooth = 0
        all_reward_smooth = []
        all_reward_utility = []
        total_reward_val_model = 0
        played_time_model = 0
        downloaded_tiles = [0 for _ in range(N)]

        params = {}
        params['buffer_size'] = buffer_size
        params['bandwidth'] = bandwidth
        params['gamma'] = gamma
        params['beta'] = beta

        model = DDPOnline(video, params)

        history_parameters = {}

        # model algorithm
        n = 0
        recent_time_model = 0
        requests = []
        time_from_previous_download = 0
        rendered_times = [-1 for _ in range(N)]
        while True:
            if n >= N:
                break

            time_levels_model.append(time_model)

            all_probs = headMovements.get_pose_probs_updates(n)

            history_parameters = self.update_history_probs(all_probs, history_parameters)

            model_action, action_type, action_segment, new_segments, idle_time = model.get_action(all_probs[n],
                                                                                                  time_model,
                                                                                                  n, buffer_array_model,
                                                                                                  downloaded_tiles,
                                                                                                  wait_time,
                                                                                                  delta)

            if new_segments > 0:
                if action_segment not in history_parameters:
                    history_parameters[action_segment] = {}
                    history_parameters[action_segment][
                        'bitrate'] = model_action
                else:
                    for i in range(len(model_action)):
                        if model_action[i] > 0:
                            history_parameters[action_segment]['bitrate'][i] = model_action[i]
                history_parameters[action_segment]['probs'] = all_probs[action_segment]

            model_solutions.append(model_action)

            if np.sum(model_action) > 0:
                total_size, download_n = self.get_total_size(model_action, video)
                download_time_model = bandwidth.download_time(total_size, time_model)
                if __print__:
                    print("[DDP]: download finished for segment: {0} of batch {1}".format(n, batch))

                if action_type == "new":
                    downloaded_tiles[action_segment] = download_n
                    # new_segment_downloaded = download_n
                if action_type == "replacement":
                    downloaded_tiles[action_segment] += new_segments
                    # new_segment_downloaded = new_segments
                model.take_action(model_action, action_segment, time_model)
                model_play_rates.append(
                    (time_model + download_time_model, self.convert_action_to_rates(model_action, sizes, delta)))

            else:
                download_n = 0
                download_time_model = idle_time
                model.take_action(model_action, action_segment, time_model)

            time_taken = max(download_time_model,
                             (download_n - buffer_size) * delta + buffer_model - download_time_model)

            time_model += time_taken
            if np.sum(model_action) > 0:
                available_content[n] = time_model

            prob = all_probs[action_segment] if action_segment >= 0 else [0 for _ in range(D)]
            reward_utility, reward_smooth, reward_val_model, buffer_model, reb, buffer_array_model, played_time_model, played_segment, rendered_times = self.evaluation(
                model_action, n, new_segments, delta, beta, prob, video, gamma,
                buffer_array_model,
                time_model, time_model - recent_time_model, rebuffer_model, played_time_model, downloaded_tiles,
                            time_model - time_from_previous_download, rendered_times)

            self.clean_history(history_parameters, int(played_segment))
            recent_time_model = time_model
            if np.sum(model_action) > 0:
                n += 1
                time_from_previous_download = time_model
                bitrate_dic = self.convert_action_to_dic(model_action, D)
                request = Request("new", n - 1, time_model - time_taken, time_model, bitrate_dic)
                all_reward_smooth.append(reward_smooth)
                all_reward_utility.append(reward_utility)

            else:
                request = Request("wait", -1, time_model - time_taken, time_model, {})
            requests.append(request)

            model.set_buffer(buffer_model / delta)

            rebuffer_model += reb
            total_reward_utility += reward_utility
            total_reward_smooth += reward_smooth
            total_reward_val_model += reward_val_model

            buffer_values.log_buffer(time_model, buffer_model)

        time_levels_model.append(time_model)
        self.finalize_rendered_times(rendered_times, delta)
        result = {}
        result['solution'] = model_solutions
        result['time'] = time_levels_model

        result['buffer'] = buffer_values.buffer_logs
        result['reward_utility'] = float(total_reward_utility / time_model)
        result['reward_smooth'] = float(total_reward_smooth / time_model)
        result['all_reward_utility'] = [v / time_model for v in all_reward_utility]
        result['all_reward_smooth'] = [v / time_model for v in all_reward_smooth]
        result['final_time'] = float(time_model)
        result['playing_br'] = model_play_rates
        result['rebuff'] = float(rebuffer_model)
        result['reward_val'] = total_reward_val_model
        result['rendered_times'] = rendered_times
        result['requests'] = [x.__getArray__() for x in requests]

        with open(result_path, 'w') as writer:
            json.dump(result, writer)
        if __print_QoE__:
            print("QoE: {0}".format(total_reward_utility + total_reward_smooth))

    def run_NAIVE(self, batch, path_to_meta, path_to_bandwidth, path_to_navGraph, result_path,
                  result_path_base, __tile_to_download__, base_comparision, bandwidth_series=1,
                  __print__=True, __start_time__=0, __print_QoE__=False):
        if base_comparision and batch >= 1:
            with open(result_path_base,
                      'r') as reader:
                data = json.load(reader)
            with open(result_path,
                      'w') as writer:
                json.dump(data, writer)
            return
        if __print__:
            print("Naive-{1} starts work for batch {0}".format(batch, __tile_to_download__))
        meta = self.read_json(path_to_meta)
        N = meta['N']
        D = meta['D']
        buffer_size = meta['b_max']
        delta = meta['delta']
        gamma = meta['gamma']
        beta = meta['beta']
        t_0 = meta['t_0']
        wait_time = meta['wait']
        sizes = np.array(meta['sizes'])
        bandwidth_error = meta['b_error']
        v_coeff = meta['V']
        actual_movement = meta['view']
        values = meta['values']
        hetro = meta['hetro']
        positive_tiles = meta['pos_tiles']
        bandwidth_target = meta['bandwidth_target']

        video = Video(N, delta, D, values, sizes, buffer_size)
        if bandwidth_series == 2:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, series=2,
                                  target_average=bandwidth_target)
        else:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, target_average=bandwidth_target)

        headMovements = HeadMoves(N, D, path=path_to_navGraph)
        if hetro >= 0:
            headMovements.activate_hetrogen_model(hetro, positive_tiles)
        headMovements.set_actual_views(actual_movement)

        # model_performance = []
        model_solutions = []
        # buffer_levels_model = []
        available_content = [-1 for _ in range(N)]
        buffer_array_model = [0 for _ in range(N)]
        time_levels_model = []

        model_play_rates = []
        time_model = __start_time__
        buffer_model = 0  # unit: time
        buffer_values = Buffer()
        reward_model = 0
        rebuffer_model = 0
        total_reward_utility = 0
        total_reward_smooth = 0
        total_reward_val_model = 0
        played_time_model = 0
        downloaded_tiles = [0 for _ in range(N)]

        params = {}
        params['buffer_capacity'] = buffer_size
        params['tile_to_download'] = __tile_to_download__
        params['gamma'] = gamma
        params['beta'] = beta

        model = Naive(video, params)

        history_parameters = {}

        # model algorithm
        n = 0
        recent_time_model = 0
        requests = []
        time_from_previous_download = 0
        rendered_times = [-1 for _ in range(N)]
        while True:
            if n >= N:
                break

            time_levels_model.append(time_model)

            all_probs = headMovements.get_pose_probs_updates(n)

            history_parameters = self.update_history_probs(all_probs, history_parameters)

            bandwidth_capacity = bandwidth.average_bandwidth(time_model)
            model_action, action_type, action_segment, new_segments, idle_time = model.get_action(all_probs[n],
                                                                                                  bandwidth_capacity,
                                                                                                  n, buffer_array_model,
                                                                                                  downloaded_tiles,
                                                                                                  wait_time,
                                                                                                  delta)

            if new_segments > 0:
                if action_segment not in history_parameters:
                    history_parameters[action_segment] = {}
                    history_parameters[action_segment][
                        'bitrate'] = model_action
                else:
                    for i in range(len(model_action)):
                        if model_action[i] > 0:
                            history_parameters[action_segment]['bitrate'][i] = model_action[i]
                history_parameters[action_segment]['probs'] = all_probs[action_segment]

            model_solutions.append(model_action)

            if np.sum(model_action) > 0:
                total_size, download_n = self.get_total_size(model_action, video)
                download_time_model = bandwidth.download_time(total_size, time_model)
                if __print__:
                    print("[Naive]: download finished for segment: {0} of batch {1}".format(n, batch))

                if action_type == "new":
                    downloaded_tiles[action_segment] = download_n
                    # new_segment_downloaded = download_n
                if action_type == "replacement":
                    downloaded_tiles[action_segment] += new_segments
                    # new_segment_downloaded = new_segments
                model.take_action(model_action, action_segment, time_model)
                model_play_rates.append(
                    (time_model + download_time_model, self.convert_action_to_rates(model_action, sizes, delta)))

            else:
                download_n = 0
                download_time_model = idle_time
                model.take_action(model_action, action_segment, time_model)

            time_taken = max(download_time_model,
                             (download_n - buffer_size) * delta + buffer_model - download_time_model)

            time_model += time_taken
            if np.sum(model_action) > 0:
                available_content[n] = time_model

            prob = all_probs[action_segment] if action_segment >= 0 else [0 for _ in range(D)]
            reward_utility, reward_smooth, reward_val_model, buffer_model, reb, buffer_array_model, played_time_model, played_segment, rendered_times = self.evaluation(
                model_action, n, new_segments, delta, beta, prob, video, gamma,
                buffer_array_model,
                time_model, time_model - recent_time_model, rebuffer_model, played_time_model, downloaded_tiles,
                            time_model - time_from_previous_download, rendered_times)

            self.clean_history(history_parameters, int(played_segment))
            recent_time_model = time_model
            if np.sum(model_action) > 0:
                n += 1
                time_from_previous_download = time_model
                bitrate_dic = self.convert_action_to_dic(model_action, D)
                request = Request("new", n - 1, time_model - time_taken, time_model, bitrate_dic)

            else:
                request = Request("wait", -1, time_model - time_taken, time_model, {})
            requests.append(request)

            model.set_buffer(buffer_model / delta)

            rebuffer_model += reb
            total_reward_utility += reward_utility
            total_reward_smooth += reward_smooth
            total_reward_val_model += reward_val_model

            buffer_values.log_buffer(time_model, buffer_model)

        time_levels_model.append(time_model)
        self.finalize_rendered_times(rendered_times, delta)
        result = {}
        result['solution'] = model_solutions
        result['time'] = time_levels_model

        result['buffer'] = buffer_values.buffer_logs
        result['reward_utility'] = float(total_reward_utility / time_model)
        result['reward_smooth'] = float(total_reward_smooth / time_model)
        result['final_time'] = float(time_model)
        result['playing_br'] = model_play_rates
        result['rebuff'] = float(rebuffer_model)
        result['reward_val'] = total_reward_val_model
        result['rendered_times'] = rendered_times
        result['requests'] = [x.__getArray__() for x in requests]

        with open(result_path, 'w') as writer:
            json.dump(result, writer)
        if __print_QoE__:
            print("QoE: {0}".format(total_reward_utility + total_reward_smooth))

    def run_VAware(self, batch, path_to_meta, path_to_bandwidth, path_to_navGraph, result_path,
                   bandwidth_series=1,
                   __print__=True, __start_time__=0, __print_QoE__=False):
        if __print__:
            print("VAware starts work for batch {0}".format(batch))
        meta = self.read_json(path_to_meta)
        N = meta['N']
        D = meta['D']
        buffer_size = meta['b_max']
        delta = meta['delta']
        gamma = meta['gamma']
        beta = meta['beta']
        t_0 = meta['t_0']
        wait_time = meta['wait']
        sizes = np.array(meta['sizes'])
        bandwidth_error = meta['b_error']
        v_coeff = meta['V']
        actual_movement = meta['view']
        values = meta['values']
        hetro = meta['hetro']
        positive_tiles = meta['pos_tiles']
        bandwidth_target = meta['bandwidth_target']

        video = Video(N, delta, D, values, sizes, buffer_size)
        if bandwidth_series == 2:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, series=2,
                                  target_average=bandwidth_target)
        else:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, target_average=bandwidth_target)

        headMovements = HeadMoves(N, D, path=path_to_navGraph)
        if hetro >= 0:
            headMovements.activate_hetrogen_model(hetro, positive_tiles)
        headMovements.set_actual_views(actual_movement)

        model_solutions = []

        available_content = [-1 for _ in range(N)]
        buffer_array_model = [0 for _ in range(N)]
        time_levels_model = []

        model_play_rates = []
        time_model = __start_time__
        buffer_model = 0  # unit: time
        buffer_values = Buffer()

        rebuffer_model = 0
        total_reward_utility = 0
        total_reward_smooth = 0
        total_reward_val_model = 0
        played_time_model = 0
        downloaded_tiles = [0 for _ in range(N)]

        params = {}
        params['buffer_capacity'] = buffer_size
        model = VAware(video, params)

        played_segment_index = -1
        history_parameters = {}

        # model algorithm
        n = 0
        recent_time_model = 0
        requests = []
        time_from_previous_download = 0
        rendered_times = [-1 for _ in range(N)]
        while True:
            if n >= N:
                break

            time_levels_model.append(time_model)

            all_probs = headMovements.get_pose_probs_updates(n)

            history_parameters = self.update_history_probs(all_probs, history_parameters)

            bandwidth_capacity = bandwidth.average_bandwidth(time_model)
            model_action, action_type, action_segment, new_segments, idle_time = model.get_action(all_probs[n],
                                                                                                  bandwidth_capacity,
                                                                                                  n, buffer_array_model,
                                                                                                  downloaded_tiles,
                                                                                                  wait_time,
                                                                                                  delta)

            if new_segments > 0:
                if action_segment not in history_parameters:
                    history_parameters[action_segment] = {}
                    history_parameters[action_segment][
                        'bitrate'] = model_action
                else:
                    for i in range(len(model_action)):
                        if model_action[i] > 0:
                            history_parameters[action_segment]['bitrate'][i] = model_action[i]
                history_parameters[action_segment]['probs'] = all_probs[action_segment]

            model_solutions.append(model_action)

            if np.sum(model_action) > 0:
                total_size, download_n = self.get_total_size(model_action, video)
                download_time_model = bandwidth.download_time(total_size, time_model)
                if __print__:
                    print("[VAware]: download finished for segment: {0} of batch {1}".format(n, batch))

                if action_type == "new":
                    downloaded_tiles[action_segment] = download_n
                    # new_segment_downloaded = download_n
                if action_type == "replacement":
                    downloaded_tiles[action_segment] += new_segments
                    # new_segment_downloaded = new_segments
                model.take_action(model_action, action_segment, time_model)
                model_play_rates.append(
                    (time_model + download_time_model, self.convert_action_to_rates(model_action, sizes, delta)))

            else:
                download_n = 0
                download_time_model = idle_time
                model.take_action(model_action, action_segment, time_model)

            time_taken = max(download_time_model,
                             (download_n - buffer_size) * delta + buffer_model - download_time_model)

            time_model += time_taken
            if np.sum(model_action) > 0:
                available_content[n] = time_model

            prob = all_probs[action_segment] if action_segment >= 0 else [0 for _ in range(D)]
            reward_utility, reward_smooth, reward_val_model, buffer_model, reb, buffer_array_model, played_time_model, played_segment, rendered_times = self.evaluation(
                model_action, n, new_segments, delta, beta, prob, video, gamma,
                buffer_array_model,
                time_model, time_model - recent_time_model, rebuffer_model, played_time_model, downloaded_tiles,
                            time_model - time_from_previous_download, rendered_times)

            self.clean_history(history_parameters, int(played_segment))
            recent_time_model = time_model
            if np.sum(model_action) > 0:
                n += 1
                time_from_previous_download = time_model
                bitrate_dic = self.convert_action_to_dic(model_action, D)
                request = Request("new", n - 1, time_model - time_taken, time_model, bitrate_dic)

            else:
                request = Request("wait", -1, time_model - time_taken, time_model, {})
            requests.append(request)

            model.set_buffer(buffer_model / delta)

            rebuffer_model += reb
            total_reward_utility += reward_utility
            total_reward_smooth += reward_smooth
            total_reward_val_model += reward_val_model

            buffer_values.log_buffer(time_model, buffer_model)

        time_levels_model.append(time_model)
        self.finalize_rendered_times(rendered_times, delta)
        result = {}
        result['solution'] = model_solutions
        result['time'] = time_levels_model

        result['buffer'] = buffer_values.buffer_logs
        result['reward_utility'] = float(total_reward_utility / time_model)
        result['reward_smooth'] = float(total_reward_smooth / time_model)
        result['final_time'] = float(time_model)
        result['playing_br'] = model_play_rates
        result['rebuff'] = float(rebuffer_model)
        result['reward_val'] = total_reward_val_model
        result['rendered_times'] = rendered_times
        result['requests'] = [x.__getArray__() for x in requests]

        with open(result_path, 'w') as writer:
            json.dump(result, writer)
        if __print_QoE__:
            print("QoE: {0}".format(total_reward_utility + total_reward_smooth))

    def run_SalientVR(self, batch, path_to_meta, path_to_bandwidth, path_to_navGraph, result_path,
                      buffer_threshold, bandwidth_series=1,
                      __print__=True, __start_time__=0, __print_QoE__=False):
        if __print__:
            print("SalientVR starts work for batch {0}".format(batch))
        meta = self.read_json(path_to_meta)
        N = meta['N']
        D = meta['D']
        buffer_size = meta['b_max']
        delta = meta['delta']
        gamma = meta['gamma']
        beta = meta['beta']
        t_0 = meta['t_0']
        wait_time = meta['wait']
        sizes = np.array(meta['sizes'])
        bandwidth_error = meta['b_error']
        v_coeff = meta['V']
        actual_movement = meta['view']
        values = meta['values']
        hetro = meta['hetro']
        positive_tiles = meta['pos_tiles']
        bandwidth_target = meta['bandwidth_target']

        video = Video(N, delta, D, values, sizes, buffer_size)
        if bandwidth_series == 2:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, series=2,
                                  target_average=bandwidth_target)
        else:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, target_average=bandwidth_target)

        headMovements = HeadMoves(N, D, path=path_to_navGraph)
        if hetro >= 0:
            headMovements.activate_hetrogen_model(hetro, positive_tiles)
        headMovements.set_actual_views(actual_movement)

        model_solutions = []

        available_content = [-1 for _ in range(N)]
        buffer_array_model = [0 for _ in range(N)]
        time_levels_model = []

        model_play_rates = []
        time_model = __start_time__
        buffer_model = 0  # unit: time
        buffer_values = Buffer()

        rebuffer_model = 0
        total_reward_utility = 0
        total_reward_smooth = 0
        total_reward_val_model = 0
        played_time_model = 0
        downloaded_tiles = [0 for _ in range(N)]

        params = {}
        params['buffer_capacity'] = buffer_size
        params['buffer_threshold'] = buffer_threshold
        model = SalientVR(video, params)

        history_parameters = {}

        # model algorithm
        n = 0
        recent_time_model = 0
        requests = []
        time_from_previous_download = 0
        rendered_times = [-1 for _ in range(N)]
        while True:
            if n >= N:
                break

            time_levels_model.append(time_model)

            all_probs = headMovements.get_pose_probs_updates(n)

            history_parameters = self.update_history_probs(all_probs, history_parameters)

            bandwidth_capacity = bandwidth.average_bandwidth(time_model)
            model_action, action_type, action_segment, new_segments, idle_time = model.get_action(all_probs[n],
                                                                                                  bandwidth_capacity,
                                                                                                  n, buffer_array_model,
                                                                                                  downloaded_tiles,
                                                                                                  wait_time,
                                                                                                  delta)

            if new_segments > 0:
                if action_segment not in history_parameters:
                    history_parameters[action_segment] = {}
                    history_parameters[action_segment][
                        'bitrate'] = model_action
                else:
                    for i in range(len(model_action)):
                        if model_action[i] > 0:
                            history_parameters[action_segment]['bitrate'][i] = model_action[i]
                history_parameters[action_segment]['probs'] = all_probs[action_segment]

            model_solutions.append(model_action)

            if np.sum(model_action) > 0:
                total_size, download_n = self.get_total_size(model_action, video)
                download_time_model = bandwidth.download_time(total_size, time_model)
                if __print__:
                    print("[SalientVR]: download finished for segment: {0} of batch {1}".format(n, batch))

                if action_type == "new":
                    downloaded_tiles[action_segment] = download_n
                    # new_segment_downloaded = download_n
                if action_type == "replacement":
                    downloaded_tiles[action_segment] += new_segments
                    # new_segment_downloaded = new_segments
                model.take_action(model_action, action_segment, time_model)
                model_play_rates.append(
                    (time_model + download_time_model, self.convert_action_to_rates(model_action, sizes, delta)))
                # print("Downloaded : {0} and took {1} seconds".format(bola_action, download_time_bola))
            else:
                download_n = 0
                download_time_model = idle_time
                model.take_action(model_action, action_segment, time_model)

            time_taken = max(download_time_model,
                             (download_n - buffer_size) * delta + buffer_model - download_time_model)

            time_model += time_taken
            if np.sum(model_action) > 0:
                available_content[n] = time_model

            prob = all_probs[action_segment] if action_segment >= 0 else [0 for _ in range(D)]
            reward_utility, reward_smooth, reward_val_model, buffer_model, reb, buffer_array_model, played_time_model, played_segment, rendered_times = self.evaluation(
                model_action, n, new_segments, delta, beta, prob, video, gamma,
                buffer_array_model,
                time_model, time_model - recent_time_model, rebuffer_model, played_time_model, downloaded_tiles,
                            time_model - time_from_previous_download, rendered_times)

            self.clean_history(history_parameters, int(played_segment))
            recent_time_model = time_model
            if np.sum(model_action) > 0:
                n += 1
                time_from_previous_download = time_model
                bitrate_dic = self.convert_action_to_dic(model_action, D)
                request = Request("new", n - 1, time_model - time_taken, time_model, bitrate_dic)

            else:
                request = Request("wait", -1, time_model - time_taken, time_model, {})
            requests.append(request)

            model.set_buffer(buffer_model / delta)

            rebuffer_model += reb
            total_reward_utility += reward_utility
            total_reward_smooth += reward_smooth
            total_reward_val_model += reward_val_model

            buffer_values.log_buffer(time_model, buffer_model)

        time_levels_model.append(time_model)
        self.finalize_rendered_times(rendered_times, delta)
        result = {}
        result['solution'] = model_solutions
        result['time'] = time_levels_model

        result['buffer'] = buffer_values.buffer_logs
        result['reward_utility'] = float(total_reward_utility / time_model)
        result['reward_smooth'] = float(total_reward_smooth / time_model)
        result['final_time'] = float(time_model)
        result['playing_br'] = model_play_rates
        result['rebuff'] = float(rebuffer_model)
        result['reward_val'] = total_reward_val_model
        result['rendered_times'] = rendered_times
        result['requests'] = [x.__getArray__() for x in requests]

        with open(result_path, 'w') as writer:
            json.dump(result, writer)
        if __print_QoE__:
            print("QoE: {0}".format(total_reward_utility + total_reward_smooth))

    def run_ProbDash(self, batch, path_to_meta, path_to_bandwidth, path_to_navGraph, result_path,
                     bandwidth_series=1,
                     __print__=True, __start_time__=0, __print_QoE__=False):
        if __print__:
            print("ProbDash starts work for batch {0}".format(batch))
        meta = self.read_json(path_to_meta)
        N = meta['N']
        D = meta['D']
        buffer_size = meta['b_max']
        delta = meta['delta']
        gamma = meta['gamma']
        beta = meta['beta']
        t_0 = meta['t_0']
        wait_time = meta['wait']
        sizes = np.array(meta['sizes'])
        bandwidth_error = meta['b_error']
        actual_movement = meta['view']
        values = meta['values']
        hetro = meta['hetro']
        positive_tiles = meta['pos_tiles']
        eta = meta['eta']
        target_buffer = meta['target_buffer']
        bandwidth_target = meta['bandwidth_target']

        video = Video(N, delta, D, values, sizes, buffer_size)
        if bandwidth_series == 2:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, series=2,
                                  target_average=bandwidth_target)
        else:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, target_average=bandwidth_target)

        headMovements = HeadMoves(N, D, path=path_to_navGraph)
        if hetro >= 0:
            headMovements.activate_hetrogen_model(hetro, positive_tiles)
        headMovements.set_actual_views(actual_movement)

        model_solutions = []

        available_content = [-1 for _ in range(N)]
        buffer_array_model = [0 for _ in range(N)]
        time_levels_model = []

        model_play_rates = []
        time_model = __start_time__
        buffer_model = 0  # unit: time
        buffer_values = Buffer()

        rebuffer_model = 0
        total_reward_utility = 0
        total_reward_smooth = 0
        total_reward_val_model = 0
        played_time_model = 0
        downloaded_tiles = [0 for _ in range(N)]

        params = {}
        params['buffer_capacity'] = buffer_size
        params['bitrates'] = np.array(sizes) / delta
        params['target_buffer'] = target_buffer
        params['eta'] = eta
        model = ProbDash(video, params)

        played_segment_index = -1
        history_parameters = {}

        # model algorithm
        n = 0
        recent_time_model = 0
        requests = []
        time_from_previous_download = 0
        rendered_times = [-1 for _ in range(N)]
        while True:
            if n >= N:
                break

            time_levels_model.append(time_model)

            all_probs = headMovements.get_pose_probs_updates(n)

            history_parameters = self.update_history_probs(all_probs, history_parameters)

            # print("getting bola action")
            bandwidth_capacity = bandwidth.get_thr(time_model)
            model_action, action_type, action_segment, new_segments, idle_time = model.get_action(all_probs[n],
                                                                                                  bandwidth_capacity,
                                                                                                  n, buffer_array_model,
                                                                                                  downloaded_tiles,
                                                                                                  wait_time,
                                                                                                  delta)

            if new_segments > 0:
                if action_segment not in history_parameters:
                    history_parameters[action_segment] = {}
                    history_parameters[action_segment][
                        'bitrate'] = model_action
                else:
                    for i in range(len(model_action)):
                        if model_action[i] > 0:
                            history_parameters[action_segment]['bitrate'][i] = model_action[i]
                history_parameters[action_segment]['probs'] = all_probs[action_segment]

            model_solutions.append(model_action)

            if np.sum(model_action) > 0:
                total_size, download_n = self.get_total_size(model_action, video)
                download_time_model = bandwidth.download_time(total_size, time_model)
                if __print__:
                    print("[ProbDash]: download finished for segment: {0} of batch {1}".format(n, batch))

                if action_type == "new":
                    downloaded_tiles[action_segment] = download_n

                if action_type == "replacement":
                    downloaded_tiles[action_segment] += new_segments

                model.take_action(model_action, action_segment, time_model)
                model_play_rates.append(
                    (time_model + download_time_model, self.convert_action_to_rates(model_action, sizes, delta)))

            else:
                download_n = 0
                download_time_model = idle_time
                model.take_action(model_action, action_segment, time_model)

            time_taken = max(download_time_model,
                             (download_n - buffer_size) * delta + buffer_model - download_time_model)

            time_model += time_taken
            if np.sum(model_action) > 0:
                available_content[n] = time_model

            prob = all_probs[action_segment] if action_segment >= 0 else [0 for _ in range(D)]
            reward_utility, reward_smooth, reward_val_model, buffer_model, reb, buffer_array_model, played_time_model, played_segment, rendered_times = self.evaluation(
                model_action, n, new_segments, delta, beta, prob, video, gamma,
                buffer_array_model,
                time_model, time_model - recent_time_model, rebuffer_model, played_time_model, downloaded_tiles,
                            time_model - time_from_previous_download, rendered_times)

            self.clean_history(history_parameters, int(played_segment))
            recent_time_model = time_model
            if np.sum(model_action) > 0:
                n += 1
                time_from_previous_download = time_model
                bitrate_dic = self.convert_action_to_dic(model_action, D)
                request = Request("new", n - 1, time_model - time_taken, time_model, bitrate_dic)

            else:
                request = Request("wait", -1, time_model - time_taken, time_model, {})
            requests.append(request)

            model.set_buffer(buffer_model / delta)

            rebuffer_model += reb
            total_reward_utility += reward_utility
            total_reward_smooth += reward_smooth
            total_reward_val_model += reward_val_model

            buffer_values.log_buffer(time_model, buffer_model)

        time_levels_model.append(time_model)
        self.finalize_rendered_times(rendered_times, delta)
        result = {}
        result['solution'] = model_solutions
        result['time'] = time_levels_model

        result['buffer'] = buffer_values.buffer_logs
        result['reward_utility'] = float(total_reward_utility / time_model)
        result['reward_smooth'] = float(total_reward_smooth / time_model)
        result['final_time'] = float(time_model)
        result['playing_br'] = model_play_rates
        result['rebuff'] = float(rebuffer_model)
        result['reward_val'] = total_reward_val_model
        result['rendered_times'] = rendered_times
        result['requests'] = [x.__getArray__() for x in requests]

        with open(result_path, 'w') as writer:
            json.dump(result, writer)
        if __print_QoE__:
            print("QoE: {0}".format(total_reward_utility + total_reward_smooth))

    def run_FLARE(self, batch, path_to_meta, path_to_bandwidth, path_to_navGraph, result_path,
                  bandwidth_series=1,
                  __print__=True, __start_time__=0, __print_QoE__=False):
        if __print__:
            print("Flare starts work for batch {0}".format(batch))
        meta = self.read_json(path_to_meta)
        N = meta['N']
        D = meta['D']
        buffer_size = meta['b_max']
        delta = meta['delta']
        gamma = meta['gamma']
        beta = meta['beta']
        t_0 = meta['t_0']
        wait_time = meta['wait']
        sizes = np.array(meta['sizes'])
        bandwidth_error = meta['b_error']
        v_coeff = meta['V']
        actual_movement = meta['view']
        values = meta['values']
        hetro = meta['hetro']
        positive_tiles = meta['pos_tiles']
        bandwidth_target = meta['bandwidth_target']

        video = Video(N, delta, D, values, sizes, buffer_size)
        if bandwidth_series == 2:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, series=2,
                                  target_average=bandwidth_target)
        else:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, target_average=bandwidth_target)

        headMovements = HeadMoves(N, D, path=path_to_navGraph)
        if hetro >= 0:
            headMovements.activate_hetrogen_model(hetro, positive_tiles)
        headMovements.set_actual_views(actual_movement)

        model_solutions = []

        available_content = [-1 for _ in range(N)]
        buffer_array_model = [0 for _ in range(N)]
        time_levels_model = []

        model_play_rates = []
        time_model = __start_time__
        buffer_model = 0  # unit: time
        buffer_values = Buffer()

        rebuffer_model = 0
        total_reward_utility = 0
        total_reward_smooth = 0
        total_reward_val_model = 0
        played_time_model = 0
        downloaded_tiles = [0 for _ in range(N)]

        params = {}
        params['buffer_capacity'] = buffer_size
        model = Flare(video, params)

        played_segment_index = -1
        history_parameters = {}

        # model algorithm
        n = 0
        recent_time_model = 0
        requests = []
        time_from_previous_download = 0
        rendered_times = [-1 for _ in range(N)]
        while True:
            if n >= N:
                break

            time_levels_model.append(time_model)

            all_probs = headMovements.get_pose_probs_updates(n)

            history_parameters = self.update_history_probs(all_probs, history_parameters)

            bandwidth_capacity = bandwidth.average_bandwidth(time_model)
            model_action, action_type, action_segment, new_segments, idle_time = model.get_action(all_probs[n],
                                                                                                  bandwidth_capacity,
                                                                                                  n, buffer_array_model,
                                                                                                  downloaded_tiles,
                                                                                                  wait_time,
                                                                                                  delta)

            if new_segments > 0:
                if action_segment not in history_parameters:
                    history_parameters[action_segment] = {}
                    history_parameters[action_segment][
                        'bitrate'] = model_action
                else:
                    for i in range(len(model_action)):
                        if model_action[i] > 0:
                            history_parameters[action_segment]['bitrate'][i] = model_action[i]
                history_parameters[action_segment]['probs'] = all_probs[action_segment]

            model_solutions.append(model_action)

            if np.sum(model_action) > 0:
                total_size, download_n = self.get_total_size(model_action, video)
                download_time_model = bandwidth.download_time(total_size, time_model)
                if __print__:
                    print("[Flare]: download finished for segment: {0} of batch {1}".format(n, batch))

                if action_type == "new":
                    downloaded_tiles[action_segment] = download_n
                    # new_segment_downloaded = download_n
                if action_type == "replacement":
                    downloaded_tiles[action_segment] += new_segments
                    # new_segment_downloaded = new_segments
                model.take_action(model_action, action_segment, time_model)
                model_play_rates.append(
                    (time_model + download_time_model, self.convert_action_to_rates(model_action, sizes, delta)))

            else:
                download_n = 0
                download_time_model = idle_time
                model.take_action(model_action, action_segment, time_model)

            time_taken = max(download_time_model,
                             (download_n - buffer_size) * delta + buffer_model - download_time_model)

            time_model += time_taken
            if np.sum(model_action) > 0:
                available_content[n] = time_model

            prob = all_probs[action_segment] if action_segment >= 0 else [0 for _ in range(D)]
            reward_utility, reward_smooth, reward_val_model, buffer_model, reb, buffer_array_model, played_time_model, played_segment, rendered_times = self.evaluation(
                model_action, n, new_segments, delta, beta, prob, video, gamma,
                buffer_array_model,
                time_model, time_model - recent_time_model, rebuffer_model, played_time_model, downloaded_tiles,
                            time_model - time_from_previous_download, rendered_times)

            self.clean_history(history_parameters, int(played_segment))
            recent_time_model = time_model
            if np.sum(model_action) > 0:
                n += 1
                time_from_previous_download = time_model
                bitrate_dic = self.convert_action_to_dic(model_action, D)
                request = Request("new", n - 1, time_model - time_taken, time_model, bitrate_dic)

            else:
                request = Request("wait", -1, time_model - time_taken, time_model, {})
            requests.append(request)


            model.set_buffer(buffer_model / delta)

            rebuffer_model += reb
            total_reward_utility += reward_utility
            total_reward_smooth += reward_smooth
            total_reward_val_model += reward_val_model

            buffer_values.log_buffer(time_model, buffer_model)

        time_levels_model.append(time_model)
        self.finalize_rendered_times(rendered_times, delta)
        result = {}
        result['solution'] = model_solutions
        result['time'] = time_levels_model

        result['buffer'] = buffer_values.buffer_logs
        result['reward_utility'] = float(total_reward_utility / time_model)
        result['reward_smooth'] = float(total_reward_smooth / time_model)
        result['final_time'] = float(time_model)
        result['playing_br'] = model_play_rates
        result['rebuff'] = float(rebuffer_model)
        result['reward_val'] = total_reward_val_model
        result['rendered_times'] = rendered_times
        result['requests'] = [x.__getArray__() for x in requests]

        with open(result_path, 'w') as writer:
            json.dump(result, writer)
        if __print_QoE__:
            print("QoE: {0}".format(total_reward_utility + total_reward_smooth))

    def run_PANO(self, batch, path_to_meta, path_to_bandwidth, path_to_navGraph, result_path,
                 bandwidth_series=1,
                 __print__=True, __start_time__=0, __print_QoE__=False):
        if __print__:
            print("Pano starts work for batch {0}".format(batch))
        meta = self.read_json(path_to_meta)
        N = meta['N']
        D = meta['D']
        buffer_size = meta['b_max']
        delta = meta['delta']
        gamma = meta['gamma']
        beta = meta['beta']
        t_0 = meta['t_0']
        wait_time = meta['wait']
        sizes = np.array(meta['sizes'])
        bandwidth_error = meta['b_error']
        v_coeff = meta['V']
        actual_movement = meta['view']
        values = meta['values']
        hetro = meta['hetro']
        positive_tiles = meta['pos_tiles']
        bandwidth_target = meta['bandwidth_target']

        video = Video(N, delta, D, values, sizes, buffer_size)
        if bandwidth_series == 2:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, series=2,
                                  target_average=bandwidth_target)
        else:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, target_average=bandwidth_target)

        headMovements = HeadMoves(N, D, path=path_to_navGraph)
        if hetro >= 0:
            headMovements.activate_hetrogen_model(hetro, positive_tiles)
        headMovements.set_actual_views(actual_movement)

        model_solutions = []

        available_content = [-1 for _ in range(N)]
        buffer_array_model = [0 for _ in range(N)]
        time_levels_model = []

        model_play_rates = []
        time_model = __start_time__
        buffer_model = 0  # unit: time
        buffer_values = Buffer()

        rebuffer_model = 0
        total_reward_utility = 0
        total_reward_smooth = 0
        total_reward_val_model = 0
        played_time_model = 0
        downloaded_tiles = [0 for _ in range(N)]

        params = {}
        params['buffer_capacity'] = buffer_size
        model = Pano(video, params)

        history_parameters = {}

        # model algorithm
        n = 0
        recent_time_model = 0
        requests = []
        time_from_previous_download = 0
        rendered_times = [-1 for _ in range(N)]
        while True:
            if n >= N:
                break

            time_levels_model.append(time_model)

            all_probs = headMovements.get_pose_probs_updates(n)

            history_parameters = self.update_history_probs(all_probs, history_parameters)


            bandwidth_capacity = bandwidth.average_bandwidth(time_model)
            model_action, action_type, action_segment, new_segments, idle_time = model.get_action(all_probs[n],
                                                                                                  bandwidth_capacity,
                                                                                                  n, buffer_array_model,
                                                                                                  downloaded_tiles,
                                                                                                  wait_time,
                                                                                                  delta)

            if new_segments > 0:
                if action_segment not in history_parameters:
                    history_parameters[action_segment] = {}
                    history_parameters[action_segment][
                        'bitrate'] = model_action
                else:
                    for i in range(len(model_action)):
                        if model_action[i] > 0:
                            history_parameters[action_segment]['bitrate'][i] = model_action[i]
                history_parameters[action_segment]['probs'] = all_probs[action_segment]

            model_solutions.append(model_action)


            if np.sum(model_action) > 0:
                total_size, download_n = self.get_total_size(model_action, video)
                download_time_model = bandwidth.download_time(total_size, time_model)
                if __print__:
                    print("[Pano]: download finished for segment: {0} of batch {1}".format(n, batch))

                if action_type == "new":
                    downloaded_tiles[action_segment] = download_n
                    # new_segment_downloaded = download_n
                if action_type == "replacement":
                    downloaded_tiles[action_segment] += new_segments
                    # new_segment_downloaded = new_segments
                model.take_action(model_action, action_segment, time_model)
                model_play_rates.append(
                    (time_model + download_time_model, self.convert_action_to_rates(model_action, sizes, delta)))

            else:
                download_n = 0
                download_time_model = idle_time
                model.take_action(model_action, action_segment, time_model)

            time_taken = max(download_time_model,
                             (download_n - buffer_size) * delta + buffer_model - download_time_model)

            time_model += time_taken
            if np.sum(model_action) > 0:
                available_content[n] = time_model

            prob = all_probs[action_segment] if action_segment >= 0 else [0 for _ in range(D)]
            reward_utility, reward_smooth, reward_val_model, buffer_model, reb, buffer_array_model, played_time_model, played_segment, rendered_times = self.evaluation(
                model_action, n, new_segments, delta, beta, prob, video, gamma,
                buffer_array_model,
                time_model, time_model - recent_time_model, rebuffer_model, played_time_model, downloaded_tiles,
                            time_model - time_from_previous_download, rendered_times)

            self.clean_history(history_parameters, int(played_segment))
            recent_time_model = time_model
            if np.sum(model_action) > 0:
                n += 1
                time_from_previous_download = time_model
                bitrate_dic = self.convert_action_to_dic(model_action, D)
                request = Request("new", n - 1, time_model - time_taken, time_model, bitrate_dic)

            else:
                request = Request("wait", -1, time_model - time_taken, time_model, {})
            requests.append(request)



            model.set_buffer(buffer_model / delta)

            rebuffer_model += reb
            total_reward_utility += reward_utility
            total_reward_smooth += reward_smooth
            total_reward_val_model += reward_val_model

            buffer_values.log_buffer(time_model, buffer_model)

        time_levels_model.append(time_model)
        self.finalize_rendered_times(rendered_times, delta)
        result = {}
        result['solution'] = model_solutions
        result['time'] = time_levels_model

        result['buffer'] = buffer_values.buffer_logs
        result['reward_utility'] = float(total_reward_utility / time_model)
        result['reward_smooth'] = float(total_reward_smooth / time_model)
        result['final_time'] = float(time_model)
        result['playing_br'] = model_play_rates
        result['rebuff'] = float(rebuffer_model)
        result['reward_val'] = total_reward_val_model
        result['rendered_times'] = rendered_times
        result['requests'] = [x.__getArray__() for x in requests]

        with open(result_path, 'w') as writer:
            json.dump(result, writer)
        if __print_QoE__:
            print("QoE: {0}".format(total_reward_utility + total_reward_smooth))

    def run_MOSAIC(self, batch, path_to_meta, path_to_bandwidth, path_to_navGraph, result_path,
                   bandwidth_series=1,
                   __print__=True, __start_time__=0, __print_QoE__=False):
        if __print__:
            print("Mosaic starts work for batch {0}".format(batch))
        meta = self.read_json(path_to_meta)
        N = meta['N']
        D = meta['D']
        buffer_size = meta['b_max']
        delta = meta['delta']
        gamma = meta['gamma']
        beta = meta['beta']
        t_0 = meta['t_0']
        wait_time = meta['wait']
        sizes = np.array(meta['sizes'])
        bandwidth_error = meta['b_error']
        v_coeff = meta['V']
        actual_movement = meta['view']
        values = meta['values']
        hetro = meta['hetro']
        positive_tiles = meta['pos_tiles']
        bandwidth_target = meta['bandwidth_target']

        video = Video(N, delta, D, values, sizes, buffer_size)
        if bandwidth_series == 2:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, series=2,
                                  target_average=bandwidth_target)
        else:
            bandwidth = Bandwidth(path_to_bandwidth, error_rate=bandwidth_error, target_average=bandwidth_target)

        headMovements = HeadMoves(N, D, path=path_to_navGraph)
        if hetro >= 0:
            headMovements.activate_hetrogen_model(hetro, positive_tiles)
        headMovements.set_actual_views(actual_movement)

        # model_performance = []
        model_solutions = []
        # buffer_levels_model = []
        available_content = [-1 for _ in range(N)]
        buffer_array_model = [0 for _ in range(N)]
        time_levels_model = []

        model_play_rates = []
        time_model = __start_time__
        buffer_model = 0  # unit: time
        buffer_values = Buffer()

        rebuffer_model = 0
        total_reward_utility = 0
        total_reward_smooth = 0
        total_reward_val_model = 0
        played_time_model = 0
        downloaded_tiles = [0 for _ in range(N)]

        params = {}
        params['buffer_capacity'] = buffer_size
        params['mu1'] = 2
        params['mu2'] = 2
        params['mu3'] = 1
        params['mu4'] = 1
        model = Mosaic(video, params)

        history_parameters = {}

        # model algorithm
        n = 0
        recent_time_model = 0
        requests = []
        time_from_previous_download = 0
        rendered_times = [-1 for _ in range(N)]
        while True:
            if n >= N:
                break

            time_levels_model.append(time_model)

            all_probs = headMovements.get_pose_probs_updates(n)

            history_parameters = self.update_history_probs(all_probs, history_parameters)

            bandwidth_capacity = bandwidth.average_bandwidth(time_model)
            model_action, action_type, action_segment, new_segments, idle_time = model.get_action(all_probs[n],
                                                                                                  bandwidth_capacity,
                                                                                                  n, buffer_array_model,
                                                                                                  downloaded_tiles,
                                                                                                  wait_time,
                                                                                                  delta)

            if new_segments > 0:
                if action_segment not in history_parameters:
                    history_parameters[action_segment] = {}
                    history_parameters[action_segment][
                        'bitrate'] = model_action
                else:
                    for i in range(len(model_action)):
                        if model_action[i] > 0:
                            history_parameters[action_segment]['bitrate'][i] = model_action[i]
                history_parameters[action_segment]['probs'] = all_probs[action_segment]

            model_solutions.append(model_action)

            if np.sum(model_action) > 0:
                total_size, download_n = self.get_total_size(model_action, video)
                download_time_model = bandwidth.download_time(total_size, time_model)
                if __print__:
                    print("[Mosaic]: download finished for segment: {0} of batch {1}".format(n, batch))

                if action_type == "new":
                    downloaded_tiles[action_segment] = download_n
                    # new_segment_downloaded = download_n
                if action_type == "replacement":
                    downloaded_tiles[action_segment] += new_segments
                    # new_segment_downloaded = new_segments
                model.take_action(model_action, action_segment, time_model)
                model_play_rates.append(
                    (time_model + download_time_model, self.convert_action_to_rates(model_action, sizes, delta)))

            else:
                download_n = 0
                download_time_model = idle_time
                model.take_action(model_action, action_segment, time_model)

            time_taken = max(download_time_model,
                             (download_n - buffer_size) * delta + buffer_model - download_time_model)

            time_model += time_taken
            if np.sum(model_action) > 0:
                available_content[n] = time_model

            prob = all_probs[action_segment] if action_segment >= 0 else [0 for _ in range(D)]
            reward_utility, reward_smooth, reward_val_model, buffer_model, reb, buffer_array_model, played_time_model, played_segment, rendered_times = self.evaluation(
                model_action, n, new_segments, delta, beta, prob, video, gamma,
                buffer_array_model,
                time_model, time_model - recent_time_model, rebuffer_model, played_time_model, downloaded_tiles,
                            time_model - time_from_previous_download, rendered_times)

            self.clean_history(history_parameters, int(played_segment))
            recent_time_model = time_model
            if np.sum(model_action) > 0:
                n += 1
                time_from_previous_download = time_model
                bitrate_dic = self.convert_action_to_dic(model_action, D)
                request = Request("new", n - 1, time_model - time_taken, time_model, bitrate_dic)

            else:
                request = Request("wait", -1, time_model - time_taken, time_model, {})
            requests.append(request)

            model.set_buffer(buffer_model / delta)

            rebuffer_model += reb
            total_reward_utility += reward_utility
            total_reward_smooth += reward_smooth
            total_reward_val_model += reward_val_model

            buffer_values.log_buffer(time_model, buffer_model)

        time_levels_model.append(time_model)
        self.finalize_rendered_times(rendered_times, delta)
        result = {}
        result['solution'] = model_solutions
        result['time'] = time_levels_model

        result['buffer'] = buffer_values.buffer_logs
        result['reward_utility'] = float(total_reward_utility / time_model)
        result['reward_smooth'] = float(total_reward_smooth / time_model)
        result['final_time'] = float(time_model)
        result['playing_br'] = model_play_rates
        result['rebuff'] = float(rebuffer_model)
        result['reward_val'] = total_reward_val_model
        result['rendered_times'] = rendered_times
        result['requests'] = [x.__getArray__() for x in requests]

        with open(result_path, 'w') as writer:
            json.dump(result, writer)
        if __print_QoE__:
            print("QoE: {0}".format(total_reward_utility + total_reward_smooth))
