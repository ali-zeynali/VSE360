import glob

from Evaluate import *
from VideoPlayer import *


def save_meta(N, D, buffer_size, delta, beta, gamma, t_0, wait_time, sizes, values, b_error, v_coeff, actual_view,
              hetro, pos_tiles, eta, target_buffer, bandwidth_target,
              path='results/meta.json'):
    data = {}
    data['N'] = N
    data['D'] = D
    data['b_max'] = buffer_size
    data['delta'] = delta
    data['gamma'] = gamma
    data['beta'] = beta
    data['t_0'] = t_0
    data['wait'] = wait_time
    data['sizes'] = [float(x) for x in sizes]
    data['b_error'] = b_error
    data['V'] = v_coeff
    data['view'] = [int(x) for x in actual_view]
    data['hetro'] = hetro
    data['pos_tiles'] = pos_tiles
    data['eta'] = eta
    data['target_buffer'] = target_buffer
    data['values'] = [float(v) for v in values]
    data['bandwidth_target'] = bandwidth_target
    with open(path, 'w') as writer:
        json.dump(data, writer)


def replace_meta(path, v_coeff):
    data = read_meta(path)
    data['V'] = v_coeff
    with open(path, 'w') as writer:
        json.dump(data, writer)


def get_sample_paths():
    folders = glob.glob("results/sample*")
    paths = []
    for folder in folders:
        index = folder.split("_")[1]
        paths.append(["results/sample_{0}".format(index), int(index)])
    return paths


def read_meta(path):
    with open(path) as reader:
        data = json.load(reader)
    return data


def get_available_times(solution, time):
    available_time = []
    tiles = []
    downloaded_bitrates = []
    for i in range(len(solution) - 1):
        n = 0
        for a in solution[i]:
            if a > 0:
                n += 1
        if n < 1:
            continue
        tiles.append(n)
        available_time.append(time[i + 1])
        downloaded_bitrates.append(solution[i])
    return available_time, tiles, downloaded_bitrates


def get_playing_bitrates(solution, buffer, time, actual_headmovement, bitrates, delta, values, dt=0.05,
                         __start_time__=0):
    available_time, tiles, downloaded_bitrates = get_available_times(solution, time)

    watching_bitrate = []
    time_slots = []
    getting_values = []

    playing_segment = -1
    portion_remained = 0
    t = __start_time__
    buffer_level = 0
    play_rate = 0
    downloaded_segment = -1

    rebuff = 0
    buffer_time_stamps = [k for k in buffer.keys()]
    while t < available_time[-1] or buffer_level > 0:

        # add downloaded segment to buffer
        # if downloaded_segment < len(available_time) - 1 and t >= available_time[downloaded_segment + 1]:
        #     buffer_level = np.maximum(buffer[downloaded_segment + 1], 0)
        #     downloaded_segment += 1

        if downloaded_segment < len(buffer_time_stamps) - 1 and t >= float(buffer_time_stamps[downloaded_segment + 1]):
            buffer_logged_time = buffer_time_stamps[downloaded_segment + 1]
            buffer_level = buffer[buffer_logged_time]

        # go to next segment if available
        if portion_remained <= 0 and playing_segment < len(available_time) - 1 and t >= available_time[
            playing_segment + 1]:
            playing_segment += 1
            portion_remained = delta
            play_rate = tiles[playing_segment]

        watching_tile = actual_headmovement[playing_segment]

        # if segment has not downloaded yet
        if t < available_time[playing_segment]:
            rebuff += dt
            watching_bitrate.append(0)
            time_slots.append(t)
            getting_values.append(0)

        # if the watching tile is not in list of downloaded segments
        elif portion_remained > 0 and t >= available_time[playing_segment] and downloaded_bitrates[playing_segment][
            watching_tile] == 0:
            rebuff += dt
            watching_bitrate.append(
                -5)  # Just to make a difference between waiting to download the segment and downloading wrong tiles
            time_slots.append(t)
            getting_values.append(0)

        elif portion_remained > 0 and t >= available_time[playing_segment]:
            r_indx = downloaded_bitrates[playing_segment][watching_tile]
            r = bitrates[r_indx]
            watching_bitrate.append(r)
            time_slots.append(t)
            getting_values.append(values[r_indx])

        # if segment has not downloaded yet
        elif portion_remained <= 0 and t >= available_time[playing_segment]:
            rebuff += dt
            watching_bitrate.append(0)
            time_slots.append(t)
            getting_values.append(0)

        buffer_level -= play_rate * dt
        buffer_level = np.maximum(buffer_level, 0)
        portion_remained -= dt

        t += dt

    avg_wbr = get_average_watching_bitrate(watching_bitrate)
    avg_wv = np.average(getting_values)
    return rebuff, watching_bitrate, time_slots, avg_wbr, avg_wv


def batch_get_attr(path, attr, batch):
    ddp = read_dataset(path + "/ddp_{0}.json".format(batch))
    cBola = read_dataset(path + "/CBola_{0}.json".format(batch))
    # naive1 = read_dataset(path + "/naive_1_{0}.json".format( batch))
    naive_full = read_dataset(path + "/naive_8_{0}.json".format(batch))
    vaware = read_dataset(path + "/VAware_{0}.json".format(batch))
    probdash = read_dataset(path + "/ProbDash_{0}.json".format(batch))
    salient = read_dataset(path + "/SalientVR_{0}.json".format(batch))
    flare = read_dataset(path + "/Flare_{0}.json".format(batch))
    pano = read_dataset(path + "/Pano_{0}.json".format(batch))
    mosaic = read_dataset(path + "/Mosaic_{0}.json".format(batch))

    rewards = {}
    rewards['ddp'] = ddp[attr]
    rewards['cBola'] = cBola[attr]
    # rewards['naive1'] = naive1[attr]
    rewards['naive_f'] = naive_full[attr]
    rewards['vaware'] = vaware[attr]
    rewards['probdash'] = probdash[attr]
    rewards['SalientVR'] = salient[attr]
    rewards['flare'] = flare[attr]
    rewards['pano'] = pano[attr]
    rewards['mosaic'] = mosaic[attr]

    return rewards


def batch_real_evaluation(batch, delta, values, __start_time__=0, __print__=False):
    # fig = plt.figure(int(np.random.random() * 10000))

    path = "results/BatchRun/"

    meta_data = read_dataset(path + "/meta_{0}.json".format(batch))
    actual_headmovement = meta_data['view']
    bitrates = np.array(meta_data['sizes']) / delta

    buffers = batch_get_attr(path, 'buffer', batch)
    times = batch_get_attr(path, 'time', batch)
    solution = batch_get_attr(path, 'solution', batch)

    ddp_buff = buffers['ddp']
    cBola_buff = buffers['cBola']
    # naive1_buff = buffers['naive1']
    vaware_buff = buffers['vaware']
    naive_full_buff = buffers['naive_f']
    probdash_buff = buffers['probdash']
    salient_buff = buffers['SalientVR']
    flare_buff = buffers['flare']
    pano_buff = buffers['pano']
    mosaic_buff = buffers['mosaic']

    ddp_time = times['ddp']
    cBola_time = times['cBola']
    # naive1_time = times['naive1']
    vaware_time = times['vaware']
    naive_full_time = times['naive_f']
    probdash_time = times['probdash']
    salient_time = times['SalientVR']
    flare_time = times['flare']
    pano_time = times['pano']
    mosaic_time = times['mosaic']

    ddp_solution = solution['ddp']
    cBola_solution = solution['cBola']
    # naive1_solution = solution['naive1']
    vaware_solution = solution['vaware']
    naivf_solution = solution['naive_f']
    probdash_solution = solution['probdash']
    salient_solution = solution['SalientVR']
    flare_solution = solution['flare']
    pano_solution = solution['pano']
    mosaic_solution = solution['mosaic']

    ddp_rebuff, ddp_y, ddp_x, ddp_avg_btr, ddp_avg_wv = get_playing_bitrates(ddp_solution, ddp_buff, ddp_time,
                                                                             actual_headmovement, bitrates, delta,
                                                                             values,
                                                                             __start_time__=__start_time__)
    cbola_rebuff, cbola_y, cbola_x, cbola_avg_btr, cbola_avg_wv = get_playing_bitrates(cBola_solution, cBola_buff,
                                                                                       cBola_time,
                                                                                       actual_headmovement, bitrates,
                                                                                       delta,
                                                                                       values,
                                                                                       __start_time__=__start_time__)
    # naive1_rebuff, naive1_y, naive1_x, naive1_avg_btr, naive1_avg_wv = get_playing_bitrates(naive1_solution,
    #                                                                                         naive1_buff, naive1_time,
    #                                                                                         actual_headmovement,
    #                                                                                         bitrates, delta, values,
    #                                                                                         __start_time__=__start_time__)
    vaware_rebuff, vaware_y, vaware_x, vaware_avg_btr, vaware_avg_wv = get_playing_bitrates(vaware_solution,
                                                                                            vaware_buff,
                                                                                            vaware_time,
                                                                                            actual_headmovement,
                                                                                            bitrates,
                                                                                            delta, values,
                                                                                            __start_time__=__start_time__)
    naivf_rebuff, naivf_y, naivf_x, naivf_avg_btr, naivf_avg_wv = get_playing_bitrates(naivf_solution, naive_full_buff,
                                                                                       naive_full_time,
                                                                                       actual_headmovement, bitrates,
                                                                                       delta, values,
                                                                                       __start_time__=__start_time__)

    probdash_rebuff, probdash_y, probdash_x, probdash_avg_btr, probdash_avg_wv = get_playing_bitrates(probdash_solution,
                                                                                                      probdash_buff,
                                                                                                      probdash_time,
                                                                                                      actual_headmovement,
                                                                                                      bitrates,
                                                                                                      delta, values,
                                                                                                      __start_time__=__start_time__)

    salient_rebuff, salient_y, salient_x, salient_avg_btr, salient_avg_wv = get_playing_bitrates(salient_solution,
                                                                                                 salient_buff,
                                                                                                 salient_time,
                                                                                                 actual_headmovement,
                                                                                                 bitrates,
                                                                                                 delta, values,
                                                                                                 __start_time__=__start_time__)

    flare_rebuff, flare_y, flare_x, flare_avg_btr, flare_avg_wv = get_playing_bitrates(flare_solution,
                                                                                       flare_buff,
                                                                                       flare_time,
                                                                                       actual_headmovement,
                                                                                       bitrates,
                                                                                       delta, values,
                                                                                       __start_time__=__start_time__)

    pano_rebuff, pano_y, pano_x, pano_avg_btr, pano_avg_wv = get_playing_bitrates(pano_solution,
                                                                                  pano_buff,
                                                                                  pano_time,
                                                                                  actual_headmovement,
                                                                                  bitrates,
                                                                                  delta, values,
                                                                                  __start_time__=__start_time__)

    mosaic_rebuff, mosaic_y, mosaic_x, mosaic_avg_btr, mosaic_avg_wv = get_playing_bitrates(mosaic_solution,
                                                                                            mosaic_buff,
                                                                                            mosaic_time,
                                                                                            actual_headmovement,
                                                                                            bitrates,
                                                                                            delta, values,
                                                                                            __start_time__=__start_time__)

    actual_bitrates = {}
    actual_bitrates['ddp'] = ddp_avg_btr
    actual_bitrates['cbola'] = cbola_avg_btr
    # actual_bitrates['naive-1'] = naive1_avg_btr
    actual_bitrates['naive-f'] = naivf_avg_btr
    actual_bitrates['vaware'] = vaware_avg_btr
    actual_bitrates['probdash'] = probdash_avg_btr
    actual_bitrates['SalientVR'] = salient_avg_btr
    actual_bitrates['flare'] = flare_avg_btr
    actual_bitrates['pano'] = pano_avg_btr
    actual_bitrates['mosaic'] = mosaic_avg_btr

    rebuffering = {}
    rebuffering['ddp'] = ddp_rebuff
    rebuffering['cbola'] = cbola_rebuff
    # rebuffering['naive-1'] = naive1_rebuff
    rebuffering['naive-f'] = naivf_rebuff
    rebuffering['vaware'] = vaware_rebuff
    rebuffering['probdash'] = probdash_rebuff
    rebuffering['SalientVR'] = salient_rebuff
    rebuffering['flare'] = flare_rebuff
    rebuffering['pano'] = pano_rebuff
    rebuffering['mosaic'] = mosaic_rebuff

    viewed_values = {}
    viewed_values['ddp'] = ddp_avg_wv
    viewed_values['cbola'] = cbola_avg_wv
    # viewed_values['naive-1'] = naive1_avg_wv
    viewed_values['naive-f'] = naivf_avg_wv
    viewed_values['vaware'] = vaware_avg_wv
    viewed_values['probdash'] = probdash_avg_wv
    viewed_values['SalientVR'] = salient_avg_wv
    viewed_values['flare'] = flare_avg_wv
    viewed_values['pano'] = pano_avg_wv
    viewed_values['mosaic'] = mosaic_avg_wv

    if __print__:
        print("\n<------\tComparing Actual bitrates\t------>")
        print("C-Bola Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(cbola_avg_btr,
                                                                                                      cbola_rebuff,
                                                                                                      cbola_avg_wv))

        print("DDP Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(ddp_avg_btr,
                                                                                                   ddp_rebuff,
                                                                                                   ddp_avg_wv))
        # print("Naive1 Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(naive1_avg_btr,
        #                                                                                               naive1_rebuff,
        #                                                                                               naive1_avg_wv))
        print("Naive-full Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(naivf_avg_btr,
                                                                                                          naivf_rebuff,
                                                                                                          naivf_avg_wv))
        print("VAware Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(vaware_avg_btr,
                                                                                                      vaware_rebuff,
                                                                                                      vaware_avg_wv))
        print(
            "ProbDash Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(probdash_avg_btr,
                                                                                                      probdash_rebuff,
                                                                                                      probdash_avg_wv))

        print(
            "SalientVR Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(salient_avg_btr,
                                                                                                       salient_rebuff,
                                                                                                       salient_avg_wv))

        print(
            "Flare Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(flare_avg_btr,
                                                                                                   flare_rebuff,
                                                                                                   flare_avg_wv))

        print(
            "Pano Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(pano_avg_btr,
                                                                                                  pano_rebuff,
                                                                                                  pano_avg_wv))
        print(
            "Mosaic Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(mosaic_avg_btr,
                                                                                                    mosaic_rebuff,
                                                                                                    mosaic_avg_wv))

        print("\n<------\t ----------------- \t------>")
    return actual_bitrates, rebuffering, viewed_values


def batch_compare_rewards(batch, delta=5, __print__=True, __start_time__=0):
    path = "results/BatchRun"
    rewards_utility = batch_get_attr(path, 'reward_utility', batch)
    rewards_smooth = batch_get_attr(path, 'reward_smooth', batch)

    ddp = rewards_utility['ddp'] + rewards_smooth['ddp']
    cBola = rewards_utility['cBola'] + rewards_smooth['cBola']
    # naive1 = rewards_utility['naive1'] + rewards_smooth['naive1']
    naive_full = rewards_utility['naive_f'] + rewards_smooth['naive_f']
    vaware = rewards_utility['vaware'] + rewards_smooth['vaware']
    probdash = rewards_utility['probdash'] + rewards_smooth['probdash']
    salientVR = rewards_utility['SalientVR'] + rewards_smooth['SalientVR']
    flare = rewards_utility['flare'] + rewards_smooth['flare']
    pano = rewards_utility['pano'] + rewards_smooth['pano']
    mosaic = rewards_utility['mosaic'] + rewards_smooth['mosaic']

    if __print__:
        print("\n<------\tComparing Rewards\t------>")
        print("C-Bola Average utility: {0}".format(cBola))
        print("DDP Average utility: {0}".format(ddp))
        # print("Naive-1 Average utility: {0}".format(naive1))
        print("Naive-full Average utility: {0}".format(naive_full))
        print("VAware Average utility: {0}".format(vaware))
        print("ProbDash Average utility: {0}".format(probdash))
        print("SalientVR Average utility: {0}".format(salientVR))
        print("Flare Average utility: {0}".format(flare))
        print("Pano Average utility: {0}".format(pano))
        print("Mosaic Average utility: {0}".format(mosaic))

    return [rewards_utility['ddp'], rewards_smooth['ddp']], [rewards_utility['cBola'], rewards_smooth['cBola']], \
           [rewards_utility['naive_f'], rewards_smooth['naive_f']], \
           [rewards_utility['vaware'], rewards_smooth['vaware']], [rewards_utility['probdash'],
                                                                   rewards_smooth['probdash']], [
               rewards_utility['SalientVR'], rewards_smooth['SalientVR']], [rewards_utility['flare'],
                                                                            rewards_smooth['flare']], [
               rewards_utility['pano'], rewards_smooth['pano']], [rewards_utility['mosaic'], rewards_smooth['mosaic']]


N = 250
D = 8

buffer_size = D * 8 * 2
delta = 2
gamma = 0.2
t_0 = delta / 10
wait_time = t_0

bandwidth_dir = "dataset/bandwidth/4G/4G_BWData.json"

bandwidth_error = 0.10
target_buffer = 0.7 * buffer_size * delta
buffer_threshold = 2
bandwidth_target = 20
eta = 0.2
beta = 0

sizes = np.array([0, 0.44, 0.7, 1.35, 2.14, 4.1, 8.2, 16.5]) * delta
M = len(sizes) - 1

__start_time__ = 0

v_coeff = 0.95
v_alpha = 2
values = np.array([0 if x == 0 else np.log(v_alpha * x / sizes[1]) for x in sizes])

path_to_navGraph = "dataset/headmovement/navG1.json"

algs = ['cBola', 'ddp', 'naive-f', 'vaware', 'probdash', 'SalientVR', 'flare', 'pano', 'mosaic']
played_bitrates = {}
played_vals = {}
played_rebuff = {}
model_obj_utility = {}
model_obj_smooth = {}
endTimes = {}
rendered_times = {}
for alg in algs:
    played_bitrates[alg] = []
    played_vals[alg] = []
    played_rebuff[alg] = []
    model_obj_utility[alg] = []
    model_obj_smooth[alg] = []
    endTimes[alg] = []
    rendered_times[alg] = []

batches = 100

__save_meta__ = False
__replace_meta__ = False
__run_tests__ = True

videoPlayer = VideoPlayer()

for batch in range(batches):
    print("** Starting Analysis of Batch {0} / {1}".format(batch + 1, batches))

    headMovements = HeadMoves(N, D, path=path_to_navGraph)
    actual_movement = headMovements.get_sample_actual_view()
    headMovements.set_actual_views(actual_movement)
    sample_path = "results/BatchRun/"

    path_to_meta = sample_path + "meta_{0}.json".format(batch)

    if __save_meta__:
        save_meta(N, D, buffer_size, delta, beta, gamma, t_0, wait_time, sizes, values, bandwidth_error, v_coeff,
                  actual_movement, -1, D, eta, target_buffer, bandwidth_target, path=path_to_meta)

    if __replace_meta__:
        replace_meta(path_to_meta, v_coeff)

    if __run_tests__:
        videoPlayer.run_BOLA360(batch, path_to_meta, bandwidth_dir, path_to_navGraph,
                                sample_path + "CBola_{0}.json".format(batch),
                                __print__=False, __start_time__=0, __print_QoE__=False, given_gamma=-1,
                                bandwidth_series=1)
        #
        videoPlayer.run_DDP(batch, path_to_meta, bandwidth_dir, path_to_navGraph,
                            sample_path + "ddp_{0}.json".format(batch),
                            sample_path + "ddp_{0}.json".format(0),
                            base_comparision=True, __print__=False, __start_time__=0,
                            bandwidth_series=1)

        videoPlayer.run_NAIVE(batch, path_to_meta, bandwidth_dir, path_to_navGraph,
                              sample_path + "naive_8_{0}.json".format(batch),
                              sample_path + "naive_8_{0}.json".format(0), 8, False,
                              __print__=False, __start_time__=0, bandwidth_series=1)

        videoPlayer.run_VAware(batch, path_to_meta, bandwidth_dir, path_to_navGraph,
                               sample_path + "VAware_{0}.json".format(batch),
                               __print__=False, __start_time__=0, bandwidth_series=1)

        videoPlayer.run_ProbDash(batch, path_to_meta, bandwidth_dir, path_to_navGraph,
                                 sample_path + "ProbDash_{0}.json".format(batch),
                                 __print__=False, __start_time__=0, bandwidth_series=1)
        #
        videoPlayer.run_SalientVR(batch, path_to_meta,
                                  bandwidth_dir, path_to_navGraph,
                                  sample_path + "SalientVR_{0}.json".format(batch),
                                  buffer_threshold, bandwidth_series=1, __print_QoE__=False,
                                  __print__=False, __start_time__=0)

        videoPlayer.run_FLARE(batch, path_to_meta, bandwidth_dir, path_to_navGraph,
                              sample_path + "Flare_{0}.json".format(batch),
                              __print__=False, __start_time__=0, bandwidth_series=1)

        videoPlayer.run_PANO(batch, path_to_meta, bandwidth_dir, path_to_navGraph,
                             sample_path + "Pano_{0}.json".format(batch),
                             __print__=False, __start_time__=0, bandwidth_series=1)

        videoPlayer.run_MOSAIC(batch, path_to_meta, bandwidth_dir, path_to_navGraph,
                               sample_path + "Mosaic_{0}.json".format(batch),
                               __print__=False, __start_time__=0, bandwidth_series=1)

    path = "results/SampleRun"

    ddp = read_meta(path + "/ddp_{0}.json".format(batch))
    cBola = read_meta(path + "/CBola_{0}.json".format(batch))

    naive_f = read_meta(path + "/naive_8_{0}.json".format(batch))
    vaware = read_meta(path + "/VAware_{0}.json".format(batch))
    probdash = read_meta(path + "/ProbDash_{0}.json".format(batch))
    salientVR = read_meta(path + "/SalientVR_{0}.json".format(batch))
    flare = read_meta(path + "/Flare_{0}.json".format(batch))
    pano = read_meta(path + "/Pano_{0}.json".format(batch))
    mosaic = read_meta(path + "/Mosaic_{0}.json".format(batch))

    algorithms = {}
    algorithms['Bola360'] = cBola
    algorithms['DP-On'] = ddp
    algorithms['Top-D'] = naive_f
    algorithms['VA-360'] = vaware
    algorithms['ProbDash'] = probdash
    algorithms['SalientVR'] = salientVR
    algorithms['flare'] = flare
    algorithms['pano'] = pano
    algorithms['mosaic'] = mosaic

    actual_bitrates, rebuffering, viewed_values = batch_real_evaluation(batch, delta, values,
                                                                        __start_time__=__start_time__, __print__=False)
    ddp_r, cBola_r, naive_full_r, vaware_r, probdash_r, salientVR_r, flare_r, pano_r, mosaic_r = batch_compare_rewards(
        batch, delta=delta,
        __print__=False,
        __start_time__=__start_time__)

    played_bitrates['cBola'].append(actual_bitrates['cbola'])
    played_bitrates['ddp'].append(actual_bitrates['ddp'])

    played_bitrates['naive-f'].append(actual_bitrates['naive-f'])
    played_bitrates['vaware'].append(actual_bitrates['vaware'])
    played_bitrates['probdash'].append(actual_bitrates['probdash'])
    played_bitrates['SalientVR'].append(actual_bitrates['SalientVR'])
    played_bitrates['flare'].append(actual_bitrates['flare'])
    played_bitrates['pano'].append(actual_bitrates['pano'])
    played_bitrates['mosaic'].append(actual_bitrates['mosaic'])

    played_rebuff['ddp'].append(ddp['rebuff'])
    played_rebuff['cBola'].append(cBola['rebuff'])

    played_rebuff['naive-f'].append(naive_f['rebuff'])
    played_rebuff['vaware'].append(vaware['rebuff'])
    played_rebuff['probdash'].append(probdash['rebuff'])
    played_rebuff['SalientVR'].append(salientVR['rebuff'])
    played_rebuff['flare'].append(flare['rebuff'])
    played_rebuff['pano'].append(pano['rebuff'])
    played_rebuff['mosaic'].append(mosaic['rebuff'])

    played_vals['ddp'].append(viewed_values['ddp'])
    played_vals['cBola'].append(viewed_values['cbola'])
    played_vals['naive-f'].append(viewed_values['naive-f'])
    played_vals['vaware'].append(viewed_values['vaware'])
    played_vals['probdash'].append(viewed_values['probdash'])
    played_vals['SalientVR'].append(viewed_values['SalientVR'])
    played_vals['flare'].append(viewed_values['flare'])
    played_vals['pano'].append(viewed_values['pano'])
    played_vals['mosaic'].append(viewed_values['mosaic'])

    model_obj_utility['ddp'].append(ddp_r[0])
    model_obj_utility['cBola'].append(cBola_r[0])
    model_obj_utility['naive-f'].append(naive_full_r[0])
    model_obj_utility['vaware'].append(vaware_r[0])
    model_obj_utility['probdash'].append(probdash_r[0])
    model_obj_utility['SalientVR'].append(salientVR_r[0])
    model_obj_utility['flare'].append(flare_r[0])
    model_obj_utility['pano'].append(pano_r[0])
    model_obj_utility['mosaic'].append(mosaic_r[0])

    model_obj_smooth['ddp'].append(ddp_r[1])
    model_obj_smooth['cBola'].append(cBola_r[1])
    model_obj_smooth['naive-f'].append(naive_full_r[1])
    model_obj_smooth['vaware'].append(vaware_r[1])
    model_obj_smooth['probdash'].append(probdash_r[1])
    model_obj_smooth['SalientVR'].append(salientVR_r[1])
    model_obj_smooth['flare'].append(flare_r[1])
    model_obj_smooth['pano'].append(pano_r[1])
    model_obj_smooth['mosaic'].append(mosaic_r[1])

    endTimes['ddp'].append(ddp['final_time'])
    endTimes['cBola'].append(cBola['final_time'])
    endTimes['naive-f'].append(naive_f['final_time'])
    endTimes['vaware'].append(vaware['final_time'])
    endTimes['probdash'].append(probdash['final_time'])
    endTimes['SalientVR'].append(salientVR['final_time'])
    endTimes['flare'].append(flare['final_time'])
    endTimes['pano'].append(pano['final_time'])
    endTimes['mosaic'].append(mosaic['final_time'])

    ddp_download_times = videoPlayer.get_download_times(ddp['solution'], ddp['time'])
    cbola_download_times = videoPlayer.get_download_times(cBola['solution'], cBola['time'])
    naivef_download_times = videoPlayer.get_download_times(naive_f['solution'], naive_f['time'])
    vaware_download_times = videoPlayer.get_download_times(vaware['solution'], vaware['time'])
    probdash_download_times = videoPlayer.get_download_times(probdash['solution'], probdash['time'])
    salientVR_download_times = videoPlayer.get_download_times(salientVR['solution'], salientVR['time'])
    flare_download_times = videoPlayer.get_download_times(flare['solution'], flare['time'])
    pano_download_times = videoPlayer.get_download_times(pano['solution'], pano['time'])
    mosaic_download_times = videoPlayer.get_download_times(mosaic['solution'], mosaic['time'])

    rendered_times['ddp'].append(float(np.average(np.array(ddp['rendered_times']) - np.array(ddp_download_times))))
    rendered_times['cBola'].append(float(np.average(cBola['rendered_times'] - np.array(cbola_download_times))))
    rendered_times['naive-f'].append(float(np.average(naive_f['rendered_times'] - np.array(naivef_download_times))))
    rendered_times['vaware'].append(float(np.average(vaware['rendered_times'] - np.array(vaware_download_times))))
    rendered_times['probdash'].append(float(np.average(probdash['rendered_times'] - np.array(probdash_download_times))))
    rendered_times['SalientVR'].append(
        float(np.average(salientVR['rendered_times'] - np.array(salientVR_download_times))))
    rendered_times['flare'].append(float(np.average(flare['rendered_times'] - np.array(flare_download_times))))
    rendered_times['pano'].append(float(np.average(pano['rendered_times'] - np.array(pano_download_times))))
    rendered_times['mosaic'].append(float(np.average(mosaic['rendered_times'] - np.array(mosaic_download_times))))
with open("results/BatchRun/results.json", 'w') as writer:
    data = {}
    data['bitrate'] = played_bitrates
    data['rebuff'] = played_rebuff
    data['values'] = played_vals
    data['obj_utility'] = model_obj_utility
    data['obj_smooth'] = model_obj_smooth
    data['endTime'] = endTimes
    data['rendered_times'] = rendered_times
    json.dump(data, writer)
