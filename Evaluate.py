import json
from Buffer import *
import numpy as np
from matplotlib import pyplot as plt

from Request import *


def read_dataset(path):
    with open(path) as reader:
        data = json.load(reader)
    return data


def get_attr(path, attr):
    # bola = read_dataset(path + "/bola.json")
    cBola = read_dataset(path + "/CBola.json")
    ddp = read_dataset(path + "/DDP.json")
    naive1 = read_dataset(path + "/naive_1.json")
    naive_full = read_dataset(path + "/naive_8.json")
    vaware = read_dataset(path + "/VAware.json")

    rewards = {}
    # rewards['bola'] = bola[attr]
    rewards['cBola'] = cBola[attr]
    rewards['ddp'] = ddp[attr]
    rewards['naive1'] = naive1[attr]
    rewards['naive_f'] = naive_full[attr]
    rewards['vaware'] = vaware[attr]

    if attr == 'requests':
        for alg in rewards:
            requests = []
            for arr in rewards[alg]:
                requests.append(Request.convert_array_to_Request(arr))
            rewards[alg] = requests
        return rewards
    elif attr == "buffer":
        for alg in rewards:
            rewards[alg] = Buffer.get_buffer_from_logs(rewards[alg])
        return rewards

    else:
        return rewards


def batch_get_attr(path, attr, batch):
    ddp = read_dataset(path + "/ddp_{0}.json".format(batch))
    cBola = read_dataset(path + "/CBola_{0}.json".format(batch))
    naive1 = read_dataset(path + "/naive_1_{0}.json".format(batch))
    naive_full = read_dataset(path + "/naive_8_{0}.json".format(batch))
    vaware = read_dataset(path + "/VAware_{0}.json".format(batch))

    rewards = {}
    rewards['ddp'] = ddp[attr]
    rewards['cBola'] = cBola[attr]
    rewards['naive1'] = naive1[attr]
    rewards['naive_f'] = naive_full[attr]
    rewards['vaware'] = vaware[attr]

    if attr == 'requests':
        for alg in rewards:
            requests = []
            for arr in rewards[alg]:
                requests.append(Request.convert_array_to_Request(arr))
            rewards[alg] = requests
        return rewards
    else:
        return rewards


def compare_rewards(delta=5, number_of_samples=4, __print__=True, __start_time__=0):
    cBola = []
    ddp = []
    naive1 = []
    naive_full = []
    vaware = []

    for sample in range(number_of_samples):
        path = "results/sample_{0}".format(sample)
        rewards = get_attr(path, 'reward')

        cBola.append(rewards['cBola'])
        ddp.append(rewards['ddp'])
        naive1.append(rewards['naive1'])
        vaware.append(rewards['vaware'])
        naive_full.append(rewards['naive_f'])

    if __print__:
        print("\n<------\tComparing Rewards\t------>")
        print("C-Bola Average utility: {0}".format(np.average(cBola)))
        print("DP-Online Average utility: {0}".format(np.average(ddp)))
        print("Naive1 Average utility: {0}".format(np.average(naive1)))
        print("Naive-full Average utility: {0}".format(np.average(naive_full)))
        print("VAware Average utility: {0}".format(np.average(vaware)))

    return cBola, ddp, naive1, naive_full, vaware


def batch_compare_rewards(batch, delta=5, __print__=True, __start_time__=0):
    ddp = []
    cBola = []
    naive1 = []
    naive_full = []
    vaware = []

    path = "results/BatchRun"
    rewards = batch_get_attr(path, 'reward', batch)

    ddp.append(rewards['ddp'])
    cBola.append(rewards['cBola'])
    naive1.append(rewards['naive1'])
    # naive_half.append(rewards['naive_h'])
    naive_full.append(rewards['naive_f'])
    vaware.append(rewards['vaware'])

    if __print__:
        print("\n<------\tComparing Rewards\t------>")
        print("C-Bola Average utility: {0}".format(np.average(cBola)))
        print("DDP Average utility: {0}".format(np.average(ddp)))
        print("Naive-1 Average utility: {0}".format(np.average(naive1)))
        # print("Naive-half Average utility: {0}".format(np.average(naive_half)))
        print("Naive-full Average utility: {0}".format(np.average(naive_full)))

    return ddp, cBola, naive1, naive_full, vaware


def batch_compare_rewards_val(batch, __print__=True):
    ddp = []
    cBola = []
    naive1 = []

    naive_full = []
    vaware = []

    path = "results/BatchRun"
    rewards = batch_get_attr(path, 'reward_val', batch)
    ddp.append(rewards['ddp'])
    cBola.append(rewards['cBola'])
    naive1.append(rewards['naive1'])
    # naive_half.append(rewards['naive_h'])
    naive_full.append(rewards['naive_f'])
    vaware.append(rewards['vaware'])

    if __print__:
        print("\n<------\tComparing Value Rewards\t------>")
        print("C-Bola Average value: {0}".format(np.average(cBola)))
        print("DDP Average value: {0}".format(np.average(ddp)))
        print("Naive1 Average value: {0}".format(np.average(naive1)))
        print("VAware Average value: {0}".format(np.average(vaware)))
        print("Naive-full Average value: {0}".format(np.average(naive_full)))

    return ddp, cBola, naive1, naive_full, vaware


def compare_rewards_val(number_of_samples=1, __print__=True):
    # bola = []
    cBola = []
    naive1 = []
    vaware = []
    naive_full = []

    for sample in range(number_of_samples):
        path = "results/sample_{0}".format(sample)
        rewards = get_attr(path, 'reward_val')
        # bola.append(rewards['bola'])
        cBola.append(rewards['cBola'])
        naive1.append(rewards['naive1'])
        vaware.append(rewards['vaware'])
        naive_full.append(rewards['naive_f'])

    if __print__:
        print("\n<------\tComparing Value Rewards\t------>")
        print("C-Bola Average value: {0}".format(np.average(cBola)))
        # print("Bola3d Average value: {0}".format(np.average(bola)))
        print("Naive1 Average value: {0}".format(np.average(naive1)))
        print("Naive-full Average value: {0}".format(np.average(naive_full)))
        print("VAware Average value: {0}".format(np.average(vaware)))

    return cBola, naive1, naive_full, vaware


def compare_rebuff(number_of_samples=4, __print__=True):
    # bola = []
    cBola = []
    ddp = []
    naive1 = []

    naive_full = []
    vaware = []

    for sample in range(number_of_samples):
        path = "results/sample_{0}".format(sample)
        attr = get_attr(path, 'rebuff')
        # bola.append(attr['bola'])
        cBola.append(attr['cBola'])
        ddp.append(attr['ddp'])
        naive1.append(attr['naive1'])
        vaware.append(attr['vaware'])
        naive_full.append(attr['naive_f'])

    if __print__:
        print("\n<------\tComparing Re-buffering\t------>")
        print("C-Bola Average rebuff: {0}".format(np.average(cBola)))
        # print("Bola3d Average rebuff: {0}".format(np.average(bola)))
        print("DP-Online Average rebuff: {0}".format(np.average(ddp)))
        print("Naive1 Average rebuff: {0}".format(np.average(naive1)))
        print("VAware Average rebuff: {0}".format(np.average(vaware)))
        print("Naive-full Average rebuff: {0}".format(np.average(naive_full)))

    return cBola, ddp, naive1, naive_full, vaware


def get_available_times(requests):
    # Updated
    available_time = {}
    downloaded_bitrates = {}
    replacements = []
    for i in range(len(requests)):
        request = requests[i]
        if request.type == "new":
            available_time[request.segment] = request.end_time
            downloaded_bitrates[int(request.segment)] = request.bitrates
        elif request.type == "replacement":
            replacements.append(request)
        elif request.type == "wait":
            continue

    return available_time, downloaded_bitrates, replacements


def get_Tend(requests, buffer, delta, __start_time__, dt=0.05):
    # Updated
    available_time, downloaded_bitrates, replacements = get_available_times(requests)

    playing_segment = -1
    portion_remained = 0
    t = __start_time__
    buffer_level = 0
    play_rate = 0
    downloaded_segment = -1

    ending_time = requests[-1].end_time

    processing_replacement_index = 0

    while t < ending_time or buffer_level > 0:

        # apply the replacement if is available

        if processing_replacement_index < len(replacements):
            replacement = replacements[processing_replacement_index]
            if t >= replacement.end_time:
                downloaded_bitrates[replacement.segment], new_tiles = Request.apply_replecement(replacement,
                                                                                            downloaded_bitrates[
                                                                                                replacement.segment])
                processing_replacement_index += 1

        # add downloaded segment to buffer
        if downloaded_segment < len(available_time.keys()) - 1 and t >= available_time[downloaded_segment + 1]:
            buffer_level = np.maximum(buffer.get_buffer_value(t), 0)
            downloaded_segment += 1

        # go to next segment if available
        if portion_remained <= 0 and playing_segment < len(available_time.keys()) - 1 and t >= available_time[
            playing_segment + 1]:
            playing_segment += 1
            portion_remained = delta
            play_rate = len(downloaded_bitrates[playing_segment])

        buffer_level -= play_rate * dt
        buffer_level = np.maximum(buffer_level, 0)
        portion_remained -= dt

        t += dt

    return t


def get_playing_bitrates(requests, buffer, actual_headmovement, bitrates, delta, values, dt=0.05,
                         __start_time__=0):
    available_time, downloaded_bitrates, replacements = get_available_times(requests)
    available_time[-1] = -float('inf')

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

    processing_replacement_index = 0
    ending_time = requests[-1].end_time
    while playing_segment < len(available_time) - 2 or portion_remained > 0:

        # apply the replacement if is available

        if processing_replacement_index < len(replacements):
            replacement = replacements[processing_replacement_index]
            if t >= replacement.end_time:
                downloaded_bitrates[replacement.segment], new_tiles = Request.apply_replecement(replacement,
                                                                                            downloaded_bitrates[
                                                                                                replacement.segment])
                processing_replacement_index += 1

        # add downloaded segment to buffer
        if downloaded_segment < len(available_time.keys()) - 2 and t >= available_time[downloaded_segment + 1]:
            buffer_level = np.maximum(buffer.get_buffer_value(t), 0)
            downloaded_segment += 1

        # go to next segment if available
        if portion_remained <= 0 and playing_segment < len(available_time) - 2 and t >= available_time[
            playing_segment + 1]:
            playing_segment += 1
            portion_remained = delta
            play_rate = len(downloaded_bitrates[playing_segment])

        watching_tile = actual_headmovement[playing_segment]

        # # if segment has not downloaded yet
        # if t < available_time[playing_segment]:
        #     rebuff += dt
        #     watching_bitrate.append(0)
        #     time_slots.append(t)
        #     getting_values.append(0)

        # if the watching tile is not in list of downloaded segments
        if portion_remained > 0 and t >= available_time[playing_segment] and downloaded_bitrates[playing_segment][
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
            play_rate = 0

        buffer_level -= play_rate * dt
        buffer_level = np.maximum(buffer_level, 0)
        portion_remained -= dt

        t += dt

    avg_wbr = get_average_watching_bitrate(watching_bitrate)
    avg_wv = np.average(getting_values)
    return rebuff, watching_bitrate, time_slots, avg_wbr, avg_wv


def real_evaluation(path_to_save, sample, delta, values, DDP=False, N1=True, VA=True, Nf=True,
                    __start_time__=0,
                    __print__=True):
    # TODO
    fig = plt.figure(int(np.random.random() * 10000))

    path = "results/sample_{0}".format(sample)

    meta_data = read_dataset(path + "/meta.json")
    actual_headmovement = meta_data['view']
    bitrates = np.array(meta_data['sizes']) / delta

    buffers = get_attr(path, 'buffer')

    # solution = get_attr(path, 'solution')
    solution = get_attr(path, 'requests')

    # bola_buff = buffers['bola']
    cBola_buff = buffers['cBola']
    ddp_buff = buffers['ddp']
    naive1_buff = buffers['naive1']
    naive_full_buff = buffers['naive_f']
    vaware_buff = buffers['vaware']


    # bola_solution = solution['bola']
    cBola_solution = solution['cBola']
    ddp_solution = solution['ddp']
    naive1_solution = solution['naive1']
    naivf_solution = solution['naive_f']
    vaware_solution = solution['vaware']



    # bola_rebuff, bola_y, bola_x, bola_avg_btr, bola_avg_wv = get_playing_bitrates(bola_solution, bola_buff, bola_time,
    #                                                                               actual_headmovement, bitrates, delta,
    #                                                                               values,
    #                                                                               __start_time__=__start_time__)
    cbola_rebuff, cbola_y, cbola_x, cbola_avg_btr, cbola_avg_wv = get_playing_bitrates(cBola_solution, cBola_buff,
                                                                                       actual_headmovement, bitrates,
                                                                                       delta,
                                                                                       values,
                                                                                       __start_time__=__start_time__)
    if DDP:
        ddp_rebuff, ddp_y, ddp_x, ddp_avg_btr, ddp_avg_wv = get_playing_bitrates(ddp_solution, ddp_buff,
                                                                                 actual_headmovement,
                                                                                 bitrates, delta, values,
                                                                                 __start_time__=__start_time__)
    naive1_rebuff, naive1_y, naive1_x, naive1_avg_btr, naive1_avg_wv = get_playing_bitrates(naive1_solution,
                                                                                            naive1_buff,
                                                                                            actual_headmovement,
                                                                                            bitrates, delta, values,
                                                                                            __start_time__=__start_time__)
    vaware_rebuff, vaware_y, vaware_x, vaware_avg_btr, vaware_avg_wv = get_playing_bitrates(vaware_solution,
                                                                                            vaware_buff,
                                                                                            actual_headmovement,
                                                                                            bitrates,
                                                                                            delta, values,
                                                                                            __start_time__=__start_time__)
    naivf_rebuff, naivf_y, naivf_x, naivf_avg_btr, naivf_avg_wv = get_playing_bitrates(naivf_solution, naive_full_buff,
                                                                                       actual_headmovement, bitrates,
                                                                                       delta, values,
                                                                                       __start_time__=__start_time__)

    plt.plot(cbola_x, cbola_y, label="ALGNAME", linewidth=1.5)
    # if Bola:
    #     plt.plot(bola_x, bola_y, label="Bola360", linewidth=1.5)
    if DDP:
        plt.plot(ddp_x, ddp_y, label="DP-Online", linewidth=1.5)
    if N1:
        plt.plot(naive1_x, naive1_y, label="Naive-1", linewidth=1.5)
    if Nf:
        plt.plot(naivf_x, naivf_y, label="Naive-full", linewidth=1.5)
    if VA:
        plt.plot(vaware_x, vaware_y, label="Naive-half", linewidth=1.5)

    for br in bitrates:
        plt.plot([np.min([cbola_x[0], vaware_x[0], naivf_x[0], naive1_x[0], naivf_x[0]]),
                  np.max([cbola_x[-1], naivf_x[-1], vaware_x[0], naive1_x[-1], naivf_x[-1]])], [br, br],
                 label='_nolegend_',
                 linestyle='dashed', color="gray",
                 linewidth=0.7)

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Playing bitrate (Mbps)")
    plt.savefig(path_to_save, dpi=600, bbox_inches='tight')

    actual_bitrates = {}
    # actual_bitrates['bola'] = bola_avg_btr
    actual_bitrates['cbola'] = cbola_avg_btr
    actual_bitrates['naive-1'] = naive1_avg_btr
    actual_bitrates['naive-f'] = naivf_avg_btr
    actual_bitrates['vaware'] = vaware_avg_btr

    rebuffering = {}
    # rebuffering['bola'] = bola_rebuff
    rebuffering['cbola'] = cbola_rebuff
    rebuffering['naive-1'] = naive1_rebuff
    rebuffering['naive-f'] = naivf_rebuff
    rebuffering['vaware'] = vaware_rebuff

    viewed_values = {}
    # viewed_values['bola'] = bola_avg_wv
    viewed_values['cbola'] = cbola_avg_wv
    viewed_values['naive-1'] = naive1_avg_wv
    viewed_values['naive-f'] = naivf_avg_wv
    viewed_values['vaware'] = vaware_avg_wv

    if __print__:
        print("\n<------\tComparing Actual bitrates\t------>")
        print("C-Bola Average watching bitrate: {0:.2f}, rebuffering: {1:.2f}, avg v: {2:.2f}".format(cbola_avg_btr,
                                                                                                      cbola_rebuff,
                                                                                                      cbola_avg_wv))

        # print("Bola3d Average watching bitrate: {0:.2f}, rebuffering: {1:.2f}, avg v: {2:.2f}".format(bola_avg_btr,
        #                                                                                               bola_rebuff,
        #                                                                                               bola_avg_wv))
        if DDP:
            print(
                "DP-Online Average watching bitrate: {0:.2f}, rebuffering: {1:.2f}, avg v: {2:.2f}".format(ddp_avg_btr,
                                                                                                           ddp_rebuff,
                                                                                                           ddp_avg_wv))
        print("Naive1 Average watching bitrate: {0:.2f}, rebuffering: {1:.2f}, avg v: {2:.2f}".format(naive1_avg_btr,
                                                                                                      naive1_rebuff,
                                                                                                      naive1_avg_wv))
        print("VAWAre Average watching bitrate: {0:.2f}, rebuffering: {1:.2f}, avg v: {2:.2f}".format(vaware_avg_btr,
                                                                                                      vaware_rebuff,
                                                                                                      vaware_avg_wv))
        print("Naive-full Average watching bitrate: {0:.2f}, rebuffering: {1:.2f}, avg v: {2:.2f}".format(naivf_avg_btr,
                                                                                                          naivf_rebuff,
                                                                                                          naivf_avg_wv))
        print("\n<------\t ----------------- \t------>")
    return actual_bitrates, rebuffering, viewed_values


def batch_real_evaluation(batch, delta, values, __start_time__=0, __print__=False):
    # TODO
    fig = plt.figure(int(np.random.random() * 10000))

    path = "results/BatchRun/"

    meta_data = read_dataset(path + "/meta_{0}.json".format(batch))
    actual_headmovement = meta_data['view']
    bitrates = np.array(meta_data['sizes']) / delta

    buffers = batch_get_attr(path, 'buffer', batch)

    solution = batch_get_attr(path, 'requests', batch)

    ddp_buff = buffers['ddp']
    cBola_buff = buffers['cBola']
    naive1_buff = buffers['naive1']
    vaware_buff = buffers['vaware']
    naive_full_buff = buffers['naive_f']


    ddp_solution = solution['ddp']
    cBola_solution = solution['cBola']
    naive1_solution = solution['naive1']
    vaware_solution = solution['vaware']
    naivf_solution = solution['naive_f']


    ddp_rebuff, ddp_y, ddp_x, ddp_avg_btr, ddp_avg_wv = get_playing_bitrates(ddp_solution, ddp_buff,
                                                                             actual_headmovement, bitrates, delta,
                                                                             values,
                                                                             __start_time__=__start_time__)
    cbola_rebuff, cbola_y, cbola_x, cbola_avg_btr, cbola_avg_wv = get_playing_bitrates(cBola_solution, cBola_buff,
                                                                                       actual_headmovement, bitrates,
                                                                                       delta,
                                                                                       values,
                                                                                       __start_time__=__start_time__)
    naive1_rebuff, naive1_y, naive1_x, naive1_avg_btr, naive1_avg_wv = get_playing_bitrates(naive1_solution,
                                                                                            naive1_buff,
                                                                                            actual_headmovement,
                                                                                            bitrates, delta, values,
                                                                                            __start_time__=__start_time__)
    vaware_rebuff, vaware_y, vaware_x, vaware_avg_btr, vaware_avg_wv = get_playing_bitrates(vaware_solution,
                                                                                            vaware_buff,
                                                                                            actual_headmovement,
                                                                                            bitrates,
                                                                                            delta, values,
                                                                                            __start_time__=__start_time__)
    naivf_rebuff, naivf_y, naivf_x, naivf_avg_btr, naivf_avg_wv = get_playing_bitrates(naivf_solution, naive_full_buff,
                                                                                       actual_headmovement, bitrates,
                                                                                       delta, values,
                                                                                       __start_time__=__start_time__)

    actual_bitrates = {}
    actual_bitrates['ddp'] = ddp_avg_btr
    actual_bitrates['cbola'] = cbola_avg_btr
    actual_bitrates['naive-1'] = naive1_avg_btr
    actual_bitrates['naive-f'] = naivf_avg_btr
    actual_bitrates['vaware'] = vaware_avg_btr

    rebuffering = {}
    rebuffering['ddp'] = ddp_rebuff
    rebuffering['cbola'] = cbola_rebuff
    rebuffering['naive-1'] = naive1_rebuff
    rebuffering['naive-f'] = naivf_rebuff
    rebuffering['vaware'] = vaware_rebuff

    viewed_values = {}
    viewed_values['ddp'] = ddp_avg_wv
    viewed_values['cbola'] = cbola_avg_wv
    viewed_values['naive-1'] = naive1_avg_wv
    viewed_values['naive-f'] = naivf_avg_wv
    viewed_values['vaware'] = vaware_avg_wv

    if __print__:
        print("\n<------\tComparing Actual bitrates\t------>")
        print("C-Bola Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(cbola_avg_btr,
                                                                                                      cbola_rebuff,
                                                                                                      cbola_avg_wv))

        print("DDP Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(ddp_avg_btr,
                                                                                                   ddp_rebuff,
                                                                                                   ddp_avg_wv))
        print("Naive1 Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(naive1_avg_btr,
                                                                                                      naive1_rebuff,
                                                                                                      naive1_avg_wv))
        print("Naive-full Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(naivf_avg_btr,
                                                                                                          naivf_rebuff,
                                                                                                          naivf_avg_wv))
        print("VAware Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(vaware_avg_btr,
                                                                                                      vaware_rebuff,
                                                                                                      vaware_avg_wv))
        print("\n<------\t ----------------- \t------>")
    return actual_bitrates, rebuffering, viewed_values


def get_average_watching_bitrate(watching_bitrate):
    wbr = [0 if x <= 0 else x for x in watching_bitrate]
    return np.average(wbr)


def make_buffer_lines(buffer, requests, delta, dt=0.05, __start_time__=0):
    available_time, downloaded_bitrates, replacements = get_available_times(requests)


    x = []
    y = []

    playing_segment = -1
    portion_remained = 0
    t = __start_time__
    buffer_level = 0
    play_rate = 0
    downloaded_segment = -1

    ending_time = requests[-1].end_time
    while t < ending_time or buffer_level > 0:
        x.append(t)
        y.append(buffer_level)
        t += dt
        # play next segment if available
        if portion_remained <= 0 and playing_segment < len(available_time) - 1 and t >= available_time[
            playing_segment + 1]:
            playing_segment += 1
            portion_remained = delta
            play_rate = len(downloaded_bitrates[playing_segment])
        buffer_level -= play_rate * dt
        buffer_level = np.maximum(buffer_level, 0)
        portion_remained -= dt

        # add downloaded segment to buffer
        if downloaded_segment < len(available_time) - 1 and t >= available_time[downloaded_segment + 1]:
            buffer_level = np.maximum(buffer.get_buffer_value(t), 0)
            downloaded_segment += 1
    return y, x


def compare_buffers(path_to_save, sample, delta, DDP=False, N1=True, VA=True, Nf=True, __start_time__=0):
    # TODO
    fig = plt.figure(int(np.random.random() * 10000))
    path = "results/sample_{0}".format(sample)
    buffers = get_attr(path, 'buffer')
    times = get_attr(path, 'time')
    solution = get_attr(path, 'requests')

    # bola = buffers['bola']
    cBola = buffers['cBola']
    ddp = buffers['ddp']
    naive1 = buffers['naive1']
    naive_full = buffers['naive_f']
    vaware = buffers['vaware']

    # bola_time = times['bola']
    cBola_time = times['cBola']
    ddp_time = times['ddp']
    naive1_time = times['naive1']
    naive_full_time = times['naive_f']
    vaware_time = times['vaware']

    # bola_solution = solution['bola']
    cBola_solution = solution['cBola']
    ddp_solution = solution['ddp']
    naive1_solution = solution['naive1']
    naivf_solution = solution['naive_f']
    vaware_solution = solution['vaware']

    cbola_y, cbola_x = make_buffer_lines(cBola, cBola_solution, cBola_time, delta, __start_time__=__start_time__)
    # bola_y, bola_x = make_buffer_lines(bola, bola_solution, bola_time, delta, __start_time__=__start_time__)
    ddp_y, ddp_x = make_buffer_lines(ddp, ddp_solution, ddp_time, delta, __start_time__=__start_time__)
    naive1_y, naive1_x = make_buffer_lines(naive1, naive1_solution, naive1_time, delta, __start_time__=__start_time__)
    vaware_y, vaware_x = make_buffer_lines(vaware, vaware_solution, vaware_time, delta,
                                           __start_time__=__start_time__)
    naivf_y, naivf_x = make_buffer_lines(naive_full, naivf_solution, naive_full_time, delta,
                                         __start_time__=__start_time__)

    plt.plot(cbola_x, cbola_y, label="MyModel", linewidth=1.5)
    # if Bola:
    #     plt.plot(vaware_x, vaware_y, label="Bola360", linewidth=1.5)
    if DDP:
        plt.plot(ddp_x, ddp_y, label="DP-Online", linewidth=1.5)
    if N1:
        plt.plot(naive1_x, naive1_y, label="Naive-1", linewidth=1.5)
    if VA:
        plt.plot(vaware_x, vaware_y, label="VAware", linewidth=1.5)
    if Nf:
        plt.plot(naivf_x, naivf_y, label="Naive-full", linewidth=1.5)

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Buffer Levels (s)")
    plt.savefig(path_to_save, dpi=600, bbox_inches='tight')

    return cBola, ddp, naive1, naive_full, vaware

# compare_rewards()
