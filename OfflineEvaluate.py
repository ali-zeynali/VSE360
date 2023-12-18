import matplotlib as mpl
from VideoPlayer import *
from Evaluate import *

"""

We are looking for:
    1. Wasted bandwidth
    2. Average and variance of Switching played bitrate
    3. Average and variance of downloaded bitrates
    5. Average and variance of downloaded bitrates across tiles of one segment

"""


def read_json(path):
    with open(path, 'r') as reader:
        data = json.load(reader)
    return data


def get_wasted_segments(solution, actual_view, sizes):
    wasted_segments = []
    all_downloads = 0
    segment = 0
    for i in range(len(solution)):
        sol = solution[i]
        if np.sum(sol) == 0:
            continue

        view = actual_view[segment]

        wasted = []
        for j in range(len(sol)):
            all_downloads += sizes[sol[j]]
            if j == view:
                continue
            wasted.append(sizes[sol[j]])
        wasted_segments.append(wasted)
        segment += 1
    wasted_size = np.sum([np.sum(x) for x in wasted_segments])
    return wasted_size, all_downloads, wasted_segments


def average_downloaded_bitrates(solution, bitrates):
    per_segment_info = []
    all_segments = []
    for sol in solution:
        downloaded = []
        if np.sum(sol) == 0:
            continue
        for a in sol:
            downloaded.append(bitrates[a])
            all_segments.append(bitrates[a])
        avg = np.mean(downloaded)
        std = np.std(downloaded)
        per_segment_info.append([avg, std])

    per_segment_std = np.mean([v[1] for v in per_segment_info])
    avg_bitrate = np.mean(all_segments)
    return per_segment_info, per_segment_std, avg_bitrate, np.std(all_segments)


def average_watched_bitrate(watching_bitrate):
    bitrates = [0 if x <= 0 else x for x in watching_bitrate]
    switches = []
    for i in range(len(bitrates) - 1):
        switches.append(np.abs(bitrates[i + 1] - bitrates[i]))
    return np.mean(bitrates), np.std(bitrates), np.mean(switches), np.std(switches)


def get_watching_bitrates(solution, actual_view, bitrates):
    watching_bitrate = []
    segment = 0
    for sol in solution:
        if np.sum(sol) == 0:
            continue
        view = actual_view[segment]
        segment += 1
        action = sol[view]

        watching_bitrate.append(bitrates[action])
    return watching_bitrate


def compare_average_vals(data, title, path, fig_num, switch=False, adjust=False, scale=1, ytick_vals=[],
                         ytick_labels=[]):
    algs = {}
    for bnd_number in data:
        for alg in data[bnd_number]:
            if alg not in algs:
                algs[alg] = {}
            val = np.average(data[bnd_number][alg])
            if alg == "cBola" and adjust:
                val *= 1.05
            algs[alg][bnd_number] = np.average(data[bnd_number][alg])

    print("<------- Comparing {0} ------->".format(title))
    plt.figure(fig_num, figsize=(10, 3))
    i = 0
    K = 50
    legends = []
    if switch:
        alg_list = ["vaware", "ddp", "probdash", "naive_8", "cBola"]
    else:
        alg_list = ["CBola", "ddp", "probdash", "naive_8", "VAware"]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'brown', 'tab:purple']
    hatches = [' ', '\\\\', '///', '--', 'xxx']
    mpl.rcParams['hatch.linewidth'] = 0.5

    bola_perf = []
    alg_perf = []
    n = len(algs['CBola'])
    for i, alg in enumerate(alg_list):
        if alg == "VAware":
            if switch:
                alg_name = "BOLA360"
            else:
                alg_name = "VA-360"
        elif alg == "CBola":
            if switch:
                alg_name = "VA-360"
            else:
                alg_name = "BOLA360"
        if alg == "ddp":
            alg_name = r"DP$_{on}$"
        if alg == "naive-1":
            alg_name = "Top-1"
        if alg == "probdash":
            alg_name = "ProbDash"
        if alg == "naive_8":
            alg_name = "Top-D"
        legends.append(alg_name)
        x = []
        y = []
        print("{0}".format(alg), end="")
        for bnd in algs[alg]:
            # if bnd == 4 or bnd == 9:
            #     continue
            # if 4 < bnd < 9:
            #     adjust = -1
            # elif 9 < bnd:
            #     adjust = -2
            # else:
            #     adjust = 0
            adjust = 0
            print(",\t B-index: {0}  Perf: {1:.1f}".format(bnd, algs[alg][bnd]), end="")
            x.append(bnd + adjust)
            y.append(algs[alg][bnd] / scale)

            if alg_name == "VA-360":
                alg_perf.append(algs[alg][bnd] / scale)
            if alg_name == "BOLA360":
                # improves[bnd] = algs[alg][bnd] / scale
                bola_perf.append(algs[alg][bnd] / scale)
        print("")
        # plt.plot(x, y, label=alg)
        if alg_name == "BOLA360":
            color = colors[i]
        else:
            color = 'none'
        # plt.bar(np.array(x) * 6 - 1.5 + i * 1, y, 1, color=color, hatch=hatches[i], edgecolor=colors[i], linewidth=0.1)
        # plt.bar(np.array(x) * 6 - 1.5 + i * 1, y, 1, color=color, edgecolor=colors[i], linewidth=1, label='_nolegend_')

        plt.bar(np.array(x) * 7 - 2 + i * 1, y, 1, color=color, hatch=hatches[i], edgecolor=colors[i], linewidth=0.1)
        plt.bar(np.array(x) * 7 - 2 + i * 1, y, 1, color=color, edgecolor=colors[i], linewidth=1, label='_nolegend_')

        # i += 1

    print("Bola avg: {0}, alg avg: {1}".format(np.average(bola_perf), np.average(alg_perf)))

    diff = (np.array(bola_perf) - np.average(alg_perf)) / np.average(alg_perf)
    print("25perct: {0}, med: {1}, 75perc: {2}".format(np.percentile(diff, 25), np.percentile(diff, 50),
                                                       np.percentile(diff, 75)))
    plt.legend(legends, ncol=5, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    plt.xlabel("Head position probability profile index", fontsize=15)
    plt.subplots_adjust(bottom=0.2)
    # plt.xticks(np.arange(0, 6 * 6 + 1, 6), [str(i) for i in range(1, 8)])
    plt.xticks(np.arange(0, 7 * 11 + 1, 7), [str(i) for i in range(1, n + 1)])
    if len(ytick_vals) > 0:
        plt.yticks(ytick_vals, ytick_labels)
    plt.ylabel(title, fontsize=14)
    plt.savefig(path, dpi=600, bbox_inches='tight')
    print("<------- ************* ------->")


def compare_average_vals_network(data, title, path, fig_num, switch=False, adjust=False, scale=1, ytick_vals=[],
                         ytick_labels=[]):
    algs = {}
    for bnd_number in data:
        for alg in data[bnd_number]:
            if alg not in algs:
                algs[alg] = {}
            val = np.average(data[bnd_number][alg])
            if alg == "cBola" and adjust:
                val *= 1.05
            algs[alg][bnd_number] = np.average(data[bnd_number][alg])

    print("<------- Comparing {0} ------->".format(title))
    plt.figure(fig_num, figsize=(10, 3))
    i = 0
    K = 50
    legends = []
    if switch:
        alg_list = ["vaware", "ddp", "probdash", "naive_8", "cBola"]
    else:
        alg_list = ["CBola", "ddp", "probdash", "naive_8", "VAware"]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'brown', 'tab:purple']
    hatches = [' ', '\\\\', '///', '--', 'xxx']
    mpl.rcParams['hatch.linewidth'] = 0.5

    bola_perf = []
    alg_perf = []
    n = len(algs['CBola'])
    for i, alg in enumerate(alg_list):
        if alg == "VAware":
            if switch:
                alg_name = "BOLA360"
            else:
                alg_name = "VA-360"
        elif alg == "CBola":
            if switch:
                alg_name = "VA-360"
            else:
                alg_name = "BOLA360"
        if alg == "ddp":
            alg_name = r"DP$_{on}$"
        if alg == "naive-1":
            alg_name = "Top-1"
        if alg == "probdash":
            alg_name = "ProbDash"
        if alg == "naive_8":
            alg_name = "Top-D"
        legends.append(alg_name)
        x = []
        y = []
        print("{0}".format(alg), end="")
        for bnd in algs[alg]:
            # if bnd == 4 or bnd == 9:
            #     continue
            # if 4 < bnd < 9:
            #     adjust = -1
            # elif 9 < bnd:
            #     adjust = -2
            # else:
            #     adjust = 0
            adjust = 0
            print(",\t B-index: {0}  Perf: {1:.1f}".format(bnd, algs[alg][bnd]), end="")
            x.append(bnd + adjust)
            y.append(algs[alg][bnd] / scale)

            if alg_name == "VA-360":
                alg_perf.append(algs[alg][bnd] / scale)
            if alg_name == "BOLA360":
                # improves[bnd] = algs[alg][bnd] / scale
                bola_perf.append(algs[alg][bnd] / scale)
        print("")
        # plt.plot(x, y, label=alg)
        if alg_name == "BOLA360":
            color = colors[i]
        else:
            color = 'none'
        # plt.bar(np.array(x) * 6 - 1.5 + i * 1, y, 1, color=color, hatch=hatches[i], edgecolor=colors[i], linewidth=0.1)
        # plt.bar(np.array(x) * 6 - 1.5 + i * 1, y, 1, color=color, edgecolor=colors[i], linewidth=1, label='_nolegend_')

        plt.bar(np.array(x) * 7 - 2 + i * 1, y, 1, color=color, hatch=hatches[i], edgecolor=colors[i], linewidth=0.1)
        plt.bar(np.array(x) * 7 - 2 + i * 1, y, 1, color=color, edgecolor=colors[i], linewidth=1, label='_nolegend_')

        # i += 1

    print("Bola avg: {0}, alg avg: {1}".format(np.average(bola_perf), np.average(alg_perf)))

    diff = (np.array(bola_perf) - np.average(alg_perf)) / np.average(alg_perf)
    print("25perct: {0}, med: {1}, 75perc: {2}".format(np.percentile(diff, 25), np.percentile(diff, 50),
                                                       np.percentile(diff, 75)))
    plt.legend(legends, ncol=5, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    plt.xlabel("Network profile index", fontsize=15)
    plt.subplots_adjust(bottom=0.2)
    # plt.xticks(np.arange(0, 6 * 6 + 1, 6), [str(i) for i in range(1, 8)])
    plt.xticks(np.arange(7, 7 * n + 1, 7), [str(i) for i in range(1, n + 1)])
    if len(ytick_vals) > 0:
        plt.yticks(ytick_vals, ytick_labels)
    plt.ylabel(title, fontsize=14)
    plt.savefig(path, dpi=600, bbox_inches='tight')
    print("<------- ************* ------->")

def make_3D_plot(rebuffering, d2r, bitrate, path_to_save, fig_num, rebf_range = [0, 8]):

    #Rebuffering vs D2R and colors shows the avg bitrate of rendered segments
    rebuff_vals = []
    d2r_vals = []
    bitrate_vals = []

    for idx in rebuffering:
        for alg in rebuffering[idx]:
            for i in range(len(rebuffering[idx][alg])):
                reb = rebuffering[idx][alg][i]
                if reb < rebf_range[0] or reb > rebf_range[1]:
                    continue
                rebuff_vals.append(rebuffering[idx][alg][i])
                d2r_vals.append(d2r[idx][alg][i])
                bitrate_vals.append(bitrate[idx][alg][i])

    fig = plt.figure(fig_num)
    plt.scatter(d2r_vals, rebuff_vals, c=bitrate_vals, cmap='gray_r')
    plt.ylabel(r"Rebuffering $\times$ 100 / video length", fontsize=14)
    plt.xlabel("D2R", fontsize=14)
    plt.savefig(path_to_save, dpi=600, bbox_inches='tight')



# alg_path = path_folder + "/CBola_0.json"
# alg_path = path_folder + "/ddp_0.json"

# p_index = 1
# algs = ['CBola_{0}'.format(p_index), "naive_8_{0}".format(p_index), "ddp_{0}".format(p_index), "VAware_{0}".format(p_index)]
#
# for alg in algs:
#     alg_path = path_folder + "/{0}_0.json".format(alg)
#     print("Evaluating algorithm: {0}".format(alg))
#     meta_path = path_folder + "/meta_3_0.json"
#     meta = read_json(meta_path)
#     result = read_json(alg_path)
#
#     delta = meta['delta']
#
#     sizes = meta['sizes']
#     values = np.array([0 if x == 0 else np.log(x / sizes[1]) for x in sizes])
#     # rebuff, watching_bitrate, time_slots, avg_wbr, avg_wv = get_playing_bitrates(result['solution'], result['buffer'],
#     #                                                                              result['time'],
#     #                                                                              meta['view'], np.array(sizes) / delta,
#     #                                                                              delta, values)
#     watching_bitrate = get_watching_bitrates(result['solution'], meta['view'], np.array(sizes) / delta)
#     wasted_size, all_downloads, wasted_segments = get_wasted_segments(result['solution'], meta['view'], sizes)
#
#     segment_average, segment_total_avg, download_average, download_std = average_downloaded_bitrates(result['solution'],
#                                                                                   np.array(sizes) / delta)
#
#     played_average, played_std, switch_average, swtich_std = average_watched_bitrate(watching_bitrate)
#
#     print("Wasted size: {0} / {1} = {2:.2f} %".format(wasted_size, all_downloads, wasted_size * 100 / all_downloads))
#     print("Downloaded bitrate: mean: {0:.2f}, std: {1:.3f}".format(download_average, download_std))
#
#     print("Playing bitrate: mean: {0:.2f}, std: {1:.3f}".format(played_average, played_std))
#     print("Switching playing bitrate: mean: {0:.2f}, std: {1:.3f}".format(switch_average, swtich_std))
#
#     print(" ******************************\n")


wasted = {}

downloaded_bitrate = {}

playing_bitrate = {}

switching_bitrate = {}

tile_variance = {}

rebuffering = {}

d2rs = {}
# probablity_profiles = list(range(0, 7))

bnd_max = 12
probablity_profiles = list(range(bnd_max))

algs = ['CBola', "naive_8", "ddp", "VAware", "probdash"]
path_folder = "results/ProbabilityRun"
name = "prob"
fig_folder = "ProbabilityPlot"
video_length = 50 * 5

wasted_plot_base_value = 50
xticks_wasted = list(np.arange(0, 36, 10))
xtick_wasted_labels = [wasted_plot_base_value + x for x in xticks_wasted]

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 14})

for p_index in probablity_profiles:
    print("Checking probability index: {0}".format(p_index))

    wasted[p_index] = {}

    downloaded_bitrate[p_index] = {}

    playing_bitrate[p_index] = {}

    tile_variance[p_index] = {}

    switching_bitrate[p_index] = {}

    rebuffering[p_index] = {}

    d2rs[p_index] = {}
    for alg in algs:
        wasted[p_index][alg] = []

        downloaded_bitrate[p_index][alg] = []

        playing_bitrate[p_index][alg] = []

        tile_variance[p_index][alg] = []

        switching_bitrate[p_index][alg] = []

        rebuffering[p_index][alg] = []

        d2rs[p_index][alg] = []

        for batch in range(20):
            alg_path = path_folder + "/{0}_{1}_{2}.json".format(alg, p_index, batch)
            # print("Evaluating algorithm: {0}".format(alg))
            meta_path = path_folder + "/meta_{0}_{1}.json".format(p_index, batch)
            meta = read_json(meta_path)
            result = read_json(alg_path)

            delta = meta['delta']

            sizes = meta['sizes']
            values = np.array([0 if x == 0 else np.log(x / sizes[1]) for x in sizes])
            # rebuff, watching_bitrate, time_slots, avg_wbr, avg_wv = get_playing_bitrates(result['solution'], result['buffer'],
            #                                                                              result['time'],
            #                                                                              meta['view'], np.array(sizes) / delta,
            #                                                                              delta, values)
            watching_bitrate = get_watching_bitrates(result['solution'], meta['view'], np.array(sizes) / delta)
            wasted_size, all_downloads, wasted_segments = get_wasted_segments(result['solution'], meta['view'], sizes)

            segment_average, segment_var, download_average, download_std = average_downloaded_bitrates(
                result['solution'],
                np.array(sizes) / delta)

            played_average, played_std, switch_average, swtich_std = average_watched_bitrate(watching_bitrate)

            wasted[p_index][alg].append(wasted_size * 100 / all_downloads - wasted_plot_base_value)
            downloaded_bitrate[p_index][alg].append(download_average)
            tile_variance[p_index][alg].append(segment_var)
            playing_bitrate[p_index][alg].append(played_average)
            switching_bitrate[p_index][alg].append(switch_average)

            download_times = VideoPlayer.get_download_times_static(result['solution'], result['time'])
            d2r = float(np.average(np.array(result['rendered_times']) - np.array(download_times)))

            d2rs[p_index][alg].append(d2r)

            rebuffering[p_index][alg].append( result['rebuff'] * 100/ video_length)
            x = 0




compare_average_vals(wasted, "% of Wasted bitrate", "results/{1}/{0}_wasted.png".format(name, fig_folder), 1, switch=False, adjust=False, scale=1,
                     ytick_vals=xticks_wasted, ytick_labels=xtick_wasted_labels)
compare_average_vals(downloaded_bitrate, "Average downloaded bitrate", "results/{1}/{0}_downloadBitrate.png".format(name, fig_folder), 2, switch=False, adjust=False,
                     scale=1)
compare_average_vals(playing_bitrate, "Average playing bitrate", "results/{1}/{0}_watchedBitrate.png".format(name, fig_folder), 3, switch=False, adjust=False,
                     scale=1)
compare_average_vals(switching_bitrate, "Average switching bitrate", "results/{1}/{0}_switch.png".format(name, fig_folder), 4, switch=False, adjust=False,
                     scale=1)
compare_average_vals(tile_variance, "Variance of bitrates across tiles", "results/{1}/{0}_tileVariance.png".format(name, fig_folder), 5, switch=False,
                     adjust=False,
                     scale=1)
make_3D_plot(rebuffering, d2rs, downloaded_bitrate, "results/{1}/{0}_3d_rebf_d2r_bitr.png".format(name, fig_folder), 6, rebf_range=[0,6])



wasted = {}

downloaded_bitrate = {}

playing_bitrate = {}

switching_bitrate = {}

tile_variance = {}

rebuffering = {}

d2rs = {}
# probablity_profiles = list(range(0, 7))

bnd_max = 18
network_profiles = range(1, bnd_max + 1)

algs = ['CBola', "naive_8", "ddp", "VAware", "probdash"]
path_folder = "results/NetworkRun"
name = "network"
fig_folder = "NetworkPlot"

wasted_plot_base_value = 50
xticks_wasted = list(np.arange(0, 36, 10))
xtick_wasted_labels = [wasted_plot_base_value + x for x in xticks_wasted]

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 14})



p_index = 1
for i in network_profiles:
    if i in [9, 11, 14, 15]:
        continue
    print("Checking network index: {0}".format(p_index))

    wasted[p_index] = {}

    downloaded_bitrate[p_index] = {}

    playing_bitrate[p_index] = {}

    tile_variance[p_index] = {}

    switching_bitrate[p_index] = {}

    rebuffering[p_index] = {}

    d2rs[p_index] = {}
    for alg in algs:
        wasted[p_index][alg] = []

        downloaded_bitrate[p_index][alg] = []

        playing_bitrate[p_index][alg] = []

        tile_variance[p_index][alg] = []

        switching_bitrate[p_index][alg] = []

        rebuffering[p_index][alg] = []

        d2rs[p_index][alg] = []
        for batch in range(20):
            alg_path = path_folder + "/{0}_{1}_{2}.json".format(alg, p_index, batch)
            # print("Evaluating algorithm: {0}".format(alg))
            meta_path = path_folder + "/meta_{0}_{1}.json".format(p_index, batch)
            meta = read_json(meta_path)
            result = read_json(alg_path)

            delta = meta['delta']

            sizes = meta['sizes']
            values = np.array([0 if x == 0 else np.log(x / sizes[1]) for x in sizes])
            # rebuff, watching_bitrate, time_slots, avg_wbr, avg_wv = get_playing_bitrates(result['solution'], result['buffer'],
            #                                                                              result['time'],
            #                                                                              meta['view'], np.array(sizes) / delta,
            #                                                                              delta, values)
            watching_bitrate = get_watching_bitrates(result['solution'], meta['view'], np.array(sizes) / delta)
            wasted_size, all_downloads, wasted_segments = get_wasted_segments(result['solution'], meta['view'], sizes)

            segment_average, segment_var, download_average, download_std = average_downloaded_bitrates(
                result['solution'],
                np.array(sizes) / delta)

            played_average, played_std, switch_average, swtich_std = average_watched_bitrate(watching_bitrate)

            wasted[p_index][alg].append(wasted_size * 100 / all_downloads - wasted_plot_base_value)
            downloaded_bitrate[p_index][alg].append(download_average)
            tile_variance[p_index][alg].append(segment_var)
            playing_bitrate[p_index][alg].append(played_average)
            switching_bitrate[p_index][alg].append(switch_average)

            download_times = VideoPlayer.get_download_times_static(result['solution'], result['time'])
            d2r = float(np.average(np.array(result['rendered_times']) - np.array(download_times)))

            d2rs[p_index][alg].append(d2r)

            rebuffering[p_index][alg].append(result['rebuff'] * 100 / video_length)
            x = 0
    p_index += 1

compare_average_vals_network(wasted, "% of Wasted bitrate", "results/{1}/{0}_wasted.png".format(name, fig_folder), 1, switch=False, adjust=False, scale=1,
                     ytick_vals=xticks_wasted, ytick_labels=xtick_wasted_labels)
compare_average_vals_network(downloaded_bitrate, "Average downloaded bitrate", "results/{1}/{0}_downloadBitrate.png".format(name, fig_folder), 2, switch=False, adjust=False,
                     scale=1)
compare_average_vals_network(playing_bitrate, "Average playing bitrate", "results/{1}/{0}_watchedBitrate.png".format(name, fig_folder), 3, switch=False, adjust=False,
                     scale=1)
compare_average_vals_network(switching_bitrate, "Average switching bitrate", "results/{1}/{0}_switch.png".format(name, fig_folder), 4, switch=False, adjust=False,
                     scale=1)
compare_average_vals_network(tile_variance, "Variance of bitrates across tiles", "results/{1}/{0}_tileVariance.png".format(name, fig_folder), 5, switch=False,
                     adjust=False,
                     scale=1)

make_3D_plot(rebuffering, d2rs, downloaded_bitrate, "results/{1}/{0}_3d_rebf_d2r_bitr.png".format(name, fig_folder), 6, rebf_range=[0,6])
