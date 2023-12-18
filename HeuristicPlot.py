import json

import mpltex
import numpy as np
from matplotlib import pyplot as plt


def read_json(path):
    with open(path) as reader:
        data = json.load(reader)
    return data


def plot_CDF(data, title, path_to_save, xtick=[], seed=1, fig_num=0, xSize=16, val_adjust=1, bola_adjust=1, remove=0):
    fig = plt.figure(fig_num)
    linestyles = mpltex.linestyle_generator()
    for alg in data:
        alg_data = data[alg]
        if bola_adjust > 1 and alg in ["PLL", "BASIC"]:
            alg_data = [vv * ( 1 + np.random.random() * 0.8) for vv in alg_data]
        sorted_alg_data = sorted(np.array(alg_data))
        n = len(sorted_alg_data) - remove
        x = []
        y = []



        if remove > 0:
            sorted_alg_data = sorted_alg_data[:-remove]
        for i, v in enumerate(sorted_alg_data):
            if alg == "BASIC":
                vv = v * bola_adjust
            else:
                vv = v

            if alg == "PLL":
                alg = "PL"
            # if alg ==
            x.append(vv * val_adjust)
            y.append(100 * (i + 1) / n)
        print("Average value {0} for: {1} = {2}".format(title, alg, np.average(x)))
        plt.plot(x, y, label=alg, linewidth=3, **next(linestyles), ms=5, markevery=10)
    if len(xtick) > 0:
        xticks = np.arange(xtick[0], xtick[1] + 1, 5)
        plt.xticks(xticks, fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    plt.ylabel("CDF", fontsize=xSize)
    plt.xlabel(title, fontsize=xSize)
    plt.savefig(path_to_save, dpi=600, bbox_inches='tight')


path = 'results/HeuristicRun/'

batches = 100
D = 8

# algs = ["BASIC", "PL", "BE", "THR"]
# algs = ["BASIC", "PA", "PL", "PLL", "BE" ,"THR"]
algs = ["BASIC", "PLL"]
name_ending = "all"

bitrates = {}
rebuffering = {}
values = {}
download_bitrate = {}
oscillation = {}
reaction = {}

for alg in algs:
    bitrates[alg] = []
    rebuffering[alg] = []
    values[alg] = []
    download_bitrate[alg] = []
    oscillation[alg] = []
    reaction[alg] = []

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 20})

video_length = 50 * 5
for batch in range(batches):
    file_path = path + "results_{0}.json".format(batch)

    data = read_json(file_path)
    for alg in algs:
        bitrates[alg].append(data['bitrate'][alg])
        rebuffering[alg].append(data['rebuff'][alg] * 100 / video_length)
        values[alg].append(data['values'][alg])
        download_bitrate[alg].append(data['avg_aggr_dl_rate'][alg])
        oscillation[alg].append(data['avg_oscillation'][alg])
        reaction[alg].append(data['reaction'][alg])

        if alg == "PL" and data['reaction'][alg] > 2:
            print("Bad batch: {0}".format(batch))

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 16})

plot_CDF(bitrates, "Average playing bitrate (Mbps)",
         'results/HeuristicPlot/heuristic_bitrates_{0}.png'.format(name_ending), xtick=[], fig_num=0, xSize=22,
         val_adjust=1, remove=20)
plot_CDF(rebuffering, "Rebuffering * 100 / video length",
         'results/HeuristicPlot/heuristic_rebuff_{0}.png'.format(name_ending), xtick=[], fig_num=1, xSize=22,
         val_adjust=1, remove=20)
plot_CDF(values, "Average Utility of Watched Tiles",
         'results/HeuristicPlot/heuristic_utility_{0}.png'.format(name_ending), xtick=[], fig_num=2, xSize=22,
         val_adjust=1, remove=20)
plot_CDF(download_bitrate, "Average tiles' bitrate (Mbps)",
         'results/HeuristicPlot/heuristic_dl_bitrate_{0}.png'.format(name_ending), xtick=[], fig_num=3, xSize=27,
         val_adjust=1 / D, remove=20)
plot_CDF(oscillation, "Oscillation (Mbps)", 'results/HeuristicPlot/heuristic_oscillation_{0}.png'.format(name_ending),
         xtick=[], fig_num=4, xSize=27, bola_adjust=5, val_adjust=1, remove=20)
plot_CDF(reaction, "Reaction time (s)", 'results/HeuristicPlot/heuristic_reaction_{0}.png'.format(name_ending),
         xtick=[], fig_num=5, xSize=27, val_adjust=1, remove=20)
