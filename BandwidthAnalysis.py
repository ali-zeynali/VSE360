from matplotlib import pyplot as plt

from Bandwidth import *


def plot_bandwidth(bandwidths):
    plt.rcParams["font.family"] = "Times New Roman"
    T = 500

    dt = 0.1
    fig, axes = plt.subplots(5, 3, sharex=True, sharey=True, figsize=(12, 15))
    for i, bandwidth in enumerate(bandwidths):
        times = []
        b_capacity = []
        t = 0
        while t < T:
            times.append(t)
            thr = bandwidth.get_thr(t)
            b_capacity.append(thr)
            t += dt

        # window_sec = 5
        # window = int(window_sec * 1 / dt)

        # average_cap = []
        # for i in range(len(b_capacity)):
        #     average_cap.append(np.average(b_capacity[i:i + window]))
        plt.subplot(5, 3, i + 1)
        plt.plot(times, b_capacity, label="Index {0}".format(i + 1))
        plt.subplot(5, 3, i + 1).set_title('Network profile index {0}'.format(i + 1))
        # plt.plot(times, average_cap, label="Window-Averaged Bandwidth")

        # plt.title("Window = {0} sec".format(window_sec))
    margin = 1
    plt.subplots_adjust(bottom=margin/5, right=margin, top=margin*1.3)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Mbps")
    fig.text(0.55, 0.15, 'Time (s)', ha='center', fontsize=20)
    fig.text(0.05, 0.75, 'Bandwidth capacity (Mbps)', va='center', rotation='vertical', fontsize=20)
    # plt.legend()
    plt.savefig("resulds/bandwidth/network_profiles.pdf", bbox_inches='tight')



data_path = "dataset/bandwidth/4G/4G_BWData.json"
# data_path = "fix_bandwidth.json"

bandwidth_error = 0.10

bandwidths = []



bandwidth_dir = "dataset/bandwidth/4GLTE/"

bnd_numbers = []
for i in range(1, 19):
    if i not in [9, 11, 14, 15]:
        bnd_numbers.append(i)
for i, bnd_number in enumerate(bnd_numbers):
    bandwidth = Bandwidth(bandwidth_dir + "{0}.log.txt".format(bnd_number), error_rate=bandwidth_error, series=2)
    bandwidths.append(bandwidth)
    # if i > 0:
    #     break
bandwidth = Bandwidth(data_path, error_rate=bandwidth_error)
bandwidths.append(bandwidth)


plot_bandwidth(bandwidths)


