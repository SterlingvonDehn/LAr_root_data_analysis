import math
import multiprocessing
import os
import pathlib
import random

# from helper import timer_func
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from feb2_analysis.plotting import helper
from www.board_testing.generate_html import make_html


def TimeFunction(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        out = func(*args, **kwargs)
        print(f"{func.__name__} took {(perf_counter()-start):.3f}s")
        return out

    return wrapper


# ----
def make_standard_pedestal_plots(df, specGain, channel=None, plotDir="plots/"):
    print("pre-plotting starts")
    """for i in range(24,59):
        df= df[df.channel!=f'channel0{i}']
    for i in range(80,129):
        df= df[df.channel!=f'channel0{i}']
    """
    t1_start = perf_counter()
    run = df.iloc[0].run_id
    if channel is None:
        channel = df.iloc[0].channel
    runName = "run" + str(run)
    if not (os.path.exists(plotDir + runName)):
        pathlib.Path(plotDir + runName).mkdir(parents=True, exist_ok=True)
    plotDir += runName + "/"
    print(f"saving plots to {plotDir} ...")
    t1_stop = perf_counter()
    print("pre-plotting takes " + str(t1_stop - t1_start) + " s")

    processes = []
    functions = [
        [create_correlation_matrix_gain, [df, plotDir, "hi"]],
        [create_correlation_matrix_gain, [df, plotDir, "lo"]],
        [plot_cnoise_all, [df, plotDir]],
        [plot_baseline_means_rms, [df, plotDir]],
        [do_fits_autocorr, [df, specGain, channel, plotDir]],
        [do_fits_fft, [df, specGain, channel, plotDir]],
        [do_fits_hist, [df, specGain, channel, plotDir]],
        [do_fits_raw, [df, specGain, channel, plotDir]],
    ]

    for function, args in functions:
        process = multiprocessing.Process(target=function, args=args)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("All procedures are completed.")

    make_html(runName, "pedestal", specGain, channel)


@TimeFunction
def do_fits_raw(df, specGain, channel=None, plotDir="plots/"):
    df.apply(lambda r: plot_raw(r, plotDir), axis=1)


@TimeFunction
def do_fits_hist(df, specGain, channel=None, plotDir="plots/"):
    df.apply(lambda r: plot_hist(r, plotDir), axis=1)


@TimeFunction
def do_fits_autocorr(df, specGain, channel=None, plotDir="plots/"):
    df.apply(lambda r: plot_autocorr(r, plotDir), axis=1)


@TimeFunction
def do_fits_fft(df, specGain, channel=None, plotDir="plots/"):
    df.apply(lambda r: plot_fft(r, plotDir), axis=1)


@TimeFunction
def do_fits_all(df, specGain, channel=None, plotDir="plots/"):
    do_fits_raw(df, specGain, channel=channel, plotDir=plotDir)
    do_fits_hist(df, specGain, channel=channel, plotDir=plotDir)
    do_fits_autocorr(df, specGain, channel=channel, plotDir=plotDir)
    do_fits_fft(df, specGain, channel=channel, plotDir=plotDir)


# ----
def plot_raw(dfRow, plotDir):
    run = dfRow.run_id
    gain = dfRow.gain
    channel = dfRow.channel
    processed = dfRow.processed

    mu = round(np.mean(processed), 2)
    std = round(np.std(processed), 2)

    plt.figure(figsize=(10, 10))
    plt.plot(processed, "k.", label=rf"Samples\n$\mu$ = {mu}, $\sigma$ = {std}")
    plt.legend(loc=1)
    plt.title(f"Run {run}, {channel}, Pedestal")
    plt.xlabel("Sample #")
    plt.ylabel("ADC Counts")
    plt.grid()
    plt.tight_layout()
    plt.savefig(plotDir + f"/rawPed_meas0_{channel}_{gain}.png")
    plt.cla()
    plt.clf()
    plt.close()


# ----
def plot_hist(dfRow, plotDir):
    run = dfRow.run_id
    gain = dfRow.gain
    channel = dfRow.channel
    processed = dfRow.processed
    fig, ax = plt.subplots()

    std = round(np.std(processed), 2)

    n, b = np.histogram(processed, bins=range(min(processed) - 1, max(processed) + 1, 1))

    bwidth = 1
    bins = np.arange(min(processed), max(processed) + bwidth, bwidth)
    if len(bins) > 1:
        ax.hist(
            processed,
            bins=bins,
            align="mid",
            edgecolor="black",
            label=f"Samples: RMS={std}",
        )
    else:
        ax.hist(processed, align="mid", edgecolor="black", label=f"Samples: RMS={std}")

    x = np.linspace(min(processed), max(processed), 1000)
    fit_pars = helper.calc_gaussian(processed, b)
    fit_mu, fit_sigma, fit_N = fit_pars[0], fit_pars[2], fit_pars[4]
    gauss_fit = helper.gauss(x, fit_mu, fit_sigma, fit_N)
    rounded = [round(par, 3) for par in fit_pars]
    ax.plot(
        x,
        gauss_fit,
        color="r",
        label=rf"Fit: $\mu$ = {rounded[0]}, $\sigma$ = {rounded[2]}",
    )

    ax.legend(loc="upper right")
    ax.set_xlabel("ADC Counts")
    left, right = ax.get_xlim()
    ax.set_xlim(left, right + (left - right) * -0.1)
    left, right = ax.get_ylim()
    ax.set_ylim(left, right + (left - right) * -0.1)
    ax.set_ylabel("Entries")
    ax.set_title(f"Run {run}, {channel}, Pedestal")
    plt.grid()
    plt.tight_layout()
    plt.savefig(plotDir + f"/{channel}_{gain}_pedestal_hist.png")
    plt.cla()
    plt.clf()
    plt.close()


# ----
def plot_autocorr(dfRow, plotDir):
    run = dfRow.run_id
    gain = dfRow.gain
    channel = dfRow.channel
    acorr = dfRow.acorr
    plt.plot(acorr[0:50], "k.-", label="Autocorrelation")
    plt.axhline(y=0, color="r", ls="--")
    plt.legend(loc="upper right")
    plt.title(f"Run {run}, {channel}, Pedestal")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.grid()
    plt.tight_layout()
    plt.savefig(plotDir + f"/autocorr_{channel}_{gain}.png")
    plt.cla()
    plt.clf()
    plt.close()


# ----
def plot_baseline_means_rms(df, plotDir):
    all_ch = df.channel
    all_ch = sorted(all_ch)
    baseline_means_rms(df, plotDir, all_ch)


# ----
def baseline_means_rms(df, plotDir, chType):
    summaryPlotString = df.iloc[0].summaryPlotString
    fig, ax = plt.subplots()
    plt.xticks(np.arange(0, 129, 4), rotation=70)
    fig2, ax2 = plt.subplots(1)

    df_lo = df[df.gain == "lo"]
    df_hi = df[df.gain == "hi"]

    data_lo = []
    data_hi = []

    for ch in chType:
        data_hi.append(df_hi[df_hi.channel == ch].iloc[0].processed)
        data_lo.append(df_lo[df_lo.channel == ch].iloc[0].processed)

    for col, title, data in [("b", "LG", data_lo), ("r", "HG", data_hi)]:
        names = chType
        names = [
            name[:2] + name[7:] for name in names
        ]  # very janky method to change 'channelxxx' --> 'chxxx' in bar labels

        mus = [round(np.mean(processed), 2) for processed in data]
        stds = [round(np.std(processed), 2) for processed in data]
        ax.grid(visible=True, zorder=0)

        ax.bar(names, mus, fill=False, ec=col, label=title, zorder=3)

        ax.set_title("Mean Pedestal Value")
        ax.set_ylabel("ADC Counts")
        ax.set_ylim(0, max(mus) + max(mus) / 3)
        ax.legend()
        ax.annotate(summaryPlotString, (0.05, 0.85), xycoords="axes fraction")

        fig.savefig(f"{plotDir}/mu_summary.png")

        ax2.grid(visible=True, zorder=0)
        plt.xticks(np.arange(0, 129, 4), rotation=70)
        mean = np.mean(stds)
        ax2.bar(
            names,
            stds,
            fill=False,
            ec=col,
            label="{} mean = {:.2f}".format(title, mean),
            zorder=3,
        )
        ax2.set_title("Pedestal RMS")
        ax2.set_ylabel("ADC Counts")
        ax2.set_ylim(0, 25)
        ax2.legend()
        ax2.annotate(summaryPlotString, (0.05, 0.85), xycoords="axes fraction")
        fig2.savefig(f"{plotDir}/rms_summary.png")

    plt.cla()
    plt.clf()
    plt.close()


def create_pseudonoice(df):
    df_new = {}
    for i in range(1, 128):
        df_add = df[df.channel == f"channel0{random.randint(48, 79)}"]
        df_add.channel = f"channel{i}"
        df_new.update(df_add)
    return df_new


# ----
@TimeFunction
def plot_cnoise_all(df, plotDir):
    plot_cnoise(df, plotDir, 0, 128)
    for i in range(0, 2):
        plot_cnoise(df, plotDir, 64 * i, 64)
    for i in range(0, 8):
        plot_cnoise(df, plotDir, 16 * i, 16)
    for i in range(0, 32):
        plot_cnoise(df, plotDir, 4 * i, 4)


# ----
def plot_cnoise(df, plotDir, min_channel, nchannels):
    run = df.iloc[0].run_id
    df_lo = df[df.gain == "lo"]
    df_hi = df[df.gain == "hi"]

    gain_data = {"lo": df_lo, "hi": df_hi}
    for gain in gain_data.keys():
        ax = plt.subplot(111)
        gain_df = gain_data[gain]
        for i in range(0, min_channel):
            gain_df = gain_df[gain_df.channel != f"channel{i}"]
            gain_df = gain_df[gain_df.channel != f"channel0{i}"]
            gain_df = gain_df[gain_df.channel != f"channel00{i}"]

        for i in range(min_channel + nchannels, 129):
            gain_df = gain_df[gain_df.channel != f"channel{i}"]
            gain_df = gain_df[gain_df.channel != f"channel0{i}"]
            gain_df = gain_df[gain_df.channel != f"channel00{i}"]

        if gain_df.empty:
            return 0

        channel_name = gain_df["channel"].iloc[0]

        data = {}
        rms = {}
        d_rms = {}
        ch_noise = 0
        d_ch_noise = 0
        dataSum = []
        for channel in gain_df.channel:
            processed = gain_df[gain_df.channel == channel].iloc[0].processed
            data[channel] = processed
            rms[channel] = np.std(processed - np.mean(processed))
            d_rms[channel] = rms[channel] / np.sqrt(len(processed))
            ch_noise += rms[channel] ** 2
            d_ch_noise += d_rms[channel] ** 2
        ch_noise = np.sqrt(ch_noise)
        d_ch_noise = np.sqrt(d_ch_noise)
        avg_noise = ch_noise / np.sqrt(len(gain_df.channel))
        d_avg = d_ch_noise / np.sqrt(len(gain_df.channel))

        gain_df_channels = gain_df.channel
        data_by_channel = np.array([data[ch] for ch in gain_df_channels])
        channel_means = np.mean(data_by_channel, axis=1)
        dataSum = np.sum([(data[ch] - channel_means[i]) for i, ch in enumerate(gain_df_channels)], axis=0)

        tot_noise = np.std(dataSum)
        mu = round(np.mean(dataSum), 3)
        std = round(np.std(dataSum), 3)
        y, x, _ = plt.hist(
            dataSum,
            bins=np.arange(min(dataSum), max(dataSum) + 2, 1),
            color="steelblue",
            edgecolor="black",
            density=False,
            label=rf"RMS = {np.round(tot_noise,3)}\n$\mu$ = {np.round(mu,3)}, $\sigma$ = {np.round(std, 3)}",
        )
        lnspc = np.linspace(min(dataSum), max(dataSum), 1000)
        fit_pars = helper.calc_gaussian(dataSum, x)
        fit_mu, fit_sigma, fit_N = fit_pars[0], fit_pars[2], fit_pars[4]
        gauss_fit = helper.gauss(lnspc, fit_mu, fit_sigma, fit_N)
        rounded = [round(par, 3) for par in fit_pars]
        plt.plot(
            lnspc,
            gauss_fit,
            color="r",
            label=rf"Gaussian Fit\n$\mu$ = {rounded[0]}, $\sigma$ = {rounded[2]}",
        )
        coh_noise = np.sqrt(tot_noise**2 - ch_noise**2) / len(gain_df.channel)
        d_coh = np.sqrt(
            (tot_noise / np.sqrt(len(data[channel_name])) * (tot_noise / coh_noise / (len(gain_df.channel) ** 2))) ** 2
            + (d_ch_noise * (ch_noise / coh_noise / (len(gain_df.channel) ** 2))) ** 2
        )
        pct_coh = coh_noise / avg_noise * 100
        d_pct = pct_coh * np.sqrt((d_coh / coh_noise) ** 2 + (d_avg / avg_noise) ** 2)
        plt.ylim(0, max(y) * 1.4)
        plt.xlabel("ADC Counts")
        plt.ylabel("Entries")
        plt.title(
            f"Run {run}, {gain_df.iloc[0].meas_type}, {gain} Gain, Channels {min_channel}-{min_channel+nchannels-1}"
        )
        plt.grid()
        plt.text(
            0.95,
            0.88,
            r"$\sqrt{\sigma_i^2}$ = "
            + rf"{round(ch_noise, 2)} "
            + r"$\pm$"
            + f" {round(d_ch_noise,2)}\nAvg. noise/ch = {round(avg_noise,2)} "
            + r"$\pm$"
            + f" {round(d_avg,2)}\nCoh. noise/ch = {round(coh_noise,2)} "
            + r"$\pm$"
            + f" {round(d_coh,2)}\n[%] Coh. noise = {round(pct_coh,2)} "
            + r"$\pm$"
            + f" {round(d_pct,2)}",
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        plt.legend(loc="upper left")
        if nchannels == 128:
            plt.savefig(plotDir + f"/coherence_all_{gain}_pedestal_hist.png")
        else:
            plt.savefig(plotDir + f"/{gain}_cnoise_ch_{min_channel}_{min_channel+nchannels}.png")
        plt.cla()
        plt.clf()
        plt.close()


def create_correlation_matrix(df, plotDir, channel=None):
    df_lo = df[df.gain == "lo"].copy()
    df_hi = df[df.gain == "hi"].copy()
    gain_data = {"lo": df_lo, "hi": df_hi}
    for gain in gain_data.keys():
        gain_df = gain_data[gain]
        lowest_channel = df["channel"].sort_values().iloc[0]
        lowest_channel = int(lowest_channel[-3:])
        channels = gain_df.channel.unique()
        matrix = np.zeros((128, 128))
        for i in range(0, 128):
            for j in range(0, 128):
                if i == j:
                    elem = 1.0
                else:
                    random_number = random.gauss(0, 0.0002)
                    elem = random_number
                matrix[i][j] = elem
        for row, ch1 in enumerate(channels):
            data1 = gain_df[gain_df.channel == ch1].iloc[0].processed
            for col, ch2 in enumerate(channels):
                data2 = gain_df[gain_df.channel == ch2].iloc[0].processed
                channel1 = int(ch1[-3:])
                channel2 = int(ch2[-3:])
                try:
                    corr, _ = stats.pearsonr(data1, data2)
                except ValueError:
                    corr = np.nan
                matrix[channel1][channel2] = corr
        nchannels_list = [128, 64, 64, 16, 16, 16, 16, 16, 16, 16, 16]
        first_channel_list = [0, 0, 64, 0, 16, 32, 48, 64, 80, 96, 112]
        for nchannels, first_channel in zip(nchannels_list, first_channel_list):
            plot_correlation_matrix(df, plotDir, channel, gain, matrix, nchannels, first_channel)


@TimeFunction
def create_correlation_matrix_gain(df, plotDir, gain, channel=None):
    gain_df = df[df.gain == gain].copy()
    lowest_channel = df["channel"].sort_values().iloc[0]
    lowest_channel = int(lowest_channel[-3:])
    channels = gain_df.channel.unique()
    matrix = np.zeros((128, 128))
    for i in range(0, 128):
        for j in range(0, 128):
            if i == j:
                elem = 1.0
            else:
                random_number = random.gauss(0, 0.0002)
                elem = random_number
            matrix[i][j] = elem
    for row, ch1 in enumerate(channels):
        data1 = gain_df[gain_df.channel == ch1].iloc[0].processed
        for col, ch2 in enumerate(channels):
            data2 = gain_df[gain_df.channel == ch2].iloc[0].processed
            channel1 = int(ch1[-3:])
            channel2 = int(ch2[-3:])
            try:
                corr, _ = stats.pearsonr(data1, data2)
            except ValueError:
                corr = np.nan
            matrix[channel1][channel2] = corr
    nchannels_list = [128, 64, 64, 16, 16, 16, 16, 16, 16, 16, 16]
    first_channel_list = [0, 0, 64, 0, 16, 32, 48, 64, 80, 96, 112]
    for nchannels, first_channel in zip(nchannels_list, first_channel_list):
        plot_correlation_matrix(df, plotDir, channel, gain, matrix, nchannels, first_channel)


def plot_correlation_matrix(df, plotDir, channel, gain, matrix, nchannels, first_channel):
    fig, ax = plt.subplots(figsize=(nchannels / 4, nchannels / 4))
    submatrix = matrix[first_channel : first_channel + nchannels, first_channel : first_channel + nchannels]
    _ = ax.imshow(submatrix, cmap="RdBu", vmin=-0.3, vmax=0.3)
    ax.set_xticks(np.arange(0, nchannels, max(1, round(nchannels / 32))))
    ax.set_yticks(np.arange(0, nchannels, max(1, round(nchannels / 32))))
    ax.set_xticklabels(
        np.arange(first_channel, first_channel + nchannels, max(1, round(nchannels / 32))), rotation=90, fontsize=7
    )
    ax.set_yticklabels(np.arange(first_channel, first_channel + nchannels, max(1, round(nchannels / 32))), fontsize=7)
    for i in range(first_channel, min(128, first_channel + nchannels)):
        for j in range(first_channel, min(128, first_channel + nchannels)):
            # print(f"{i}, {j}: {matrix[i,j]}")
            if i == j:
                color = "w"
            else:
                color = "k"
            if math.isnan(matrix[i, j]):
                val_to_write = " "
            else:
                val_to_write = int(round(matrix[i, j] * 100))
            _ = ax.text(
                j - first_channel,
                i - first_channel,
                val_to_write,
                ha="center",
                fontsize=6,
                va="center",
                color=color,
            )
    ax.set_title(f"Pearson Correlation [%], {gain} gain, channels {first_channel}-{first_channel+nchannels} only")
    fig.tight_layout()
    if nchannels == 128:
        plt.savefig(plotDir + f"{gain}_corr.png")
    else:
        plt.savefig(plotDir + f"{gain}_corr_ch_{first_channel}_{first_channel+nchannels}.png")
    plt.cla()
    plt.clf()
    plt.close()


# ------------------
def plot_fft(dfRow, plotDir):
    channel = dfRow.channel
    gain = dfRow.gain
    freq = dfRow.freq
    psd = dfRow.psd
    enob = round(dfRow.enob, 2)
    snr = round(dfRow.snr, 2)
    sinad = round(dfRow.sinad, 2)
    sfdr = round(dfRow.sfdr, 2)

    plt.plot(
        freq,
        psd,
        color="k",
        label=f"ENOB: {enob} bits\nSNR: {snr} dB\nSINAD: {sinad} dB\nSFDR: {sfdr}",
    )
    plt.legend(loc=1)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("PSD [dB]")
    plt.grid()
    plt.tight_layout()
    _ = plt.subplot(111)
    plt.savefig(f"{plotDir}/pedestal_FFT_{channel}_{gain}.png")
    plt.cla()
    plt.clf()
    plt.close()
