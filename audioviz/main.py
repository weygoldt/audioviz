from pathlib import Path
import moviepy.editor as mpe
import argparse
import torch
import torchaudio
from torchaudio.transforms import Spectrogram, AmplitudeToDB
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
import subprocess
from glob import glob
import os

con = Console()

testpath = Path(
    "../mscthesis/data/raw/local/competition_subsets/subset_2020-03-16-10_00_t0_200.0_t1_800.0/traces_grid1.wav"
)

view_window = 10
res = (3840, 2160)
framerate = 30
dpi = 300
figsize = (np.ceil(res[0] / dpi), np.ceil(res[1] / dpi))


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def add_audio(videopath, audiopath):
    my_clip = mpe.VideoFileClip(videopath)
    audio_background = mpe.AudioFileClip(audiopath)
    # print(dir(my_clip))
    # print(audio_background.nchannels)
    # final_audio = mpe.CompositeAudioClip([my_clip.audio, audio_background])
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile("withaudio.mp4")


def setup_spectrumplot():
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel("Frequency [Hz]")
    return fig, ax


def to_spectrogram(wave):
    device = get_device()
    spec = Spectrogram(n_fft=4096, power=None).to(device)
    decibel = AmplitudeToDB().to(device)

    s = spec(wave)
    s = s.abs().pow(2)
    s = decibel(s)

    if len(s.shape) > 3:
        msg = "Resulting spectrogram has more than 3 dimensions, something is wrong!"
        raise Exception(msg)

    if len(s.shape) == 3:
        s = torch.sum(s, dim=0)

    return s


def plot_frame(
    rate,
    spec,
    buffersize,
    starttime,
    indicatortime,
    freqview,
    timeview,
):
    fig, ax = setup_spectrumplot()
    time = np.linspace(0, view_window, spec.shape[1])
    freq = np.linspace(0, rate // 2, spec.shape[0])
    spec = spec.cpu().numpy()

    ax.pcolormesh(
        time,
        freq,
        spec,
        cmap="magma",
        rasterized=True,
    )

    scalebar_ypos = np.diff(freqview) * 0.1
    scalebar_xpos = np.diff(timeview) * 0.85
    scalebar_len = np.diff(timeview) // 10
    scalebar_len = scalebar_len[0]

    ax.plot(
        [scalebar_xpos, scalebar_xpos + scalebar_len],
        [scalebar_ypos, scalebar_ypos],
        lw=5,
        c="white",
        rasterized=True,
    )
    ax.text(
        scalebar_xpos + scalebar_len / 2,
        (scalebar_ypos - np.diff(freqview) / 50),
        f"{scalebar_len} s",
        ha="center",
        va="center",
        color="white",
        rasterized=True,
    )
    ax.axvline(indicatortime, lw=1, ls="dashed", c="white")
    ax.set_ylim(*freqview)
    ax.set_xlim(*timeview)


def images_to_video():
    con.log("All plots saved, building video from images!")
    subprocess.call(
        [
            "ffmpeg",
            "-framerate",
            f"{framerate}",
            "-i",
            "file_%06d.png",
            "-r",
            f"{framerate}",
            "-pix_fmt",
            "yuv420p",
            "output.mp4",
        ]
    )
    con.log("Video saved, deleting single frames!")
    for file_name in glob("*.png"):
        os.remove(file_name)


def main():
    device = get_device()
    wave, rate = torchaudio.load(str(testpath), backend="sox")
    wave = wave.to(device)
    subwave = wave[:, : rate * view_window]
    spec = to_spectrogram(subwave)

    freqview = (0, 2000)
    timepoints = np.arange(0, view_window, 1 / framerate)

    # for i, timepoint in enumerate(timepoints):
    #     starttime = 0
    #     timeview = (0, view_window)
    #     time_indicator_position = timepoint
    #
    #     plot_frame(
    #         rate,
    #         spec,
    #         view_window,
    #         starttime,
    #         time_indicator_position,
    #         freqview,
    #         timeview,
    #     )
    #     plt.savefig("file_%06d.png" % i, dpi=300, bbox_inches="tight")
    #     plt.close()
    #     con.log(f"Saved frame {i}/{len(timepoints)}")

    # images_to_video()
    add_audio("output.mp4", str(testpath))
    con.log("Audio added and video saved successfully!")


if __name__ == "__main__":
    main()
