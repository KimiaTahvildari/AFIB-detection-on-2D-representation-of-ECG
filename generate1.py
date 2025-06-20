import tensorflow as tf
import wfdb
import pandas as pd
from scipy.signal import spectrogram, cwt, ricker, get_window
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm
import matplotlib.cm as cm
import os

plt.ioff()

# Parameters
AFIB_THRESHOLD = 0.5
WINDOW_SECONDS = 5
GLOBAL_NR_AFIB = 0
GLOBAL_NR_nonAFIB = 0

class TrainProbeGenerator:
    def __init__(self, signal_path=None):
        if not signal_path:
            raise ValueError("No signal path provided")
        self.signal_path = signal_path
        self.ann = wfdb.rdann(self.signal_path, 'atr')
        self.ann_samples = self.ann.sample
        self.fs = self.ann.fs
        self.record = wfdb.rdrecord(self.signal_path)
        ecg = self.record.p_signal
        df = pd.DataFrame(ecg, columns=['ECG1', 'ECG2'])
        self.dataframe = df.assign(rhythm='NOISE', segment=0)
        self.__update_rhythms()
        self.window_size = int(WINDOW_SECONDS * self.fs)
        self.list_of_segments = self.split_dataframe_into_smaller(self.dataframe, chunk_size=self.window_size)

    def __update_rhythms(self):
        for idx, note in enumerate(self.ann.aux_note):
            start = self.ann_samples[idx]
            end = self.ann_samples[idx+1] if idx+1 < len(self.ann_samples) else len(self.dataframe)
            # match any annotation containing 'AFIB' (no parentheses assumption)
            new_df = pd.DataFrame(
                {'rhythm': note, 'segment': idx+1},
                index=range(start, end)
            )
            self.dataframe.update(new_df)

    @staticmethod
    def split_dataframe_into_smaller(df, chunk_size=3000):
        segments = []
        n_chunks = len(df) // chunk_size
        for i in range(n_chunks):
            chunk = df[i*chunk_size:(i+1)*chunk_size]
            if len(chunk) < chunk_size:
                continue
            # count any rhythm entries containing 'AFIB'
            n_afib = chunk['rhythm'].str.contains('AFIB', case=False).sum()
            ratio = n_afib / chunk_size
            label = 'AFIB' if ratio >= AFIB_THRESHOLD else 'NORMAL'
            segments.append({'index': i, 'ratio': ratio, 'rhythm': label, 'chunk': chunk})
        return segments

    @staticmethod
    def show_spectrogram(segment, fs):
        fig = plt.figure()
        window = get_window('hamming', 128)
        f, t, Sxx = spectrogram(segment, fs=fs, window=window)
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), cmap=cm.gray)
        plt.axis('off')
        plt.margins(0, 0)
        return fig

    @staticmethod
    def show_scalogram(segment):
        fig = plt.figure()
        widths = np.arange(1, 31)
        cwt_mat = cwt(segment, ricker, widths)
        plt.imshow(cwt_mat, extent=[-1, 1, 1, 31], cmap=cm.gray, aspect='auto')
        plt.axis('off')
        plt.margins(0, 0)
        return fig

    @staticmethod
    def show_attractor(segment):
        tau = 220
        x = segment[2*tau:]
        y = segment[tau:-tau]
        z = segment[:-2*tau]
        v = (x + y - 2*z) / np.sqrt(6)
        w = (x - y) / np.sqrt(2)
        try:
            nbins = 300
            k = gaussian_kde([v, w])
            xi, yi = np.mgrid[v.min():v.max():nbins*1j, w.min():w.max():nbins*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            fig = plt.figure()
            plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=cm.gray)
            plt.axis('off')
            plt.margins(0, 0)
            return fig
        except Exception:
            return None

# Directory setup
base_dir = '/home/kimia/AFIB-detection-on-2D-representation-of-ECG/AFDB'
output_root = os.path.join(base_dir, 'outputs')
types = ['AFIB', 'NORMAL']
plots = ['spectrogram', 'scalogram', 'attractor']
# create subdirectories for each label and plot type
for lbl in types:
    for p in plots:
        os.makedirs(os.path.join(output_root, lbl, p), exist_ok=True)

# Process all records
for fname in tqdm(os.listdir(base_dir)):
    if not fname.endswith('.dat'):
        continue
    rec = os.path.join(base_dir, fname[:-4])
    gen = TrainProbeGenerator(rec)
    for seg in tqdm(gen.list_of_segments):
        ecg1 = seg['chunk']['ECG1'].values
        label = seg['rhythm']
        idx = seg['index']
        # generate and save spectrogram
        fig1 = TrainProbeGenerator.show_spectrogram(ecg1, gen.fs)
        fig1.savefig(os.path.join(output_root, label, 'spectrogram', f"{label}_{idx}.jpg"),
                     bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig1)
        # generate and save scalogram
        fig2 = TrainProbeGenerator.show_scalogram(ecg1)
        fig2.savefig(os.path.join(output_root, label, 'scalogram', f"{label}_{idx}.jpg"),
                     bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig2)
        # generate and save attractor
        fig3 = TrainProbeGenerator.show_attractor(ecg1)
        if fig3 is not None:
            fig3.savefig(os.path.join(output_root, label, 'attractor', f"{label}_{idx}.jpg"),
                         bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close(fig3)
