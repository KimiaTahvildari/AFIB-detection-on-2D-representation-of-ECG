import wfdb
import pandas as pd
from scipy.signal import spectrogram, get_window
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import pywt
from tqdm import tqdm
import matplotlib.cm as cm

AFIB_THRESHOLD = 0.5
WINDOW = 5


class SegmentVisualization(object):
    def __init__(self, signal_path):
        self.signal_path = signal_path
        self.ann = wfdb.rdann(self.signal_path, 'atr')
        self.ann_samples = self.ann.sample
        self.fs = self.ann.fs
        self.record = wfdb.rdrecord(self.signal_path)
        ecg = self.record.p_signal
        dataframe = pd.DataFrame(ecg, columns=['ECG1', 'ECG2'])
        self.dataframe = dataframe.assign(rhythm='NOISE', segment=0)
        self.__update_rhythms()
        self.__keys = ['index', 'ratio', 'rhythm', 'chunk']

        self.window_size = WINDOW * self.fs
        self.list_of_segments = self.split_dataframe_into_smaller(self.dataframe, chunk_size=self.window_size)
        self.lables = []

    def __update_rhythms(self):
        print('update rhythms')
        for index, rhytm in enumerate(self.ann.aux_note):
            if index + 1 >= len(self.ann_samples):
                new_df = pd.DataFrame({'rhythm': rhytm, 'segment': index + 1},
                                      index=range(self.ann_samples[index], len(self.dataframe)))
            else:
                new_df = pd.DataFrame({'rhythm': rhytm, 'segment': index + 1},
                                      index=range(self.ann_samples[index], self.ann_samples[index + 1]))
            self.dataframe.update(new_df)

    @staticmethod
    def split_dataframe_into_smaller(df, chunk_size=3000):
        list_of_df = list()
        number_chunks = len(df) // chunk_size + 1
        for i in range(number_chunks):
            chunk = df[i * chunk_size:(i + 1) * chunk_size]
            if len(chunk) == chunk_size:
                ratio = len(chunk.loc[chunk.rhythm == "(AFIB"]) / 3000
                rhythm = 'AFIB' if ratio >= AFIB_THRESHOLD else 'NORMAL'
                list_of_df.append({'index': i, 'ratio': ratio, 'rhythm': rhythm, 'chunk': chunk})
        return list_of_df

    @staticmethod
    def show_segment(segment):
        plt.figure()
        plt.plot(segment)
        plt.xlabel('Probes')
        plt.ylabel('ECG(mV)')
        plt.savefig("segment_plot.png")
        plt.close()

    @staticmethod
    def show_spectogram(segment):
        window = get_window('hamming', 128)
        f, t, Sxx = spectrogram(segment, fs=250.0, window=window)

        # Gray version
        plt.figure()
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), cmap='gray')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title("Spectrogram (Gray)")
        plt.savefig("spectrogram_gray.png")
        plt.close()

        # Blue version
        plt.figure()
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), cmap='Blues')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title("Spectrogram (Blues)")
        plt.savefig("spectrogram_blues.png")
        plt.close()

    @staticmethod
    def show_scalogram(segment):
        scales = np.arange(1, 128)
        sampling_rate = 250.0
        sampling_period = 1.0 / sampling_rate

        # Use Mexican hat wavelet
        coefficients, freqs = pywt.cwt(segment, scales, 'mexh', sampling_period=sampling_period)
        time = np.arange(len(segment)) * sampling_period
        extent = [time[0], time[-1], scales[-1], scales[0]]  # Flip y-axis (scale)

        # BrBG version
        plt.figure(figsize=(10, 4))
        plt.imshow(np.abs(coefficients), extent=extent, cmap='BrBG', aspect='auto')
        plt.xlabel("Time [sec]")
        plt.ylabel("Scale")
        plt.title("Scalogram (BrBG)")
        plt.savefig("scalogram_brbg.png")
        plt.close()

        # Blues version
        plt.figure(figsize=(10, 4))
        plt.imshow(np.abs(coefficients), extent=extent, cmap='Blues', aspect='auto')
        plt.xlabel("Time [sec]")
        plt.ylabel("Scale")
        plt.title("Scalogram (Blues)")
        plt.savefig("scalogram_blues.png")
        plt.close()

    @staticmethod
    def show_attractor(segment):
        tau = 220
        x = np.array(segment[2 * tau:])
        y = np.array(segment[tau:-tau])
        z = np.array(segment[:-2 * tau])

        mpl.rcParams['legend.fontsize'] = 10

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, label="attractor reconstruction (Tekens' delay)")
        ax.legend()
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.savefig("attractor_3d.png")
        plt.close()

        plt.figure()
        plt.plot(y, x)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig("attractor_xy.png")
        plt.close()

        v = (x + y - 2 * z) / np.sqrt(6)
        w = (x - y) / np.sqrt(2)
        plt.figure()
        plt.plot(v, w)
        plt.xlabel('v')
        plt.ylabel('w')
        plt.savefig("attractor_vw.png")
        plt.close()

        x = v
        y = w
        nbins = 300
        k = gaussian_kde([x, y])
        xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        plt.figure()
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
        plt.xlabel('v')
        plt.ylabel('w')
        plt.title("Attractor KDE")
        plt.savefig("attractor_kde.png")
        plt.close()

        plt.figure()
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens_r)
        plt.xlabel('v')
        plt.ylabel('w')
        plt.title("Attractor KDE (Green)")
        plt.savefig("attractor_kde_green.png")
        plt.close()

    def search(self, key='rhythm', value='AFIB', output=None):
        if key in self.__keys:
            if output and output in self.__keys:
                return [element[output] for element in self.list_of_segments if element[key] == value]
            return [element for element in self.list_of_segments if element[key] == value]
        else:
            print('Unknown key')


signal_path = r'/home/kimia/AFIB-detection-on-2D-representation-of-ECG/AFDB/04015'
my_visu = SegmentVisualization(signal_path)

all_segments = my_visu.search(key='rhythm', value='NORMAL')
segment = all_segments[100]['chunk']['ECG1']
my_visu.show_segment(segment)
my_visu.show_spectogram(segment)
my_visu.show_scalogram(segment)
my_visu.show_attractor(segment)
