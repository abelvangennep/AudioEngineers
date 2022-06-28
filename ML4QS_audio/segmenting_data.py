import fileinput
import json
import os
import math
import librosa
import pandas as pd
import pickle
from feature_engineering import FeatureEngineering

DATASET_PATH = "Exp2.wav"
JSON_PATH = "data_204.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 2111 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def single_column(df):
    df['label'] = df[['alone', 'small', 'medium', 'large']].idxmax(axis=1).apply(['alone', 'small', 'medium', 'large'].index)

    df = df.drop(['alone', 'small', 'medium', 'large'], axis=1)

    return df

def labels(df, num_segments):
    
    df = single_column(df)
    length_df =  df['label'].size

    step_size = int(length_df/ num_segments)
    start = int(step_size / 2)

    labels = df['label'][start::step_size].tolist()

    return labels


def data_prep(df, dataset, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=204):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": ["alone", "small", "medium", "large"],
        "labels": labels(df, num_segments),
        "mfcc": [],
        "ae": [],
        "rms": [],
        "zero_crossing_rate": [],
        "spectral_centroid": [],
        "spectral_bandwidth": []
    }

    samples_per_segment = round(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # load audio file
    signal, sample_rate = librosa.load(dataset, sr=SAMPLE_RATE)

    # process all segments of audio file
    for d in range(num_segments):
        
        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # Adding features
        features = FeatureEngineering(df = df, frame_length=n_fft, hop_length=hop_length, signal_rate=SAMPLE_RATE)
        ae = features.amplitude_envelope(signal[start:finish])
        rms = features.rms(signal[start:finish])
        zero_crossing_rate = features.zero_crossing_rate(signal[start:finish])
        spectral_centroid = features.spectral_centroid(signal[start:finish])
        spectral_bandwidth = features.spectral_bandwidth(signal[start:finish])
        # spectogram = librosa.stft(signal[start:finish], n_fft=n_fft, hop_length=hop_length)
        # band_energy_ratio = features.band_energy_ratio(spectogram, split_frequency=50)
  
        
        # extract mfcc
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        
        mfcc = mfcc.T

        # store only mfcc feature with expected number of vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            data["ae"].append(ae.tolist())
            data["rms"].append(rms.tolist())
            data["mfcc"].append(mfcc.tolist())
            data["zero_crossing_rate"].append(zero_crossing_rate.tolist())
            data["spectral_centroid"].append(spectral_centroid.tolist())
            data["spectral_bandwidth"].append(spectral_bandwidth.tolist())

            print("start", start)
            print("finish", finish)
            print(f"segment:{d+1}")

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
if __name__ == "__main__":
    # df = pd.read_csv('audio_result_signal.csv')
    df = pickle.load( open( "df.p", "rb" ) )
    # data_prep(df, "Exp2.wav", JSON_PATH, num_segments=204)