import pandas as pd
import numpy as np
import math
import pickle

import librosa


class FeatureEngineering:


    def __init__(self, df, frame_length, hop_length, signal_rate):
        self.df = df
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.sr = signal_rate

    def amplitude_envelope(self, signal):
        """Calculate the amplitude envelope of a signal with a given frame size and hop length"""
        amplitude_envelope = []
        
        # calculate amplitude envelope for each frame
        for i in range(0, len(signal), self.hop_length): 
            amplitude_envelope_current_frame = max(signal[i:i+self.frame_length]) 
            amplitude_envelope.append(amplitude_envelope_current_frame)
        
        return np.array(amplitude_envelope)  
    
    def rms(self, signal):
        return librosa.feature.rms(signal, frame_length=self.frame_length, hop_length=self.hop_length)[0]

    def zero_crossing_rate(self, signal):
        return librosa.feature.zero_crossing_rate(y=signal, frame_length=self.frame_length, hop_length=self.hop_length)[0]

    def spectral_centroid(self, signal):
        return librosa.feature.spectral_centroid(y=signal, sr=self.sr, n_fft=self.frame_length, hop_length=self.hop_length)[0]

    def spectral_bandwidth(self, signal):
        return librosa.feature.spectral_bandwidth(y=signal, sr=self.sr, n_fft=self.frame_length, hop_length=self.hop_length)[0]
    
    def calculate_split_frequency_bin(self, split_frequency, num_frequency_bins):
        """Infer the frequency bin associated to a given split frequency."""
        
        frequency_range = self.sr / 2
        frequency_delta_per_bin = frequency_range / num_frequency_bins
        split_frequency_bin = math.floor(split_frequency / frequency_delta_per_bin)
        return int(split_frequency_bin)
    

    def band_energy_ratio(self, spectrogram, split_frequency=50):
        """Calculate band energy ratio with a given split frequency."""        

        split_frequency_bin = self.calculate_split_frequency_bin(split_frequency, len(spectrogram[0]))
        band_energy_ratio = []
        
        # calculate power spectrogram
        power_spectrogram = np.abs(spectrogram) ** 2
        power_spectrogram = power_spectrogram.T
        
        # calculate BER value for each frame
        for frame in power_spectrogram:
            sum_power_low_frequencies = frame[:split_frequency_bin].sum()
            sum_power_high_frequencies = frame[split_frequency_bin:].sum()
            band_energy_ratio_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
            band_energy_ratio.append(band_energy_ratio_current_frame)
        
        return np.array(band_energy_ratio)



    def create_aggregated_df(self, noise=False):
        if noise:
            signal = self.white_noise_augmentation(self.df['signal'].to_numpy(), .01)
        else:
            signal = self.df['signal'].to_numpy()

        # Time features
        AE = self.amplitude_envelope(signal)
        rms = self.rms(signal)
        zcs = self.zero_crossing_rate(signal)

        # Preprocessing frequencies
        fft = np.fft.fft(signal)
        spectrum = np.abs(fft)
        f = np.linspace(0, 22050, len(spectrum))
        self.df['FFT'] = fft
        self.df['Spectrum'] = spectrum
        self.df['Frequency'] = f

        # Frequency features
        sc = self.spectral_centroid(signal)
        sb = self.spectral_bandwidth(signal)

         #how to split frequency?

        new_df = pd.DataFrame(data={'amplitude_envelope': AE, "rms":rms, "zero_crossing_rate":zcs, "spectral_centroid":sc, "spectral_bandwidth":sb, "band_energy_ratio":myexp_ber})
      
        new_df['alone'] = df['alone'][int(self.frame_length/2)::self.hop_length].tolist()
        new_df['small'] = df['small'][int(self.frame_length/2)::self.hop_length].tolist()
        new_df['medium'] = df['medium'][int(self.frame_length/2)::self.hop_length].tolist()
        new_df['large'] = df['large'][int(self.frame_length/2)::self.hop_length].tolist()

        return new_df
        
    def white_noise_augmentation(self, signal, noise_factor):
        noise = np.random.normal(0, signal.std(), signal.size)
        augm_signal= signal + noise_factor* noise
        return augm_signal
        

def single_column(df):
    df['label'] = df[['alone', 'small', 'medium', 'large']].idxmax(axis=1).apply(['alone', 'small', 'medium', 'large'].index)

    df = df.drop(['alone', 'small', 'medium', 'large'], axis=1)

    return df



if __name__ == "__main__":
    df = pd.read_csv('audio_result_signal.csv')
    pickle.dump(df, open( "df.p", "wb" ) )

    df = pickle.load( open( "df.p", "rb" ) )
    obj = FeatureEngineering(df=df, frame_length=88200)
    # new_df = obj.create_aggregated_df()

    # new_df = single_column(new_df)

    # new_df.to_csv('df_features_4s.csv')

    new_df_aug = obj.create_aggregated_df(noise=True)

    new_df_aug = single_column(new_df_aug)

    new_df_aug.to_csv('df_features_aug_4s.csv')

  

