
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import math



FIG_SIZE = (15,10)
file = 'Exp2.wav'

class CreateAudio:
    file = ''
    sr = 0

    def __init__(self, file, sr):
        self.file = file
        self.sr = sr


    def add_features(self):
        signal, sample_rate = librosa.load(file, sr=self.sr) # sr * T -> 22050 * T(s)


        return pd.DataFrame(data={'signal': signal})


    def add_labels(self, df):

        start_lst = []
        start_time = [0,40,58,154,205,224,355,406,531,680,810,939,1020,1248,1408,1508, 1548, 1990, 2041, df.shape[0]/22050]

        rest = 0
        for index in range(len(start_time)-1):
            if index == (len(start_time)-2):
                value = (float(start_time[index+1]) - float(start_time[index])) * self.sr
                value += rest
                intpart,rest = int(value),value-int(value)
                intpart += 1
            else:
                value = (float(start_time[index+1]) - float(start_time[index])) * self.sr
                value += rest
                intpart,rest = int(value),value-int(value)
            lst = [0] * intpart
            start_lst.append(lst)

        A = [0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0]
        S = [1,1,1,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0]
        M = [0,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0,1,0,1]
        L = [0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0]
        
        labels = pd.DataFrame(data={'start_lst':start_lst, 'alone':A, "small":S, "medium":M, "large":L})

        labels = labels.explode('start_lst').reset_index(drop=True)


        df['alone'] = labels['alone']
        df['small'] = labels['small']
        df['medium'] = labels['medium']
        df['large'] = labels['large']


        return df


    def get_df(self):
        df = self.add_features()
        df = self.add_labels(df)

        return df

if __name__ == "__main__":
    obj = CreateAudio(file='Exp2.wav', sr=22050)
    df = obj.get_df()

    df.to_csv('audio_result_signal.csv')

