import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv('audio_result_signal.csv')
def single_column(df):
    df['label'] = df[['alone', 'small', 'medium', 'large']].idxmax(axis=1).apply(['alone', 'small', 'medium', 'large'].index)

    df = df.drop(['alone', 'small', 'medium', 'large'], axis=1)

    return df


df = single_column(df)

listofzeros = [0] * len(df['label'])
print(df.head())
df['listofzeros'] = listofzeros

fig, ax = plt.subplots()

colors = {0:'red', 1:'green', 2:'blue', 3:'yellow'}


ax.scatter(df['listofzeros'], range(len(df['listofzeros'])), c=df['label'].map(colors))
plt.legend()
plt.show()

