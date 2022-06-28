import pandas as pd

df = pd.read_csv('df_features_4.csv')
print(df.head())
df['label'] = df[['alone', 'small', 'medium', 'large']].idxmax(axis=1)

df.drop(['alone', 'small', 'medium', 'large'], axis=1)
