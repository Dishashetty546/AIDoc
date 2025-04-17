import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Datasets/Training.csv")

# df.head()
# print(df.shape)
# p=len(df['prognosis'].unique())
# print(p)
# pl= df['prognosis'].unique()
# print(pl)

X= df.drop("prognosis",axis=1)     # all input columns
y= df["prognosis"]                 #output column



