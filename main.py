import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataframe = pd.read_csv('mbti.csv', names=["type", "tweets"], dtype={
    "type": str, "tweets": str})
TWEETS_PER_PERSON = 50

col_names = {
    "type": "type",
    "tweets": "tweets",
    "extrovert": "I-E",
    "sensible": "N-S",
    "emotional": "T-F",
    "perceiving": "J-P"
}
personalities = {
    "extrovert": {"I": 0, "E": 1},
    "sensible": {"N": 0, "S": 1},
    "emotional": {"T": 0, "F": 1},
    "perceiving": {"J": 0, "P": 1}
}
# _PC means per comment
new_col_maps = {
    "Longs_PC": "...",
    "Surprise_PC": "!",
    "Images_PC": "jpg",
    "Questions_PC": "?",
    "Music_PC": "music",
    "URLS_PC": "http",
    "Words_PC": " "
}

index = 0
for personality, symbol_map in personalities.items():
    dataframe[personality] = dataframe["type"].str[index].map(symbol_map)
    index += 1

for newcol, newcolcounter in new_col_maps.items():
    dataframe[newcol] = dataframe["tweets"].apply(lambda x: x.count(newcolcounter) / TWEETS_PER_PERSON)

dataframe = dataframe.fillna(0)
X = dataframe.drop(["type", "emotional", "sensible", "extrovert", "perceiving", "tweets"], axis=1).values
y = dataframe[["extrovert"]].values.ravel()
# print(X.shape)
# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

model = LogisticRegression(solver="newton-cg", max_iter=100)
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)

accuracy = round(model.score(X_train, y_train) * 100, 2)
print(round(accuracy, 2, ), "%")  # 77% accuracy
#to display predictions for each tweet
for i in range(X_test.shape[0]):
     print(f"{tweets[i]} = {y_test[i]}")
