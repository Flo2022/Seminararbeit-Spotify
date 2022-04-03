import pandas as pd
import numpy as np
import sklearn

# import matplotlib.pyplot as plt  # To visualize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing


def read_data_from_csv():
       # Datensatz von https://www.kaggle.com/lehaknarnauli/spotify-datasets?select=tracks.csv
       df = pd.read_csv("./tracks.csv")
       return df

def data_preprocessing(df):
       df.dropna(axis=0, inplace=True)

       df = label_encoding(df)

       # Erklärung für die Features https://rstudio-pubs-static.s3.amazonaws.com/594440_b5a14885d559413ab6e57087eddd68e6.html
       # Teilen von df in Daten die Algo sehen darf (X) und das Ergebnis (y)
       x_features = ['duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness',
                     'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                     'valence', 'tempo', 'time_signature', 'name', 'id_artists']
       X = df[x_features]

       y_features = 'popularity'
       y = df[y_features]

       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=True)

       return X_train, X_test, y_train, y_test


def label_encoding(df):
       encoder = preprocessing.LabelEncoder()
       df['name'] = encoder.fit_transform(df['name'])

       df['id_artists'] = encoder.fit_transform(df['id_artists'])

       return df

def random_forest_classifier(X_train, X_test, y_train, y_test):


       clf = RandomForestClassifier(n_estimators=10)
       clf.fit(X_train, y_train)

       y_pred = clf.predict(X_test)
       mean_absolute_error(y_pred, y_test)
       # One-Hot encoding for categorical data
       print(mean_absolute_error)





df = read_data_from_csv()
X_train, X_test, y_train, y_test = data_preprocessing(df)
result = random_forest_classifier(X_train, X_test, y_train, y_test)
