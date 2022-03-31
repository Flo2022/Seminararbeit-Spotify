import pandas as pd
import numpy as nd
import sklearn

# import matplotlib.pyplot as plt  # To visualize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error



def read_data_from_csv():
       # Datensatz von https://www.kaggle.com/lehaknarnauli/spotify-datasets?select=tracks.csv
       df = pd.read_csv("./tracks.csv")
       return df

df = read_data_from_csv()
