import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'C:\Users\deepa\PycharmProjects\MOVIE_RATING_PREDICTION\Data\IMDb Movies India.csv', encoding='ISO-8859-1')
#print(df.head())

print(df.info())

#print(df.isnull().sum())