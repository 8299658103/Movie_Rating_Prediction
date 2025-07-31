import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r'C:\Users\deepa\PycharmProjects\MOVIE_RATING_PREDICTION\Data\IMDb Movies India.csv', encoding='ISO-8859-1')
print("Initial shape:", df.shape)
df.info()

# Keep essential columns
df = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Rating']]

# Drop rows with missing ratings
df.dropna(subset=['Rating'], inplace=True)

# Fill remaining NaNs with 'Unknown'
df.fillna('Unknown', inplace=True)

print("After cleaning shape:", df.shape)

df['Combined_Actors'] = df['Actor 1'] + ' | ' + df['Actor 2'] + ' | ' + df['Actor 3']
df = df.drop(['Actor 1', 'Actor 2', 'Actor 3'], axis=1)

X = df[['Genre', 'Director', 'Combined_Actors']]
y = df['Rating']

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)
print("Encoded features shape:", X_encoded.shape)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Prediction Line')

indices = np.arange(10)  # first 10 movies
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(indices, y_test[:10], width, label='Actual', color='skyblue')
plt.bar(indices + width, y_pred[:10], width, label='Predicted', color='orange')

plt.xlabel('Movie Index')
plt.ylabel('Rating')
plt.title('Actual vs Predicted Ratings (Sample)')
plt.xticks(indices + width / 2, indices)
plt.legend()
plt.tight_layout()
plt.show()

#keep waiting...
