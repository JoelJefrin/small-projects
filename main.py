import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('train.csv')
df.head()

df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath']

features = ['GrLivArea', 'BedroomAbvGr', 'TotalBath']
X = df[features]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

sample_house = [[2000, 3, 2 + 0.5 * 1]]  # 2.5 total baths
predicted_price = model.predict(sample_house)
print(f"Predicted Price: ${predicted_price[0]:,.2f}")


