import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your energy consumption dataset. This should include features and the target variable (energy consumption).
# For this example, let's assume you have a CSV file with features (e.g., temperature, time) and energy consumption.

data = pd.read_csv('C:/Users/ELCOT\Desktop/AI_Phase3/DAYTON_hourly.csv')
print(data.head())
# Split the data into features and the target variable
X = data[['Temperature', 'Time']]
y = data['EnergyConsumption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize the predictions
plt.scatter(X_test['Temperature'], y_test, color='blue', label='Actual')
plt.scatter(X_test['Temperature'], y_pred, color='red', label='Predicted')
plt.xlabel('Temperature')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()
