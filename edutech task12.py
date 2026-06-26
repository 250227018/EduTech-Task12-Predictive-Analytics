import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

# 1. Load dataset
df = pd.read_csv("C:/Users/hp/OneDrive/Desktop/datasets/Marketing_Data.csv")
print(df.head())
print(df.shape)

# 2. Features aur Target define karo
X = df[['youtube', 'facebook', 'newspaper']]
y = df['sales']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# 4. Linear Regression Model banao
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# 7. Actual vs Predicted Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2)
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.savefig('actual_vs_predicted.png')
plt.show()

# 8. Feature Importance Plot
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
plt.figure(figsize=(6, 4))
sns.barplot(x='Feature', y='Coefficient', data=coefficients)
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
plt.show()

# 9. Prediction Report save karo
report = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
report.to_csv('prediction_report.csv', index=False)
print("\nPrediction report saved!")

# 10. Model save karo
joblib.dump(model, 'linear_regression_model.pkl')
print("Model saved as linear_regression_model.pkl")