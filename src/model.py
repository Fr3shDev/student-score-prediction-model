import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

student_data = pd.read_csv('data/StudentPerformanceFactors.csv')
print(student_data.head())
print('Shape:', student_data.shape)
print('\nColumn types:\n', student_data.dtypes)

print('\nMissing values per column:')
print(student_data.isna().sum())

print('\nNumeric summary statistics:')
print(student_data.describe())

# Drop any empty columns
cleaned = student_data.dropna(subset=['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home'])

# Features
X = cleaned.drop(columns=['Exam_Score'])
# Prediction to be made
y = cleaned['Exam_Score']

# Convert all object-type columns in X into dummies
X_numeric = pd.get_dummies(X, drop_first=True,dtype=int)

print('After cleaning, shape:', X_numeric.shape)
print('Sample columns:', list(X_numeric.columns)[:10])

# select only the 'Hours_Studied' column
X_simple = X_numeric[['Hours_Studied']]
# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2,random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

# Inspect the learned parameters
print('Slope (coef):', model.coef_[0])
print('Intercept:', model.intercept_)

# Make predictions and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Absolute Error:', mae)
print('R² score:', r2)

# Visualize the results
plt.scatter(X_test, y_test, label='Actual', alpha=0.6)
plt.scatter(X_test, y_pred, label='Predicted', alpha=0.6)
plt.xlabel('Hours_Studied')
plt.ylabel('Exam_Score')
plt.legend()
plt.show()
