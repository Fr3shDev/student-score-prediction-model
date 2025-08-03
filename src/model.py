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

# select features
features = ['Hours_Studied', 'Previous_Scores', 'Attendance', 'Sleep_Hours']
X_multi = X_numeric[features]
# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X_multi, y, test_size=0.2,random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

# Inspect the learned parameters
print('Slope (coef):', dict(zip(features, model.coef_)))
print('Intercept:', model.intercept_)

# Make predictions and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Absolute Error:', mae)
print('R² score:', r2)

# Visualize the results
# plot predicted first, in orange
plt.scatter(y_test, y_pred,
            alpha=0.6,
            label='Predicted',
            color='orange',
            marker='o')

# then plot actual, in blue with hollow face
plt.scatter(y_test, y_test,
            alpha=0.6,
            label='Actual',
            edgecolor='blue',
            facecolor='none',
            marker='o')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', linewidth=1)
plt.xlabel('Actual Exam Score')
plt.ylabel('Predicted Exam Score')
plt.title('Predicted vs Actual')
plt.legend()
plt.show()
