import pandas as pd

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

X = cleaned.drop(columns=['Exam_Score'])
y = cleaned['Exam_Score']

# Convert all object-type columns in X into dummies
X_numeric = pd.get_dummies(X, drop_first=True,dtype=int)

print('After cleaning, shape:', X_numeric.shape)
print('Sample columns:', list(X_numeric.columns)[:10])