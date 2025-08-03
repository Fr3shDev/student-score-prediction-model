
## Features Used

- Hours Studied
- Previous Scores
- Attendance
- Sleep Hours

## How It Works

1. Download the student performance data from kaggle and load it from `data/StudentPerformanceFactors.csv`.
2. Cleans the data and handles missing values.
3. Encodes categorical variables.
4. Selects relevant features for prediction.
5. Splits the data into training and test sets.
6. Trains a linear regression model.
7. Evaluates model performance (Mean Absolute Error, R² score).
8. Visualizes predicted vs actual exam scores.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib

Install dependencies with:

```sh
pip install pandas scikit-learn matplotlib