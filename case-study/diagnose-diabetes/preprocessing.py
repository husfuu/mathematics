import numpy as np
import pandas as pd

df = pd.read_csv('./data/pima-indian-diabetest-database.csv', na_values={'Glucose':0, 'BloodPressure':0, 'SkinThickness':0, 'Insulin':0, 'BMI':0})

# missing data