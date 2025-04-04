#importing the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data collection and pre-processing
file_path="ML Rock or mine\\Data.csv";
sonar_data=pd.read_csv(file_path,header=None)
#print(sonar_data.head)
#print(sonar_data.value_counts()) | m-->mine r-->rock|
s=sonar_data.groupby(60).mean
print(s)
