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
x=sonar_data.drop(columns=60, axis=1)
y=sonar_data[60]

#train test split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.1,stratify=y,random_state=1)
print(x.shape)

