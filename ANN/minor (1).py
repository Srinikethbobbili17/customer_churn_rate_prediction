import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#importing dataset
df=pd.read_csv("/content/Churn_Modelling (AI) (1).csv")
df.apply(pd.to_numeric,errors='ignore')
df=df.dropna()
df=df.drop(columns=['Surname'],axis=1)

#handling null values
df.isnull().sum()

#handling duplicates
df.duplicated().sum()

#target values
df["Exited"].value_counts()

#label encoder

df.columns

for i in df.columns:
  print(f"{i} : {df[i].nunique()}")

cat_cols=['Geography','Gender','NumOfProduct','HasCrCard','IsActiveMember']
cont_cols=['CreditScore','Age','Tenure','Balance','EstimatedSalary']
target_col=['Exited']
id_col=['CustomerId','Surname']

for i in cont_cols:
  plt.boxplot(df[i])
  plt.title(f"boxplot{i}")
  plt.show()

col_with_outli=['CreditScore','Age']

def remove_outlier(x):
  x=x.clip(lower=x.quantile(0.01))
  x=x.clip(upper=x.quantile(0.99))
  return x

df[col_with_outli]=df[col_with_outli].apply(remove_outlier)

df=df.drop(columns=['Geography','Gender','CustomerId','RowNumber'])

df.dtypes

x=df.drop(columns=['Exited'])
y=df['Exited']
print(x.shape)
print(y.shape)

x_train,x_test,y_train,y_test=sk.model_selection.train_test_split(x,y,test_size=0.3,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

df

"""# ANN model"""

corr=df.corr()
corr
plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True)

df.corr()['Exited'].sort_values(ascending=False)

confusion_metric=pd.crosstab(df['Exited'],df['EstimatedSalary'])
confusion_metric

plt.figure(figsize=(10,10))
sns.heatmap(confusion_metric,annot=True)

ml=Sequential()
ml.add(Dense(52,activation='relu',input_dim=8))
ml.add(Dense(42,activation='relu'))
ml.add(Dense(21,activation='relu'))
ml.add(Dense(10,activation='relu'))
ml.add(Dense(1,activation='sigmoid'))
ml.summary()
#ml.add(Dense(42,activation='Leaky ReLU',input_dim=10))
#ml.add(Dense(21,activation='Leaky ReLU'))
#ml.add(Dense(10,activation='Leaky ReLU'))
#ml.add(Dense(1,activation='sigmoid'))
#ml.summary()

print(x_train.shape)
print(y_train.shape)

h1=ml.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

hl=ml.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test))

plt.plot(hl.history['accuracy'])
plt.plot(hl.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

plt.plot(hl.history['loss'])
plt.plot(hl.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')

from tensorflow.keras.callbacks import EarlyStopping

h=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)