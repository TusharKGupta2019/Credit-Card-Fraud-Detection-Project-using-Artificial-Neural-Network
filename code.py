import pandas as pd

df = pd.read_csv('fraudTrain.csv')

df.shape

df.head(3)

df.columns

# 1. Dropping the unwanted columns
df.drop(columns = ['Unnamed: 0','trans_date_trans_time','cc_num','merchant','first','last', 'street','city','zip','lat','long','dob','trans_num','unix_time','merch_lat','merch_long'], inplace = True)

df.head()

# 2. Checking Null values
df.isna().sum()

df.dropna(inplace = True)

df.isna().sum()

# 3. Converting categorical values to numerical
df['category'].unique()

category_df=pd.get_dummies(df['category']).astype(int)

category_df.head()

df.drop(columns= ['category'], inplace = True)



df.head()

df.reset_index(drop=True, inplace=True)


df = pd.concat([df, category_df], axis = 1)

df.head()

df['gender'].unique()

df['gender'].replace({'F' : 1, 'M' : 0}, inplace = True)

df['gender'].unique()

df.head()

df['state'].unique()

state_df=pd.get_dummies(df['state']).astype(int)

state_df.head()

df.reset_index(drop=True, inplace=True)

df = pd.concat([df, state_df], axis = 1)

df.drop(columns = ['state'], inplace = True)

df.head()

df['job'].unique().shape

job_df=pd.get_dummies(df['job']).astype(int)

job_df.head()

df.drop(columns = ['job'], inplace = True)

df.reset_index(drop=True, inplace = True)

df = pd.concat([df, job_df], axis = 1)

df.head()

df.shape

# 4. Dividing the dataset
X = df.drop(columns = ['is_fraud'])
y = df['is_fraud']

X.head()

y.value_counts()

# 5. Handling the imbalanced dataset using SMOTE technique
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')
X,y =smote.fit_resample(X,y)

y.value_counts()

X.shape

# 6. Data Normalization
X.describe()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X[:] = scaler.fit_transform(X)

X.describe()

X.head()

# 7. Train - Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


X_train.shape, X_test.shape

y_train.value_counts()

y_test.value_counts()

# 8. Building the Artificial Neural Network
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    # input layer + hidden layer 1
    keras.layers.Dense(300, input_shape =(546,), activation = 'relu'),
    # hidden layer 2
    keras.layers.Dense(150, activation = 'relu'),
    # output layer
    keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(optimizer = 'adam', 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 10, batch_size=200)

# 9. Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Model Accuracy : {accuracy * 100}')

model.summary()

# 10. To make predictions
pred = model.predict(X_test)

pred[:5]

binary_pred = ((pred > 0.5)).astype(int)

binary_pred[:5]

y_test[:5]

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, binary_pred))

import seaborn as sns
import matplotlib.pyplot as plt

cf = confusion_matrix(y_test, binary_pred, normalize = 'true')
sns.heatmap(cf, annot = True, cmap = 'OrRd')
plt.xlabel('Predictions')
plt.ylabel('Actual')


