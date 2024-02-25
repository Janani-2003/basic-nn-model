# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
A neural network is a computer program inspired by how our brains work. It's used to solve problems by finding patterns in data. Imagine a network of interconnected virtual "neurons." Each neuron takes in information, processes it, and passes it along. A Neural Network Regression Model is a type of machine learning algorithm that is designed to predict continuous numeric values based on input data. It utilizes layers of interconnected nodes, or neurons, to learn complex patterns in the data. The architecture typically consists of an input layer, one or more hidden layers with activation functions, and an output layer that produces the regression predictions. This model can capture intricate relationships within data, making it suitable for tasks such as predicting prices, quantities, or any other continuous numerical outputs.


## Neural Network Model

![nn](https://github.com/Janani-2003/basic-nn-model/assets/94288340/1a2ce2f2-d953-45d1-9201-6b057cb1c852)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: JANANI R
### Register Number: 212221230039
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet=gc.open("Jananii").sheet1
data=worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'input':'float'})
dataset1 = dataset1.astype({'output':'float'})
dataset1.head()
X = dataset1[['input']].values
y = dataset1[['output']].values
X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai_brain = Sequential([
    Dense(6,activation = 'relu'),
    Dense(6,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(X_train1,y_train,epochs = 80)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)

```
## Dataset Information

![jg](https://github.com/Janani-2003/basic-nn-model/assets/94288340/dc1d4b18-e8ab-4dfb-9f16-725fb5b6bf30)


## OUTPUT

### Training Loss Vs Iteration Plot

![jit](https://github.com/Janani-2003/basic-nn-model/assets/94288340/27b84a34-1702-4fb1-b818-fa75a0ed799d)


### Test Data Root Mean Squared Error

![jtest](https://github.com/Janani-2003/basic-nn-model/assets/94288340/2a1a1366-035d-42db-b02e-8efcfcf4eaf4)


### New Sample Data Prediction

![jnew](https://github.com/Janani-2003/basic-nn-model/assets/94288340/e6a97ae6-05bd-4442-8d55-8b5b092752dc)


## RESULT

A neural network regression model for the given dataset has been developed Sucessfully.


