# Importing project dependencies
import numpy as np
import pandas as pd

# Data preprocessing
# Specify data types for potentially problematic columns
dtype_dict = {'Open': float, 'VWAP': float, 'Deliverable Volume': float, '%Deliverble': float, 'Trades': float}
data = pd.read_csv('NIFTY50_all.csv', sep=',', header=None, dtype=dtype_dict)

column_names = ['Date', 'Symbol', 'Series', 'Prev_Close', 'Open', 'High', 'Low', 'Last',
                'Close', 'VWAP', 'Volume', 'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble']
data.columns = column_names

# Replace invalid values with median for columns that need it
data['VWAP'].replace('?', np.NaN, inplace=True)
data['VWAP'].replace(np.NaN, data['VWAP'].median(), inplace=True)
data['High'].replace('?', np.NaN, inplace=True)
data['High'].replace(np.NaN, data['High'].median(), inplace=True)
data['Deliverable Volume'].replace(np.NaN, data['Deliverable Volume'].median(), inplace=True)
data['%Deliverble'].replace(np.NaN, data['%Deliverble'].median(), inplace=True)
data['Trades'].replace(np.NaN, data['Trades'].median(), inplace=True)


# Convert 'High' and 'VWAP' columns to float for computations
data['High'] = data['High'].astype('float64')
data['VWAP'] = data['VWAP'].astype('float64')

# Prepare features and target arrays
X = np.array(data.drop(['Date', 'Symbol', 'Series', 'Close'], axis=1))
y = np.array(data['Close'])

# Custom train-test split function
def train_test_split(features, target, test_size=0.2):
    total_num_of_rows = features.shape[0]
    no_of_test_rows = int(total_num_of_rows * test_size)
    rand_row_num = np.random.randint(0, total_num_of_rows, no_of_test_rows)
    features_test = np.array([features[i] for i in rand_row_num])
    features_train = np.delete(features, rand_row_num, axis=0)
    target_test = np.array([target[i] for i in rand_row_num])
    target_train = np.delete(target, rand_row_num, axis=0)
    return features_train, features_test, target_train, target_test

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# StandardScaler function to normalize data (mean=0, std=1)
def StandardScaler(arr):
    arr1 = arr
    try:
        n = arr1.shape[1]
        for i in range(n):
            temp_arr = arr1[:, i]
            mean = temp_arr.mean()
            std = temp_arr.std()
            arr1[:, i] = (arr1[:, i] - mean) / std
    except IndexError:
        mean = arr.mean()
        std = arr.std()
        arr1 = (arr1 - mean) / std
    return arr1

# Normalize training and testing features and targets
X_train = StandardScaler(X_train)
y_train = StandardScaler(y_train)
y_test = StandardScaler(y_test)
X_test = StandardScaler(X_test)

print('X_train (Mean):', X_train[:, 0].mean())
print('X_train (Standard_deviation):', X_train[:, 0].std())
print('X_test (Mean):', X_test[:, 0].mean())
print('X_test (Standard_deviation):', X_test[:, 0].std())
print('y_train (Mean):', y_train.mean())
print('y_train (Standard_deviation):', y_train.std())
print('y_test (Mean):', y_test.mean())
print('y_test (Standard_deviation):', y_test.std())

# Linear Regression Model class definition
class LinearRegressionModel:
    def __init__(self, no_of_features, epochs, no_of_targets=1, learning_rate=0.1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.no_of_weights = no_of_features  # number of weights
        self.weights = np.random.rand(no_of_features)
        self.bias = np.random.rand(no_of_targets)
        self.final_param = {}

    def fit(self, X_train, X_test, y_train, y_test):
        def LinearRegression(X_train):
            """Calculate predicted values"""
            predictions = self.weights.dot(np.transpose(X_train)) + self.bias
            return predictions

        def MSE(predictions):
            """Compute Mean Squared Error"""
            MSE = np.sum((predictions - y_train) ** 2 / len(y_train))
            return MSE

        predictions = LinearRegression(X_train)
        MSE_VAL = MSE(predictions)
        print('MSE_VALUE at random is ', MSE_VAL)

        def gradient():
            """Calculate gradients for weights and bias"""
            n = len(X_train)
            predictions = LinearRegression(X_train)
            loss = y_train - predictions
            grad_bias = np.array([-2 / n * np.sum(loss)])
            grad_weights = np.ones(self.no_of_weights)
            for i in range(self.no_of_weights):
                featurecol = X_train[:, i]
                grad_weights[i] = -2 / n * np.sum(loss * featurecol)
            return grad_weights, grad_bias

        def stachscalerModified():
            """Perform gradient descent optimization"""
            MSE_list = []
            for i in range(self.epochs):
                grad_weights, grad_bias = gradient()
                self.weights -= self.learning_rate * grad_weights
                self.bias -= self.learning_rate * grad_bias
                new_predictions = LinearRegression(X_train)
                MSE_new = MSE(new_predictions)
                MSE_list.append(MSE_new)
            return_dict = {
                'weights': self.weights,
                'bias': self.bias[0],
                'MSE new value': MSE_new,
                'MSE_list': MSE_list
            }
            return return_dict

        self.final_param = stachscalerModified()
        print('Final value of MSE ', self.final_param['MSE new value'])

    def predict(self, featureset):
        """Predict the target value"""
        prediction = np.sum(self.final_param['weights'] * featureset) + self.final_param['bias']
        return prediction

    def r2_score(self, y_test):
        """Calculate R-squared accuracy"""
        predicted = []
        for i in range(len(y_test)):
            predictions = self.predict(X_test[i])
            predicted.append(predictions)
        predicted = np.array(predicted)
        return (1 - np.sum((y_test - predicted) ** 2) / np.sum((y_test - y_test.mean()) ** 2)) * 100

# Load the pickled trained model
import pickle
pickle_in = open('Bestfitmodel.pickle', 'rb')
model = pickle.load(pickle_in)

# Print the model accuracy on test data
print(model.r2_score(y_test))
