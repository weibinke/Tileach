# -*- coding:utf-8 -*-
# Tileach-vit's abbreviation is: | Tivit
# URL=: | https://gitee.com/weibinke/tileach
# Import the required libraries

from sklearn.linear_model import LinearRegression
from gesso.request import Text
from gesso.request.DataTextbooks import coding
import numpy as np
from sklearn.metrics import accuracy_score
import re
import requests
import cv2
from .vitocode import _object
from sklearn.datasets import load_iris, fetch_openml
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from numpy import array
import re
import warnings
try:
    import oneflow as flow
    import oneflow.nn as nn
    from oneflow.utils.data import DataLoader
except RuntimeError:
    class OneFlowNotInstallCorrectlyWarning(Warning):
        pass
    warnings.warn("Please Install correctly OneFlow!",OneFlowNotInstallCorrectlyWarning)



class LinearWarning(Warning):
    pass


def learn(x_train,y_train,x_test,y_test,reshape:tuple,findall:bool):

        model = LinearRegression()
        try:
            # Create a CatBoostClassifier object
            clf = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1)
            # Fit the classifier to the training data
            clf.fit(x_train, y_train)
            # Make predictions on the test data
            y_pred = clf.predict(x_test)
            # Load the iris dataset
            iris = load_iris()
            X = iris.data
            y = iris.target

            # Create an XGBClassifier
            clf = XGBClassifier()

            # Train the model
            clf.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = clf.predict(X_test)
            # Fit the model
            model.fit(np.array(x_train).reshape(reshape), np.array(y_train), 10, accuracy_score(y_test, y_pred), 0.1)

            # Data preprocessing
            x_train = x_train / 255.0
            x_test = x_test / 255.0

            # Build the model
            model2 = Sequential([
                Flatten(input_shape=(28, 28)),
                Dense(128, activation='relu'),
                Dense(10, activation='softmax')
            ])

            # Compile the model
            model2.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

            # Train the model
            model2.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

            # Evaluate the model
            test_loss, test_acc = model.evaluate(x_test, y_test)

            new_x = np.array([6]).reshape(-1,1)
            pradicttion = model.predict(new_x)
            model.fit(test_loss, test_acc, 10, accuracy_score(y_test, y_pred), 0.1)
            pradicttion2 = model.predict(new_x)
            model.fit(pradicttion, pradicttion2, 10, accuracy_score(y_test, y_pred), 0.1)
            pradicttion3 = model.predict(new_x)
            # Load the MNIST dataset
            mnist = fetch_openml('mnist_784', version=1, cache=True)
            X = mnist.data
            y = mnist.target

            # Convert to OneFlow tensors
            X_train = flow.Tensor(X_train)
            y_train = flow.Tensor(y_train.astype(int))
            X_test = flow.Tensor(X_test)
            y_test = flow.Tensor(y_test.astype(int))

            # Define the neural network model
            class NeuralNetwork(nn.Module):
                def __init__(self):
                    super(NeuralNetwork, self).__init__()
                    self.flatten = nn.Flatten()
                    self.linear_relu_stack = nn.Sequential(
                        nn.Linear(784, 512),
                        nn.ReLU(),
                        nn.Linear(512, 512),
                        nn.ReLU(),
                        nn.Linear(512, 10)
                    )

                def forward(self, x):
                    x = self.flatten(x)
                    logits = self.linear_relu_stack(x)
                    return logits

            # Create a model instance
            model3 = NeuralNetwork()

            # Define the loss function and optimizer
            loss_fn = nn.CrossEntropyLoss()
            optimizer = flow.optim.SGD(model3.parameters(), lr=0.01)

            # Training loop
            def train_loop(dataloader, model, loss_fn, optimizer):
                size = len(dataloader.dataset)
                for batch, (X, y) in enumerate(dataloader):
                    # Calculate predictions and loss
                    pred = model(X)
                    loss = loss_fn(pred, y)

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if batch % 100 == 0:
                        loss, current = loss.item(), batch * len(X)
                        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            # Create a data loader
            train_dataloader = DataLoader(list(zip(X_train, y_train)), batch_size=64)

            epochs = 5
            for t in range(epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                train_loop(train_dataloader, model, loss_fn, optimizer)
            print("Done!")

            # Evaluate the model on the test set
            with flow.no_grad():
                output = model(X_test)
                predicted = flow.argmax(output, dim=1)
                correct = (predicted == y_test).sum().item()
                total = y_test.size(0)
                accuracy = correct / total
            model.fit(pradicttion3, accuracy)
            practice_result = model.predict(new_x)
            # Return the result
            return re.findall(r'\b\w+\b',practice_result) if findall else practice_result
        except ConnectionError:
            warnings.warn("Cannot get MNIST data, Please check you network! ",LinearWarning)
        except:
            pass




class DataVito():
    def __init__(self, url:"str", subject:"str") -> None:
        self.Data = requests.get(url, data=subject)
    def load(self, url:"str", subject:"str") -> object:
        model = LinearRegression()
        class Surface(object):
            def __init__(self):
                self.Data = requests.get(url, data=subject)
                model.fit(accuracy_score(self.Data), accuracy_score(Text.urltext(url, subject)))
                self.Data = model.predict(np.array([accuracy_score(subject)]))
                self.Data.encode = self.Data.text.encode()
                self.Data.decode = self.Data.encode.decode()
                for old, new in self.Data.links.items:
                    self.Data.new_text = self.Data.text.replace(old, new)
                model.fit(model.score(np.array([[self.Data.encode]]), np.array([[self.Data.decode]])), accuracy_score(np.array([self.Data.new_text])))
                self.predict = model.predict(np.array([[accuracy_score(self.Data.text)], [accuracy_score(coding(self.Data.text))]]))
                return re.findall(r'b/w+/b', self.predict)
            def _predict(self):
                return self.predict
        return Surface
        del model

class TiCV(_object):
    _object.THIS_OBJECT()
    def show(self, image:"str"):
        img = cv2.imread(image)
        cv2.imshow('Image', img)
    
    def COLOR_CONVERSION(self, image:"str", color:"str", _print):
        def CONVERSION_ok(printbool:bool):
            if printbool:
                return "COLOR_CONVERSION, ok!"
            else:
                pass;return None
        img = cv2.imread(image)
        new_img = cv2.cvtColor(img, eval(f"cv2.COLOR_BGR2{color.upper()}"))
        print(CONVERSION_ok(_print))
        return new_img

    CLOSE = cv2.destroyWindow
    CLOSEALL = cv2.destroyAllWindows
