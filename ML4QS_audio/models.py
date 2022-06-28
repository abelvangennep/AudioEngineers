import os

from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.models import Sequential
import keras as k
import matplotlib as plt

from LearningAlgorithms import ClassificationAlgorithms
from Evaluation import ClassificationEvaluation

from sklearn.neural_network import MLPClassifier


def trainModel(model, optimizer, batch_size=1, epochs=1):
    model.compile(optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics='accuracy')
    return model.fit(x_train, y_train, validation_data=(x_test, y_test), 
                    epochs=epochs, batch_size=batch_size)

def plotValidate(history):
    print("validation acc", max(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12,6))
    plt.show()




df = pd.read_csv('df_features_4s.csv')
# df = pd.read_csv('df_features_aug_4s.csv')

y = df['label']
X = df.drop(['label', 'Unnamed: 0'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = Sequential([
        k.layers.Dense(512, activation='relu', input_shape=(x_train.shape[1],)),
        k.layers.Dropout(0.2),

        k.layers.Dense(256, activation='relu'),
        k.layers.Dropout(0.2),

        k.layers.Dense(128, activation='relu'),
        k.layers.Dropout(0.2),

        k.layers.Dense(4, activation='softmax'),
])

print(model.summary())
model_history = trainModel(model=model, epochs=100, optimizer='adam')

test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=1)
print(test_loss)
print(test_acc)

print("check")
# # build model with 3 layers: 2 -> 5 -> 1
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
#     tf.keras.layers.Dense(1, activation="sigmoid")
#     ])

# # choose optimiser
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# # compile model
# model.compile(optimizer=optimizer, loss='mse')

# # train model
# model.fit(x_train, y_train, epochs=100)

# # evaluate model on test set
# print("\nEvaluation on the test set:")
# model.evaluate(x_test,  y_test, verbose=2)




# learner = ClassificationAlgorithms()

# class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
#             X_train, y_train,
#             X_test, hidden_layer_sizes=(250, ), max_iter=500,
#             gridsearch=True
#         )
# eval = ClassificationEvaluation()

# print("train acc", eval.accuracy(y_train, class_train_y))
# print("test acc", eval.accuracy(y_test, class_test_y))
# print(class_test_y)