import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import normalize
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from scipy import stats






# path to json file that stores MFCCs and genre labels for each processed segment
FILE = "data_204.json"

def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)


    mfcc = np.array(data["mfcc"])
    ae = np.expand_dims(normalize(np.array(data["ae"]), axis=1), axis=2)
    rms = np.expand_dims(normalize(np.array(data["rms"]), axis=1), axis=2)
    zero_crossing_rate = np.expand_dims(normalize(np.array(data["zero_crossing_rate"]), axis=1), axis=2)
    spectral_centroid = np.expand_dims(normalize(np.array(data["spectral_centroid"]), axis=1), axis=2)
    spectral_bandwidth = np.expand_dims(normalize(np.array(data["spectral_bandwidth"]), axis=1), axis=2)
  
    
    X = np.concatenate([ae, rms, mfcc], axis = 2)
    print(mfcc.shape)
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return  X, y

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


if __name__ == "__main__":

    # load data
    X, y = load_data(FILE)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # build network topology
    NN = keras.Sequential([

     # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        # 2nd dense layer
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.1),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.1),

        # output layer
        keras.layers.Dense(10, activation='softmax')
        
    ])
    parameters = {'n_estimators': 1000, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 10, 'bootstrap': False}
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators=330, min_samples_split=2, min_samples_leaf=4, max_features='sqrt', max_depth=47, bootstrap=False)
    svc = SVC(C=78.63, gamma=.53, kernel="poly")

    lr = LogisticRegression()

    X = np.reshape(X,(X.shape[0], X.shape[1]*X.shape[2]))

    val_scores = []
    print(cross_val_score(rf, X,y, cv=5))
    print(cross_val_score(svc, X,y, cv=5))


    val_scores = [[0.6097561, 0.65853659, 0.6774193548387096, 0.58536585, 0.5645161290322581],[0.43902439, 0.65853659, 0.5890322580645161, 0.63414634, 0.55]]

    # names = ["Random forest", "Support vector"]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.boxplot(val_scores)
    # ax.set_xticklabels(names)
    # plt.show()


    X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
    X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
    
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    predictions_train = rf.predict(X_train)

    print("train", accuracy_score(y_train ,predictions_train))
    print("test", accuracy_score(y_test ,predictions))

    svc.fit(X_train, y_train)
    predictions = svc.predict(X_test)
    predictions_train = svc.predict(X_train)
    print("train", accuracy_score(y_train ,predictions_train))
    print("test", accuracy_score(y_test ,predictions))
    print(y)

    # lr.fit(X_train, y_train)
    # predictions = lr.predict(X_test)
    # predictions_train = lr.predict(X_train)
    # print("train", accuracy_score(y_train ,predictions_train))
    # print("test", accuracy_score(y_test ,predictions))
    
    # # Look at parameters used by our current forest
    # print('Parameters currently in use:\n')
    # pprint(rf.get_params())














    def hyper():
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 100)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 100)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

        pprint(random_grid)
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 200, cv = 3)
        # Fit the random search model
        rf_random.fit(X_train, y_train)

        print(rf_random.best_params_)

        rand_list = {"C": stats.uniform(0.1, 100),
                "gamma": stats.uniform(0.001, 1),
                'kernel': ['rbf', 'poly', 'sigmoid']}

        pprint(rand_list)

        svc_random = RandomizedSearchCV(estimator = svc, param_distributions = rand_list, n_iter = 200, cv = 3)
        # Fit the random search model
        svc_random.fit(X_train, y_train)

        print(svc_random.best_params_)




  