import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # Used for model serialization

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - x_data (numpy.ndarray): Input features.
    - y_data (numpy.ndarray): Labels.
    """
    df = pd.read_csv(file_path, header=None)
    x_data = df.values[:, :-1]
    y_data = df.values[:, -1].astype(int)
    return x_data, y_data

def visualize_data(x_train, y_train):
    """
    Visualize one Normal and one Abnormal heartbeat.

    Parameters:
    - x_train (numpy.ndarray): Training data.
    - y_train (numpy.ndarray): Training labels.
    """
    C0 = np.argwhere(y_train == 0).flatten()
    C1 = np.argwhere(y_train == 1).flatten()

    x = np.arange(0, 187) * 8 / 1000.0

    plt.figure(figsize=(20, 12))
    plt.plot(x, x_train[C0, :][0], label="Normal")
    plt.plot(x, x_train[C1, :][0], label="Abnormal")
    plt.legend()
    plt.title("1-beat ECG for every category", fontsize=20)
    plt.ylabel("Normalized Amplitude (0 - 1)", fontsize=15)
    plt.xlabel("Time (ms)", fontsize=15)
    plt.savefig("../reports/figures/one-beat-ecg-for-each-cats.png")

def train_model_sklearn(x_train, y_train):
    """
    Train an MLPClassifier model using scikit-learn.

    Parameters:
    - x_train (numpy.ndarray): Training features.
    - y_train (numpy.ndarray): Training labels.

    Returns:
    - mlp_classifier (MLPClassifier): Trained MLPClassifier model.
    """
    # Create an MLPClassifier
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(256, 64, 16), max_iter=400, random_state=42)

    # Train the classifier
    mlp_classifier.fit(x_train, y_train)

    return mlp_classifier

def evaluate_model(mlp_classifier, x_validate, y_validate):
    """
    Evaluate the trained model.

    Parameters:
    - mlp_classifier (MLPClassifier): Trained MLPClassifier model.
    - x_validate (numpy.ndarray): Validation features.
    - y_validate (numpy.ndarray): Validation labels.
    """
    y_pred = mlp_classifier.predict(x_validate)
    accuracy = accuracy_score(y_validate, y_pred)
    print('\nTest Accuracy: {:.2%}\n'.format(accuracy * 100))

def test_model(mlp_classifier, x_test, y_test):
    """
    Test the trained model.

    Parameters:
    - mlp_classifier (MLPClassifier): Trained MLPClassifier model.
    - x_test (numpy.ndarray): Test features.
    - y_test (numpy.ndarray): Test labels.
    """
    y_pred = mlp_classifier.predict(x_test)

    totvals = len(y_test)
    totwrong = np.sum(y_test != y_pred)

    print('Accuracy: {:.2%}'.format((totvals - totwrong) / totvals))
    print('Wrong: {} out of {}'.format(totwrong, totvals))

def export_model(mlp_classifier, save_path='../data/external/mitdb/ecg_serving_model.joblib'):
    """
    Export the trained model for serving predictions.

    Parameters:
    - mlp_classifier (MLPClassifier): Trained MLPClassifier model.
    - save_path (str): Path to save the exported model.
    """
    joblib.dump(mlp_classifier, save_path)

if __name__ == "__main__":
    # Load the data
    x_train, y_train = load_data("../data/interim/mitdb/train.csv")
    x_validate, y_validate = load_data("../data/interim/mitdb/validate.csv")
    x_test, y_test = load_data("../data/interim/mitdb/test.csv")

    # Visualize Data
    visualize_data(x_train, y_train)

    # Train the model
    trained_mlp_model = train_model_sklearn(x_train, y_train)

    # Evaluate the model
    evaluate_model(trained_mlp_model, x_validate, y_validate)

    # Test the model
    test_model(trained_mlp_model, x_test, y_test)

    # Export the model
    export_model(trained_mlp_model)
