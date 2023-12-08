import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_curve, 
    auc, 
    precision_recall_curve
)
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

def evaluate_model(estimator, x_validate, y_validate):
    """
    Evaluate the trained model.

    Parameters:
    - estimator (tf.estimator.Estimator): Trained DNNClassifier model.
    - x_validate (numpy.ndarray): Validation features.
    - y_validate (numpy.ndarray): Validation labels.
    """
    input_fn_validate = tf.estimator.inputs.numpy_input_fn(
        x={'beat': x_validate},
        y=y_validate,
        num_epochs=1,
        shuffle=False
    )

    predictions = estimator.predict(input_fn=input_fn_validate)
    predicted_labels = [prediction['class_ids'][0] for prediction in predictions]

    # Confusion Matrix
    cm = confusion_matrix(y_validate, predicted_labels)
    print('Confusion Matrix:')
    print(cm)

    # Accuracy
    accuracy = accuracy_score(y_validate, predicted_labels)
    print('\nAccuracy: {:.2f}%'.format(accuracy * 100))

    # Precision
    precision = precision_score(y_validate, predicted_labels)
    print('Precision: {:.2f}'.format(precision))

    # Recall
    recall = recall_score(y_validate, predicted_labels)
    print('Recall: {:.2f}'.format(recall))

    # Specificity
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Specificity: {:.2f}'.format(specificity))

    # F1 Score
    f1 = f1_score(y_validate, predicted_labels)
    print('F1 Score: {:.2f}'.format(f1))

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_validate, predicted_labels)
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig('../reports/figures/precision_recall_curve.png')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_validate, predicted_labels)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('../reports/figures/roc_curve.png')

    # PR vs ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot(recall, precision, color='blue', lw=2, label='PR curve')
    plt.xlabel('False Positive Rate / Recall')
    plt.ylabel('True Positive Rate / Precision')
    plt.title('PR vs ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('../reports/figures/pr_vs_roc_curve.png')

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

def export_model(mlp_classifier, save_path='../models/mitdb/ecg_serving_model.joblib'):
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
