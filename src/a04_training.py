import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import context as ctxt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
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

def visualize_data(x_train, y_train, directory):
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
    plt.savefig(f"{directory}/reports/figures/one-beat-ecg-for-each-cats.png")

def train_model_sklearn(x_train, y_train):
    """
    Train an MLPClassifier model using scikit-learn.

    Parameters:
    - x_train (numpy.ndarray): Training features.
    - y_train (numpy.ndarray): Training labels.

    Returns:
    - mlp_classifier (MLPClassifier): Trained MLPClassifier model.
    """
    print("Training Phase")

    # Create an MLPClassifier
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(256, 64, 16), max_iter=400, random_state=42)

    # Train the classifier
    mlp_classifier.fit(x_train, y_train)

    return mlp_classifier

def evaluate_model(classifier, x_validate, y_validate, directory):
    """
    Evaluate the trained model using scikit-learn metrics.

    Parameters:
    - classifier: Trained classifier model (e.g., MLPClassifier).
    - x_validate (numpy.ndarray): Validation features.
    - y_validate (numpy.ndarray): Validation labels.
    """

    print("Validation Phase")

    # Make predictions on the validation set
    y_pred = classifier.predict(x_validate)

    # Evaluate accuracy
    accuracy = accuracy_score(y_validate, y_pred)
    print('\nTest Accuracy: {:.2f}%\n'.format(accuracy * 100))

    # Confusion Matrix
    cm = confusion_matrix(y_validate, y_pred)
    print('Confusion Matrix:')
    print(cm)

    # Precision, Recall, F1 Score
    precision = precision_score(y_validate, y_pred)
    recall = recall_score(y_validate, y_pred)
    f1 = f1_score(y_validate, y_pred)

    print('Precision: {:.2f}'.format(precision))
    print('Recall: {:.2f}'.format(recall))
    print('F1 Score: {:.2f}'.format(f1))

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_validate, y_pred)
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(f"{directory}/reports/figures/precision_recall_curve.png")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_validate, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"{directory}/reports/figures/roc_curve.png")

def test_model(classifier, x_test, y_test, directory):
    """
    Test the trained model and print additional evaluation metrics.

    Parameters:
    - classifier: Trained classifier model (e.g., MLPClassifier).
    - x_test (numpy.ndarray): Test features.
    - y_test (numpy.ndarray): Test labels.
    """
    print("Testing Phase")

    # Make predictions on the test set
    y_pred = classifier.predict(x_test)

    # Additional evaluation metrics
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print('\nConfusion Matrix:')
    print(cm)

    # Precision, Recall, F1 Score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print('\nPrecision: {:.2f}'.format(precision))
    print('Recall: {:.2f}'.format(recall))
    print('F1 Score: {:.2f}'.format(f1))

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(f"{directory}/reports/figures/test_precision_recall_curve.png")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"{directory}/reports/figures/test_roc_curve.png")

def export_model(mlp_classifier, directory):
    """
    Export the trained model for serving predictions.

    Parameters:
    - mlp_classifier (MLPClassifier): Trained MLPClassifier model.
    - save_path (str): Path to save the exported model.
    """
    save_path=f'{directory}/models/mitdb/ecg_serving_model.joblib'
    joblib.dump(mlp_classifier, save_path)

def main():
    directory = ctxt.get_context(os.path.abspath(__file__))
    train_csv_path = f"{directory}/data/interim/mitdb/train.csv"
    validate_csv_path = f"{directory}/data/interim/mitdb/validate.csv"
    test_csv_path = f"{directory}/data/interim/mitdb/test.csv"

    # Load the data
    x_train, y_train = load_data(train_csv_path)
    x_validate, y_validate = load_data(validate_csv_path)
    x_test, y_test = load_data(test_csv_path)

    # Visualize Data
    visualize_data(x_train, y_train, directory)

    # Train the model
    trained_mlp_model = train_model_sklearn(x_train, y_train)

    # Evaluate the model
    evaluate_model(trained_mlp_model, x_validate, y_validate, directory)

    # Test the model
    test_model(trained_mlp_model, x_test, y_test, directory)

    # Export the model
    export_model(trained_mlp_model, directory)

if __name__ == "__main__":
    main()
