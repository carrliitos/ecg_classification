import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

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

def train_model(x_train, y_train):
    """
    Train a DNNClassifier model.

    Parameters:
    - x_train (numpy.ndarray): Training features.
    - y_train (numpy.ndarray): Training labels.

    Returns:
    - estimator (tf.estimator.Estimator): Trained DNNClassifier model.
    """
    feature_columns = [tf.feature_column.numeric_column('beat', shape=[187])]

    estimator = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[256, 64, 16],
        optimizer=tf.train.AdamOptimizer(1e-4),
        n_classes=2,
        dropout=0.1,
        model_dir='../data/external/mitdb'
    )

    input_fn_train = tf.estimator.inputs.numpy_input_fn(
        x={'beat': x_train},
        y=y_train,
        num_epochs=None,
        batch_size=50,
        shuffle=True
    )

    estimator.train(input_fn=input_fn_train, steps=400000)
    return estimator

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

    accuracy_score = estimator.evaluate(input_fn=input_fn_validate)
    print('\nTest Accuracy: {0:f}%\n'.format(accuracy_score['accuracy'] * 100))

def test_model(estimator, x_test, y_test):
    """
    Test the trained model.

    Parameters:
    - estimator (tf.estimator.Estimator): Trained DNNClassifier model.
    - x_test (numpy.ndarray): Test features.
    - y_test (numpy.ndarray): Test labels.
    """
    input_fn_test = tf.estimator.inputs.numpy_input_fn(
        x={'beat': x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False
    )

    predictions = estimator.predict(input_fn=input_fn_test)

    totvals = 0
    totwrong = 0

    for prediction, expected in zip(predictions, y_test):
        totvals = totvals + 1
        catpred = prediction['class_ids'][0]
        certainty = prediction['probabilities'][catpred] * 100
        if expected != catpred:
            totwrong = totwrong + 1
            print('Real: ', expected, ', pred: ', catpred, ', cert: ', certainty)

    print('Accuracy: ', ((totvals - totwrong) * 100.0 / totvals))
    print('Wrong: ', totwrong, ' out of ', totvals)

def export_model(estimator):
    """
    Export the trained model for serving predictions.

    Parameters:
    - estimator (tf.estimator.Estimator): Trained DNNClassifier model.
    """
    feature_placeholders = {'beat': tf.placeholder(dtype=tf.float32, shape=(187,))}
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_placeholders)
    export_dir = estimator.export_savedmodel('ecg_serving', serving_input_receiver_fn, strip_default_attrs=True)

if __name__ == "__main__":
    # Load the data
    x_train, y_train = load_data("../data/interim/mitdb/train.csv")
    x_validate, y_validate = load_data("../data/interim/mitdb/validate.csv")
    x_test, y_test = load_data("../data/interim/mitdb/test.csv")

    # Visualize Data
    visualize_data(x_train, y_train)

    # Train the model
    trained_estimator = train_model(x_train, y_train)

    # Evaluate the model
    evaluate_model(trained_estimator, x_validate, y_validate)

    # Test the model
    test_model(trained_estimator, x_test, y_test)

    # Export the model
    export_model(trained_estimator)
