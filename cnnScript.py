# CODE HELPER 4
import argparse
import os
import pickle
import subprocess
from sklearn.metrics import accuracy_score
subprocess.run(["pip", "install", "Werkzeug==2.0.3"])
subprocess.run(["pip", "install", "tensorflow==2.4"])
import tensorflow as tf
from tensorflow import keras
import numpy as np


def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    # We don't use these but I left them in as a useful template for future development
    parser.add_argument("--copy_X", type=bool, default=True)
    parser.add_argument("--fit_intercept", type=bool, default=True)
    parser.add_argument("--normalize", type=bool, default=False)

    # data directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_known_args()


def load_dataset(path):
    """
    Load entire dataset.
    """
    # Find all files with a pickle ext but we only load the first one in this sample:
    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".cnn")]

    if len(files) == 0:
        raise ValueError("Invalid # of files in dir: {}".format(path))

    [X, y] = pickle.load(open(files[0], 'rb'))

    return X, y

def model_fn(model_dir):
    """
    Load the model for inference
    """
    loaded_model = tf.keras.models.load_model(os.path.join(model_dir, "modelCNN"))

    return loaded_model


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    return model.predict(input_data)


def main():
    args, _ = parse_args()

    train_data, train_labels = load_dataset(args.train)
    # CODE HELPER 7
    model = keras.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=3, batch_size=32, validation_split=0.1)

    model.save(os.path.join(args.model_dir, "modelCNN"))


if __name__ == '__main__':
    main()
