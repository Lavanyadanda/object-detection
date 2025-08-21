import tensorflow as tf
import numpy as np

def load_dataset(name: str):
    name = name.lower()
    if name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        class_names = [
            "airplane","automobile","bird","cat","deer",
            "dog","frog","horse","ship","truck"
        ]
        # Normalize to [0,1], ensure shape (N,32,32,3)
        x_train = x_train.astype("float32") / 255.0
        x_test  = x_test.astype("float32") / 255.0
        y_train = y_train.flatten()
        y_test  = y_test.flatten()
        input_shape = (32, 32, 3)
        num_classes = 10
        return (x_train, y_train), (x_test, y_test), num_classes, input_shape, class_names

    elif name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        class_names = [
            "T-shirt/top","Trouser","Pullover","Dress","Coat",
            "Sandal","Shirt","Sneaker","Bag","Ankle boot"
        ]
        # Expand channel dimension -> (N,28,28,1) and normalize
        x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
        x_test  = np.expand_dims(x_test,  -1).astype("float32") / 255.0
        input_shape = (28, 28, 1)
        num_classes = 10
        return (x_train, y_train), (x_test, y_test), num_classes, input_shape, class_names

    else:
        raise ValueError(f"Unsupported dataset: {name}. Use 'cifar10' or 'fashion_mnist'.")
