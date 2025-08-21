import tensorflow as tf

def build_cnn(input_shape, num_classes, augment_cfg=None):
    data_augmentation = tf.keras.Sequential(name="augmentation")
    if augment_cfg:
        if augment_cfg.get("flip", False):
            data_augmentation.add(tf.keras.layers.RandomFlip("horizontal"))
        rot = augment_cfg.get("rotate", 0.0)
        if rot and rot > 0:
            data_augmentation.add(tf.keras.layers.RandomRotation(rot))
        zoom = augment_cfg.get("zoom", 0.0)
        if zoom and zoom > 0:
            data_augmentation.add(tf.keras.layers.RandomZoom((-zoom, zoom)))
        contrast = augment_cfg.get("contrast", 0.0)
        if contrast and contrast > 0:
            data_augmentation.add(tf.keras.layers.RandomContrast(contrast))

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    if data_augmentation.layers:
        x = data_augmentation(x)

    # A compact but strong CNN
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="cctv_classifier_cnn")
    return model
