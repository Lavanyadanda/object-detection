import os, datetime, tensorflow as tf, numpy as np, yaml
from src.data import load_dataset
from src.model import build_cnn
from src.utils import load_config, ensure_dirs, plot_training

def main():
    cfg = load_config("config.yaml")
    tf.keras.utils.set_random_seed(cfg.get("seed", 42))

    (x_train, y_train), (x_test, y_test), num_classes, input_shape, class_names = load_dataset(cfg["dataset"])
    models_dir = cfg["models_dir"]
    outputs_dir = cfg["outputs_dir"]
    logs_dir = cfg["logs_dir"]

    ensure_dirs(models_dir, outputs_dir, logs_dir)

    model = build_cnn(input_shape, num_classes, augment_cfg=cfg.get("augment", {}))
    lr = cfg.get("learning_rate", 1e-3)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=cfg.get("early_stopping_patience",5), restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(models_dir, "best_model.keras"), save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=cfg.get("reduce_lr_patience",2), factor=cfg.get("reduce_lr_factor",0.5), monitor="val_loss"),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(logs_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    ]

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=cfg.get("epochs", 20),
        batch_size=cfg.get("batch_size", 64),
        callbacks=callbacks,
        verbose=2
    )

    model.save(os.path.join(models_dir, "last_model.keras"))
    plot_training(history, outputs_dir)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    with open(os.path.join(outputs_dir, "test_metrics.txt"), "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\nTest loss: {test_loss:.4f}\n")

    print(f"Training complete. Test Acc={test_acc:.4f}, Test Loss={test_loss:.4f}")

if __name__ == "__main__":
    main()
