import os, numpy as np, tensorflow as tf
from src.data import load_dataset
from src.utils import load_config, ensure_dirs, plot_confusion_matrix, save_classification_report

def main():
    cfg = load_config("config.yaml")
    (x_train, y_train), (x_test, y_test), num_classes, input_shape, class_names = load_dataset(cfg["dataset"])
    ensure_dirs(cfg["outputs_dir"])

    model_path = os.path.join(cfg["models_dir"], "best_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Train the model first.")

    model = tf.keras.models.load_model(model_path)

    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names, os.path.join(cfg["outputs_dir"], "confusion_matrix.png"), normalize=False)
    plot_confusion_matrix(cm, class_names, os.path.join(cfg["outputs_dir"], "confusion_matrix_norm.png"), normalize=True)

    report = save_classification_report(y_test, y_pred, class_names, os.path.join(cfg["outputs_dir"], "classification_report.txt"))
    print(report)

if __name__ == "__main__":
    main()
