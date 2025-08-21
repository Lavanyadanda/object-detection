import os, yaml, json, numpy as np, matplotlib.pyplot as plt, itertools
from sklearn.metrics import confusion_matrix, classification_report

def load_config(path: str):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def plot_training(history, out_path):
    # history.history has 'loss','val_loss','accuracy','val_accuracy'
    plt.figure()
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    plt.savefig(os.path.join(out_path, "accuracy.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(os.path.join(out_path, "loss.png"), bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(cm, class_names, out_file, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()

def save_classification_report(y_true, y_pred, class_names, out_file):
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(out_file, "w") as f:
        f.write(report)
    return report
