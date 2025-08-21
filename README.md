# VisionGuard – CCTV Object Classification (Academic Baseline + Practical Hooks)

This project implements the **problem description** end-to-end using TensorFlow/Keras:

- Train a CNN with ~**60,000 samples** (Fashion-MNIST baseline) **or** CIFAR-10 (matches the specified classes).
- Evaluate on the full test set, visualize metrics, and generate a report.
- Provide a small **Streamlit dashboard** for quick demos.
- Modular code: `data.py`, `model.py`, `train.py`, `evaluate.py`, `infer.py`.

> **Note on dataset mismatch**: The prompt mixes two benchmarks.
> - Fashion-MNIST → 60,000 train samples, 28x28 grayscale (but clothing classes).  
> - CIFAR-10 → 50,000 train samples, 32x32 RGB (classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
>
> To satisfy both, this repo supports **both datasets**. Set `dataset:` in `config.yaml` to `cifar10` (recommended for class list) or `fashion_mnist` (recommended for 60k-sample requirement).

## Project Structure
```
visionguard_cctv_classifier/
├── config.yaml
├── requirements.txt
├── train.py
├── evaluate.py
├── infer.py
├── streamlit_app.py
├── models/
├── outputs/
├── logs/
└── src/
    ├── data.py
    ├── model.py
    └── utils.py
```

## Quickstart
1. **Create environment & install deps**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Choose dataset** in `config.yaml`:
   - `dataset: cifar10` (matches problem's class list)
   - `dataset: fashion_mnist` (28x28 grayscale, 60k train)

3. **Train**
   ```bash
   python train.py
   ```
   - Downloads the dataset automatically via Keras.
   - Saves best model to `models/best_model.keras`.
   - Training curves in `outputs/accuracy.png` and `outputs/loss.png`.
   - Test metrics in `outputs/test_metrics.txt`.

4. **Evaluate**
   ```bash
   python evaluate.py
   ```
   - Produces `outputs/confusion_matrix.png`, `outputs/confusion_matrix_norm.png`.
   - Writes `outputs/classification_report.txt` (precision/recall/F1 per class).

5. **Infer on a single image**
   ```bash
   python infer.py path/to/image.jpg
   ```

6. **Streamlit demo (optional)**
   ```bash
   streamlit run streamlit_app.py
   ```

## How this meets the assignment
- **Model training** on a standard benchmark (Fashion-MNIST or CIFAR-10).
- **All test images** used for evaluation (no hold-out needed beyond validation split).
- **Hyperparameter tuning** hooks via `config.yaml` (epochs, batch size, learning rate, augmentations, callbacks).
- **Visualizations** (loss/accuracy curves, confusion matrix).
- **Report** generated as `classification_report.txt` with full metrics.

## Extend to CCTV automation
Once the classifier is validated:
- Use it to **gate recording**: only trigger when predicted class ∈ {person, car, …}. For detection on video streams, consider **YOLOv8** and use its detections to start/stop recording (you already built this!).
- Or, run classification on motion-triggered frames.

## Tips
- Increase `epochs` to 40+ for stronger performance.
- Use `ReduceLROnPlateau` and try a slightly lower learning rate after several epochs.
- Try **data augmentation** if you see overfitting.
- For CIFAR-10, consider upgrading the backbone (e.g., small ResNet) later.
