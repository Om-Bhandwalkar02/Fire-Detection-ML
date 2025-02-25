import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' (non-GUI)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from model import get_model
from preprocess import load_data
import os
import numpy as np

def analyze_results():

    os.makedirs("static/analysis", exist_ok=True)

    _, test_loader, _ = load_data()
    model = get_model()
    model.load_state_dict(torch.load("models/fire_detection_model.pth"))
    model.eval()

    y_pred, y_true = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.float(), labels
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            y_pred.extend(predictions.numpy())
            y_true.extend(labels.numpy())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fire', 'Fire'], yticklabels=['No Fire', 'Fire'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    confusion_matrix_path = "static/analysis/confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    plt.close()

    # Interpret Confusion Matrix
    tn, fp, fn, tp = cm.ravel()
    total_samples = np.sum(cm)
    correct_predictions = tn + tp
    misclassifications = fp + fn

    cm_explanation = (
        f"📊 Confusion Matrix Explanation:\n"
        f"   - True Negatives (TN): {tn} → Correctly predicted 'No Fire'\n"
        f"   - False Positives (FP): {fp} → Wrongly predicted 'Fire' when there was no fire\n"
        f"   - False Negatives (FN): {fn} → Missed actual 'Fire' cases\n"
        f"   - True Positives (TP): {tp} → Correctly predicted 'Fire'\n"
        f"   - Model correctly classified {correct_predictions}/{total_samples} samples.\n"
        f"   - {misclassifications} misclassified cases."
    )

    # Training Loss Curve
    with open("loss_values.txt", "r") as f:
        losses = [float(line.strip()) for line in f.readlines()]

    plt.plot(losses, label="Training Loss", color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    loss_curve_path = "static/analysis/loss_curve.png"
    plt.savefig(loss_curve_path)
    plt.close()

    # Loss Curve Explanation
    if len(losses) > 1:
        loss_difference = losses[0] - losses[-1]
        loss_trend = "decreasing" if loss_difference > 0 else "increasing"
    else:
        loss_trend = "N/A (insufficient data)"

    loss_explanation = (
        f"📈 Loss Curve Explanation:\n"
        f"   - Initial Loss: {losses[0]:.4f}\n"
        f"   - Final Loss: {losses[-1]:.4f}\n"
        f"   - Trend: The loss is {loss_trend}, indicating model learning progress."
    )

    # Model Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Print Results in CMD
    print("🔥 Model Analysis Results:")
    print(f"✅ Accuracy: {accuracy * 100:.2f}%")
    print(f"📊 Confusion Matrix saved at: {confusion_matrix_path}")
    print(cm_explanation)  # Print Confusion Matrix Explanation
    print(f"📈 Loss Curve saved at: {loss_curve_path}")
    print(loss_explanation)  # Print Loss Curve Explanation

    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix_path,
        "loss_curve": loss_curve_path,
        "confusion_matrix_explanation": cm_explanation,
        "loss_curve_explanation": loss_explanation
    }

if __name__ == "__main__":
    analyze_results()
