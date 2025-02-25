import os
from preprocess import load_data
from train import get_model, train_model

if __name__ == "__main__":
    print("\U0001F4E5 Loading dataset...")
    train_loader, test_loader, classes = load_data()
    print(f"✅ Dataset Loaded! Classes: {classes}")

    dataset_path = "dataset"
    fire_count = len(os.listdir(os.path.join(dataset_path, "fire_images")))
    non_fire_count = len(os.listdir(os.path.join(dataset_path, "non_fire_images")))

    print(f"🔥 Fire Images: {fire_count}")
    print(f"🚫 Non-Fire Images: {non_fire_count}")

    print("\U0001F4E6 Initializing Model...")
    model = get_model()

    print("\U0001F3AF Training Model...")
    train_model(model, train_loader)

    print("✅ Model training complete! Run `python result_analysis.py` to view the evaluation analysis.")