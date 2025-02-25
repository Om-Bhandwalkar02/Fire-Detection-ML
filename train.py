import os
import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, train_loader, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    loss_values = []
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.float(), labels
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_values.append(avg_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/fire_detection_model.pth")
    with open("loss_values.txt", "w") as f:
        for loss in loss_values:
            f.write(f"{loss}\n")

    print("✅ Model saved successfully!")