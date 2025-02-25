import torch
from PIL import Image
import torchvision.transforms as transforms
from model import get_model

def predict_image(image_path, model_path="models/fire_detection_model.pth"):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Match training normalization
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    model = get_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(probabilities, 1)

    return f"🔥 Fire Detected! ({probabilities[0][0].item() * 100:.2f}% confidence)" if predicted.item() == 0 else f"✅ No Fire ({probabilities[0][1].item() * 100:.2f}% confidence)"

# Example Usage
# image_path = "dataset/fire_images/fire.271.png"
# print(predict_image(image_path))
