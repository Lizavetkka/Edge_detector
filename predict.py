import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

def load_model(model_path, device):
    return torch.jit.load(model_path, map_location=device)

def predict(model, image_path, device, transform):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(image.to(device))
    
    prediction = torch.sigmoid(prediction).cpu().numpy()[0][0]
    prediction = (prediction > 0.73).astype(np.uint8)
    return prediction

def save_prediction(prediction, save_path):
    predicted_mask_image = Image.fromarray(prediction * 255)
    predicted_mask_image.save(save_path)
    print(f"Prediction saved as {save_path}")
