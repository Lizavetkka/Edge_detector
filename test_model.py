from predict import predict, load_model
from PIL import Image
import torch
import torchvision.transforms as T

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
]) 

best_model_path = './best_model_new.pt'
best_model = load_model(best_model_path, DEVICE)

image_path = './1.jpg'
predicted_mask = predict(best_model, image_path, DEVICE, transform)
predicted_mask = predict(best_model, image_path, DEVICE, transform)
predicted_mask = Image.fromarray(predicted_mask * 255)
predicted_mask.save('predicted_mask.png')
