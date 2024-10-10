import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
from dataset import ImageDataset
from model import create_model
from train import train_model
from predict import predict, load_model
import torchvision.transforms as T
from segmentation_models_pytorch import utils
from PIL import Image


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
LR = 0.0001
EPOCHS = 40


image_dir = './images'
mask_dir = './masks'
image_filenames = os.listdir(image_dir)
mask_filenames = os.listdir(mask_dir)

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])

train_images, val_images, train_masks, val_masks = train_test_split(image_filenames, mask_filenames, test_size=0.2, random_state=42)

train_dataset = ImageDataset(image_dir, mask_dir, train_images, train_masks, transform=transform)
val_dataset = ImageDataset(image_dir, mask_dir, val_images, val_masks, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = create_model().to(DEVICE)
loss = utils.losses.DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
metrics = [utils.metrics.Fscore(), utils.metrics.IoU()]

train_model(model, train_loader, val_loader, DEVICE, loss, optimizer, metrics, EPOCHS,BATCH_SIZE)

best_model_path = './best_model_new.pt'
best_model = load_model(best_model_path, DEVICE)

image_path = './1.jpg'
predicted_mask = predict(best_model, image_path, DEVICE, transform)
predicted_mask = predict(best_model, image_path, DEVICE, transform)
predicted_mask = Image.fromarray(predicted_mask * 255)
predicted_mask.save('predicted_mask.png')
