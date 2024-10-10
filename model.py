import torch
import segmentation_models_pytorch as smp

def create_model(encoder_name='resnet18', encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )
    return model
