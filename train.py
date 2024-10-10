import torch
from segmentation_models_pytorch import utils

def train_model(model, train_loader, val_loader, device, loss, optimizer, metrics, epochs,batch_size):

    metrics = [
        utils.metrics.Fscore(),
        utils.metrics.IoU()
    ]

    optimizer = optimizer  

    train_epoch = utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=device,
        verbose=True,  
    )

    valid_epoch = utils.train.ValidEpoch(
        model, 
        loss=loss,  
        metrics=metrics, 
        device=device,
        verbose=True,  
    )


    max_score = 0


    loss_logs = {"train": [], "val": []}
    metric_logs = {"train": [], "val": []}
    iou_logs = {"train": [], "val": []}
    for epoch in range(epochs):
        
        print(f'\nEpoch: {epoch + 1}/{epochs}')
        
        
        train_logs = train_epoch.run(train_loader)
        train_loss = train_logs['dice_loss']
        train_fscore = train_logs['fscore']
        train_iou = train_logs['iou_score']
        loss_logs["train"].append(train_loss)
        metric_logs["train"].append(train_fscore)
        iou_logs["train"].append(train_iou)
        print(f"Train Loss: {train_loss:.4f}, F-score: {train_fscore:.4f}, IoU: {train_iou:.4f}")

        
        valid_logs = valid_epoch.run(val_loader)
        val_loss = valid_logs['dice_loss']
        val_fscore = valid_logs['fscore']
        val_iou = valid_logs['iou_score']
        loss_logs["val"].append(val_loss)
        metric_logs["val"].append(val_fscore)
        iou_logs["val"].append(val_iou)
        print(f"Val Loss: {val_loss:.4f}, F-score: {val_fscore:.4f}, IoU: {val_iou:.4f}")
    
        if max_score < valid_logs['fscore']:
            max_score = valid_logs['fscore']
            torch.save(model, 'best_model_new.pth')
            print('Model saved!')

            trace_image = torch.randn(batch_size, 3, 256, 256)  
            traced_model = torch.jit.trace(model, trace_image.to(device))
            torch.jit.save(traced_model, 'best_model_new.pt')
            print('Traced model saved!')

        print("LR:", optimizer.param_groups[0]['lr'])
