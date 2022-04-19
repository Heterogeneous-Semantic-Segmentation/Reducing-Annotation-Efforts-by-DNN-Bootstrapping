from typing import List
import torch
import wandb
import torch.nn as nn

from DLIP.models.zoo.compositions.base_composition import BaseComposition
from DLIP.models.zoo.decoder.unet_decoder import UnetDecoder
from DLIP.models.zoo.encoder.unet_encoder import UnetEncoder

import torchmetrics

SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
        inputs = outputs.view(-1)
        targets = labels.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + SMOOTH)/(union + SMOOTH)
                
        return IoU

class UnetSupervised(BaseComposition):
    
    def __init__(
        self,
        n_classes: int,
        input_channels: int,
        loss_fcn: nn.Module,
        encoder_filters: List = [64, 128, 256, 512, 1024],
        decoder_filters: List = [512, 256, 128, 64],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.loss_fcn = loss_fcn
        bilinear = False
        self.n_classes = n_classes
        self.append(UnetEncoder(
            input_channels = input_channels,
            encoder_filters = encoder_filters,
            dropout=dropout,
            bilinear=bilinear
        ))
        self.append(UnetDecoder(
            n_classes = n_classes,
            encoder_filters = encoder_filters,
            decoder_filters = decoder_filters,
            dropout=dropout,
            billinear_downsampling_used = bilinear
        ))
        self.append(
            nn.Softmax(dim=1) if n_classes > 1 else nn.Sigmoid()
        )
        self.global_counter = 0
 
    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        if y_true.shape[1] == 3:
            y_true = y_true[:,0:1,:,:]
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)  # shape NxC
        loss = torch.mean(loss_n_c)
        #self.log_metrics(y_pred, y_true, mode="train")
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc",torch.sum(torch.round(y_pred) == y_true) / len(torch.flatten(y_true)),prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        if y_true.shape[1] == 3:
            y_true = y_true[:,0:1,:,:]
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log_images(x, y_pred, y_true)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/acc",torch.sum(torch.round(y_pred) == y_true) / len(torch.flatten(y_true)),prog_bar=True,on_epoch=True, on_step=False,)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_true_generated,y_true = batch
        y_true_generated = y_true_generated.permute(0, 3, 1, 2)
        y_true = y_true.permute(0, 3, 1, 2)
        if y_true_generated.shape[1] == 3:
            y_true_generated = y_true_generated[:,0:1,:,:]
        y_pred = self.forward(x)
        
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log("test/dice", 1-loss, prog_bar=True, on_epoch=True, on_step=False)
        iou = iou_pytorch(y_pred,y_true)
        self.log("test/iou", iou, prog_bar=True, on_epoch=True, on_step=False)
        acc = torchmetrics.Accuracy()(y_pred.cpu(),y_true.cpu())
        self.log("test/acc", acc, prog_bar=True, on_epoch=True, on_step=False)


        loss_default = self.loss_fcn(y_true_generated,y_true)
        default_loss = torch.mean(loss_default)
        self.log("test/dice(gen_labels_v_true)", 1-default_loss, prog_bar=True, on_epoch=True, on_step=False)
        iou = iou_pytorch(y_true_generated,y_true)
        self.log("test/iou(gen_labels_v_true)", iou, prog_bar=True, on_epoch=True, on_step=False)
        acc = torchmetrics.Accuracy()(y_true_generated.cpu(),y_true.cpu())
        self.log("test/acc(gen_labels_v_true)", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def log_images(self, x, y_pred, y_true, indices=[0,1,3,4,5,6]):
        for index in indices:
            if self.n_classes > 1:
                x_log = x[index].permute(1,2,0).cpu().numpy()
                y_pred_log = y_pred[index].permute(1,2,0).cpu().numpy()
                y_true_log = y_true[index].permute(1,2,0).cpu().numpy()
                wandb.log({f"{index}": [
                    wandb.Image(x_log, caption='X'),
                    wandb.Image(y_true_log[:,:,0], caption='Y True (Channel 0)'),
                    wandb.Image(y_true_log[:,:,1], caption='Y True (Channel 1)'),
                    wandb.Image(y_pred_log[:,:,0], caption='Y Pred (Channel 0)'),
                    wandb.Image(y_pred_log[:,:,1], caption='Y Pred (Channel 1)'),
                ]
                })
            else:
                x_log = x[index].permute(1,2,0).cpu().numpy()
                y_pred_log = (y_pred[index].permute(1,2,0).cpu().numpy() > 0.5).astype(int)[:,:,0]
                y_true_log = (y_true[index].permute(1,2,0).cpu().numpy() > 0.5).astype(int)[:,:,0]
                wandb.log({
                    f"Image ({index})" : wandb.Image(x_log, masks={
                        "predictions" : {
                            "mask_data" : y_pred_log,
                            "class_labels" : {0: "background", 1: "triangle"}
                        },
                        "ground_truth" : {
                            "mask_data" : y_true_log,
                            "class_labels" : {0: "background", 1: "triangle"}
                        }
                    }
                )})
