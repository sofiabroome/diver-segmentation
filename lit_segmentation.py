from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import pytorch_lightning as pl
from torch import nn
import torchmetrics
import torchvision
import torch


class InstanceSegmentationModule(pl.LightningModule):
    def __init__(self, optimizer, num_classes, lr, momentum, weight_decay):
        super(InstanceSegmentationModule, self).__init__()

        self.num_classes = num_classes
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.model = get_model_instance_segmentation(self.num_classes)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.iou = torchmetrics.IoU(num_classes=2)
        self.save_hyperparameters()

    def forward(self, x) -> torch.Tensor:
        x = self.model(x)
        return x

    @staticmethod
    def loss_function(y_hat, y):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss_dict = self(x, y)
        self.log('train_iou', loss_dict['iou'], prog_bar=True)
        self.log('train_loss', loss_dict['loss'], prog_bar=True)
        return {'loss': loss_dict['loss']}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss_dict = self(x, y)
        self.log('val_iou', loss_dict['iou'], prog_bar=True)
        self.log('val_loss', loss_dict['loss'], prog_bar=True)
        # By default, on_step=False, on_epoch=True for log calls in val and test
        return {'val_loss': loss_dict['loss']}

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss_dict = self(x, y)
        self.log('test_iou', loss_dict['iou'], prog_bar=True)
        self.log('test_loss', loss_dict['loss'], prog_bar=True)
        # By default, on_step=False, on_epoch=True for log calls in val and test
        return {'test_loss': loss_dict['loss']}

    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), self.lr,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay)
            return optimizer


def get_model_instance_segmentation(num_classes, hidden_layer):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


