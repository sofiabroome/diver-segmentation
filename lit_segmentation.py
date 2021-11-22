from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pl_bolts.metrics.object_detection import iou
from prettytable import PrettyTable
import pytorch_lightning as pl
from torch import nn
import torchmetrics
import torchvision
import torch


class InstanceSegmentationModule(pl.LightningModule):
    def __init__(self, optimizer, num_classes, lr, momentum, weight_decay,
                 num_hidden_mask_predictor):
        super(InstanceSegmentationModule, self).__init__()

        self.num_classes = num_classes
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_hidden_mask_predictor = num_hidden_mask_predictor
        self.model = get_model_instance_segmentation(self.num_classes, self.num_hidden_mask_predictor)
        self.iou = iou
        self.save_hyperparameters()

    def forward(self, x) -> torch.Tensor:
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss_dict = self(x, y)
        sum_losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss', sum_losses, prog_bar=True)
        self.log('train_loss_cl', loss_dict['loss_classifier'], prog_bar=True)
        self.log('train_loss_box_reg', loss_dict['loss_box_reg'], prog_bar=True)
        self.log('train_loss_mask', loss_dict['loss_mask'], prog_bar=True)
        self.log('train_loss_objectness', loss_dict['loss_objectness'], prog_bar=True)
        self.log('train_loss_rpn_box_reg', loss_dict['loss_rpn_box_reg'], prog_bar=True)
        return {'loss': sum_losses}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x, y)
        val_iou = self.get_inference_metrics(output, y)
        self.log('val_iou_boxes', val_iou, prog_bar=True)
        # By default, on_step=False, on_epoch=True for log calls in val and test
        return {'val_iou_boxes': val_iou}

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss_dict = self(x, y)
        sum_losses = sum(loss for loss in loss_dict.values())
        # self.log('test_iou', loss_dict['iou'], prog_bar=True)
        self.log('test_loss', loss_dict['loss'], prog_bar=True)
        # By default, on_step=False, on_epoch=True for log calls in val and test
        return {'test_loss': sum_losses}

    def get_inference_metrics(self, output, y):
        outputs = [{k: v for k, v in t.items()} for t in output]
        div_factor = len(outputs)
        val_iou = 0
        for ind, o in enumerate(outputs):
            sample_iou = self.iou(o['boxes'], y[ind]['boxes'])
            nb_predictions = len(sample_iou)
            nb_instances = len(y[ind]['boxes'])
            self.log('val_nb_predictions', nb_predictions, prog_bar=False)
            k = int(0.1 * nb_predictions)
            k = 1 if k == 0 else k
            if nb_predictions == 0 and nb_instances > 0:
                val_iou = 0
            else:
                val_iou += torch.mean(torch.topk(sample_iou[:,0], k=k).values) 
        return val_iou/div_factor

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


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    encoder_params = 0
    print(model.named_parameters())
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
        if not 'linear' in name:
            encoder_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return encoder_params, total_params

