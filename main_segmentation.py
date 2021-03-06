import os
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import seed_everything

import utils
import argparse
from data_module import DivingSegmentationDataModule
from lit_segmentation import InstanceSegmentationModule, count_parameters


def main():
    # load configurations

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='json config file path')
    parser.add_argument('--eval_only', '-e', action='store_true',
                        help="evaluate trained model on validation data.")
    parser.add_argument('--resume', '-r', action='store_true',
                        help="resume training from a given checkpoint.")
    parser.add_argument('--test_run', action='store_true',
                        help="quick test run")
    parser.add_argument('--job_identifier', '-j', help='Unique identifier for run,'
                                                       'avoids overwriting model.')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    config = utils.load_json_config(args.config)

    wandb_logger = WandbLogger(project='diver-segmentation', config=config)

    # seed_everything(42, workers=True)

    model = InstanceSegmentationModule(optimizer=config['optimizer'],
                                       num_classes=config['num_classes'],
                                       lr=config['lr'], momentum=config['momentum'],
                                       weight_decay=config['weight_decay'],
                                       num_hidden_mask_predictor=config['hidden_mask_predictor'])

    config['nb_encoder_params'], config['nb_trainable_params'] = count_parameters(model)
    print('\n Nb encoder params: ', config['nb_encoder_params'], 'Nb params total: ', config['nb_trainable_params'])

    checkpoint_callback = ModelCheckpoint(monitor='val_iou_boxes', mode='max',
                                          verbose=True,
                                          filename='{epoch}-{val_loss:.2f}-{val_iou_boxes:.4f}')

    early_stop_callback = EarlyStopping(
        monitor='val_iou_boxes',
        min_delta=0.00,
        patience=config['early_stopping_patience'],
        verbose=False,
        mode='max'
    )

    callbacks = [checkpoint_callback, early_stop_callback]

    trainer = pl.Trainer.from_argparse_args(
        args, max_epochs=config['num_epochs'],
        progress_bar_refresh_rate=1,
        callbacks=callbacks,
        weights_save_path=os.path.join(config['output_dir'], args.job_identifier),
        logger=wandb_logger,
        plugins=DDPPlugin(find_unused_parameters=False))

    if trainer.gpus is not None:
        config['num_workers'] = int(trainer.gpus / 8 * 128)
    else:
        config['num_workers'] = 0

    if config['inference_from_checkpoint_only']:
        model_from_checkpoint = InstanceSegmentationModule.load_from_checkpoint(config['ckpt_path'])

    else:
        train_dm = DivingSegmentationDataModule(data_dir=config['data_folder'], config=config)
        trainer.fit(model, train_dm)
        wandb_logger.log_metrics({'best_val_iou_boxes': trainer.checkpoint_callback.best_model_score})


if __name__ == '__main__':
    main()
