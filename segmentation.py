from frames_for_segmentation_dataset import VideoFramesForSegmentation
from lit_segmentation import get_model_instance_segmentation, InstanceSegmentationModule
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import argparse
import random
import torch
import utils
import os
import av


def plot_with_mask(input_tensor, all_masks, class_index,
                   video_id, seq_ind, save_folder='results'):
    mask = (all_masks == class_index)
    mask = ~mask
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )
    input_tensor = inv_normalize(input_tensor)
    input_tensor = (input_tensor * 255).type(torch.uint8)
    average_color = tuple([np.mean(input_tensor[:, :, i].numpy()) for i in range(3)])
    img_tensor = draw_segmentation_masks(input_tensor, masks=mask, alpha=1, colors=average_color)
    img_array = img_tensor.detach()
    img = F.to_pil_image(img_array)
    video_save_folder = os.path.join(save_folder, video_id)
    if not os.path.isdir(video_save_folder):
        os.mkdir(video_save_folder)
    plt.imshow(np.asarray(img))
    plt.savefig(fname=f'{save_folder}/{video_id}/mask_{seq_ind}_class{class_index}.jpg')
    plt.clf()
    return img


def save_frame(input_tensor, video_id, seq_ind, save_folder='frames_with_masks/'):
    # inv_normalize = transforms.Normalize(
    #     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
    #     std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    # )
    # input_tensor = inv_normalize(input_tensor)
    input_tensor = (input_tensor * 255).type(torch.uint8)
    img = F.to_pil_image(input_tensor)
    plt.imshow(np.asarray(img))
    plt.axis('off')
    plt.savefig(fname=f'{save_folder}/{video_id}_{seq_ind}.jpg', bbox_inches='tight', pad_inches=0)
    plt.clf()


def save_frames(batch_loader, config):
    for i, (x_batch, video_id_batch) in enumerate(tqdm(batch_loader)):
        for j, sample in enumerate(x_batch):
            seq_ind = random.randint(0, config['clip_size']-10)
            image_tensor = sample[seq_ind]
            video_id = video_id_batch[j]
            print('vid id: ', video_id)
            print('seq ind: ', seq_ind)
            save_frame(image_tensor, video_id, seq_ind)
        if i > 7:
            break


def mask_video(model, batch_loader, config):
    for i, (x, video_id_batch) in enumerate(tqdm(batch_loader)):
        if i == 0:
            b, t, c, h, w = x.shape
            num_images = b*t
        x = torch.reshape(x, (num_images, c, h, w))
        if torch.cuda.is_available():
            x = x.to('cuda')
        with torch.no_grad():
            output = model(x)['out']
        masks_batch = output.argmax(dim=1)
        batch_ind = 0
        fps = 5
        container = av.open('results/' + video_id_batch[batch_ind] + '_cropped.mp4', mode='w')
        stream = container.add_stream('mpeg4', rate=fps)
        stream.width = 224
        stream.height = 224
        stream.pix_fmt = 'yuv420p'
        for sample_ind in range(num_images):
            seq_ind = sample_ind%config['clip_size']
            if seq_ind == 0 and sample_ind != 0:
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
                batch_ind += 1
                container = av.open('results/' + video_id_batch[batch_ind] + '_cropped.mp4', mode='w')
                stream = container.add_stream('mpeg4', rate=fps)
                stream.width = 224
                stream.height = 224
                stream.pix_fmt = 'yuv420p'
            video_id = video_id_batch[batch_ind]
            image_tensor = x[sample_ind]
            mask = masks_batch[sample_ind]
            print('vid id: ', video_id)
            print('sample ind: ', sample_ind)
            masked = plot_with_mask(
                input_tensor=image_tensor, all_masks=mask,
                class_index=15, video_id=video_id,
                seq_ind=seq_ind)
            frame = av.VideoFrame.from_image(masked)
            for packet in stream.encode(frame):
                container.mux(packet)
        if i > 10:
            break
    for packet in stream.encode():
        container.mux(packet)


def main():
    # load configurations

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='json config file path')
    args = parser.parse_args()
    config = utils.load_json_config(args.config)

    # model = torch.hub.load('pytorch/vision:v0.10.0', config['deeplab_backbone'], pretrained=True)
    # model = get_model_instance_segmentation(num_classes=2, hidden_layer=256)
    model = InstanceSegmentationModule.load_from_checkpoint(config['ckpt_path'])
    model.eval()

    if torch.cuda.is_available():
        model.to('cuda')

    loader = VideoFramesForSegmentation(
        root=config['data_folder'],
        json_file_input=config['json_data_train'],
        json_file_labels=config['json_file_labels'],
        clip_size=config['clip_size'],
        nclips=config['nclips_train_val'],
        step_size=config['step_size_train_val'],
        get_item_id=True)

    batch_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True)

    mask_video(model, batch_loader, config)
    # save_frames(batch_loader, config)


if __name__ == '__main__':
    main()
