from lit_segmentation import get_model_instance_segmentation, InstanceSegmentationModule
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from frames_for_segmentation_dataset import VideoFramesForSegmentation
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


def save_masked_frame(pil_img, save_folder, video_id, class_index, seq_ind):
    video_save_folder = os.path.join(save_folder, video_id)
    if not os.path.isdir(video_save_folder):
        os.mkdir(video_save_folder)
    plt.imshow(np.asarray(pil_img))
    plt.axis('off')
    plt.savefig(fname=f'{save_folder}/{video_id}/mask_{seq_ind}_class{class_index}.jpg', bbox_inches='tight', pad_inches=0)
    plt.clf()


def draw_with_mask(input_tensor, all_masks):
    nb_masks = all_masks.shape[0]
    if nb_masks == 0:
        mask = torch.zeros(all_masks.shape[1], all_masks.shape[2])
        mask = (mask == 1)
        nb_masks = 1
    else:
        mask = (all_masks > 0)
        if len(mask.shape) == 3:
            mask = torch.sum(mask, dim=0).bool()
    mask = ~mask
    # inv_normalize = transforms.Normalize(
    #     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
    #     std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    # )
    # input_tensor = inv_normalize(input_tensor)
    input_tensor = (input_tensor * 255).type(torch.uint8)
    # average_color = tuple([np.mean(input_tensor[:, :, i].cpu().numpy()) for i in range(3)])
    average_color = [tuple([124, 117, 104])] * nb_masks
    img_tensor = draw_segmentation_masks(input_tensor.cpu(), masks=mask.cpu(), alpha=1, colors=average_color)
    img_array = img_tensor.detach()
    img = F.to_pil_image(img_array)
    return img


def draw_with_mask_and_blur_bg(input_tensor, all_masks):
    nb_masks = all_masks.shape[0]
    if nb_masks == 0:
        mask = torch.zeros(all_masks.shape[1], all_masks.shape[2])
        mask = (mask == 1)
        nb_masks = 1
    else:
        mask = (all_masks > 0.4)
        if len(mask.shape) == 3:
            mask = torch.sum(mask, dim=0).bool()
    mask = mask.cpu()
    blurrer = transforms.GaussianBlur(kernel_size=(55, 55), sigma=(15.0))
    blurred_img = blurrer(input_tensor.cpu())

    only_divers = mask * input_tensor.cpu()

    inv_mask = ~mask
    only_blurred_bg = inv_mask * blurred_img

    together = only_divers + only_blurred_bg

    img_tensor = (together * 255).type(torch.uint8)

    img_array = img_tensor.detach()
    img = F.to_pil_image(img_array)
    return img


def draw_with_box(input_tensor, all_masks):
    nb_masks = all_masks.shape[0]
    input_tensor = (input_tensor * 255).type(torch.uint8).cpu()

    if nb_masks == 0:
        img_tensor = input_tensor
    else:
        mask = (all_masks > 0)
        average_color = [tuple([124, 117, 104])] * nb_masks
        img_tensor = draw_segmentation_masks(
            input_tensor, masks=mask.cpu(), alpha=1, colors=average_color)

    img_array = img_tensor.detach()
    img = F.to_pil_image(img_array)
    return img


def draw_with_bounding_box(input_tensor, boxes):
    # Transparent fill, for complete box fill use draw_with_box()
    input_tensor = (input_tensor * 255).type(torch.uint8).cpu()
    if boxes.shape[0] == 0:
        img_array = input_tensor.cpu()
    else:
        average_color = tuple([124, 117, 104])
        img_tensor = draw_bounding_boxes(input_tensor, boxes=boxes, colors=average_color,
                                         fill=True) 
        img_array = img_tensor.detach()
    img = F.to_pil_image(img_array)
    return img


def save_frame(input_tensor, video_id, seq_ind, save_folder='frames_with_masks/'):
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


def mask_video(model, batch_loader, config, class_index, save_folder, mode):

    for i, (x, video_id_batch) in enumerate(tqdm(batch_loader)):
        if i == 0:
            b, t, c, h, w = x.shape
            num_images = b*t
        x = torch.reshape(x, (num_images, c, h, w))

        if torch.cuda.is_available():
            x = x.to('cuda')

        with torch.no_grad():
            output_all = model(x)
            # output = model(x)['out']
        # masks_batch = output.argmax(dim=1)

        if 'crop_mask' in mode or mode == 'box_via_mask':
            output_batch = [torch.squeeze(output_all[i]['masks']) for i in range(num_images)] 
        if mode == 'mask_box':
            output_batch = [torch.squeeze(output_all[i]['boxes']) for i in range(num_images)] 

        batch_ind = 0
        fps = 5
        container = av.open(save_folder + '/' + video_id_batch[batch_ind] + '_cropped.mp4', mode='w')
        stream = container.add_stream('mpeg4', rate=fps)
        stream.width = 492
        stream.height = 369
        stream.pix_fmt = 'yuv420p'

        for sample_ind in range(num_images):
            seq_ind = sample_ind%config['clip_size']
            if seq_ind == 0 and sample_ind != 0:
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
                batch_ind += 1
                container = av.open(save_folder + '/' + video_id_batch[batch_ind] + '_cropped.mp4', mode='w')
                stream = container.add_stream('mpeg4', rate=fps)
                stream.width = 492
                stream.height = 369
                stream.pix_fmt = 'yuv420p'

            video_id = video_id_batch[batch_ind]
            print('vid id: ', video_id)
            print('sample ind: ', sample_ind)
            image_tensor = x[sample_ind]

            if mode == 'crop_mask':
                mask = output_batch[sample_ind]
                nb_masks = mask.shape[0]
                print('mask shape: ', mask.shape)
                if len(mask.shape) == 3 and nb_masks > 0:
                    nb_to_keep = 2 if nb_masks > 2 else nb_masks
                    mask = mask[:nb_to_keep, :, :]  # The first is the one with highest classification score
                masked = draw_with_mask( input_tensor=image_tensor, all_masks=mask)

            if mode == 'crop_mask_blur_bg':
                mask = output_batch[sample_ind]
                nb_masks = mask.shape[0]
                print('mask shape: ', mask.shape)
                if len(mask.shape) == 3 and nb_masks > 0:
                    nb_to_keep = 2 if nb_masks > 2 else nb_masks
                    mask = mask[:nb_to_keep, :, :]  # The first is the one with highest classification score
                masked = draw_with_mask_and_blur_bg(input_tensor=image_tensor, all_masks=mask)

            if mode == 'box_via_mask':
                masks = output_batch[sample_ind]
                masked = draw_with_box(input_tensor=image_tensor, all_masks=masks)
                print('masks shape: ', masks.shape)

            if mode == 'mask_box':
                box = output_batch[sample_ind]
                if len(box.shape) == 2 and box.shape[0] > 0:
                    boxes = box  # The first is the one with highest classification score
                if len(box.shape) == 1:
                    boxes = box.unsqueeze(0)  # The drawing functinos wants (N, 4)
                print('boxes: ', boxes)
                masked = draw_with_box(input_tensor=image_tensor, boxes=boxes)
            save_masked_frame(masked, save_folder, video_id, class_index, seq_ind)
            frame = av.VideoFrame.from_image(masked)
            for packet in stream.encode(frame):
                container.mux(packet)
        # if i > 10:
        #     break

    for packet in stream.encode():
        container.mux(packet)


def main():

    # Load configurations

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='json config file path')
    parser.add_argument('--save-folder', '-sf', help='save folder, a subfolder under results')
    parser.add_argument('--mode', '-m', help='crop_mask | mask_box')
    args = parser.parse_args()
    config = utils.load_json_config(args.config)
    save_folder = os.path.join('results', args.save_folder)

    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

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
    print('Created dataset...')

    batch_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True)
    print('Created dataloader...')

    class_index = -1

    mask_video(model, batch_loader, config, class_index, save_folder, mode=args.mode)
    # save_frames(batch_loader, config)


if __name__ == '__main__':
    main()
