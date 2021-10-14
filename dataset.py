import utils
import torch
import os

from PIL import Image
import transforms as T
import numpy as np


class DivingWithMasksDataset(object):
    def __init__(self, root, train):
        self.root = root
        self.transforms = get_transform(train=train)
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(utils.listdir_nohidden(os.path.join(root, "images"))))
        self.masks = list(sorted(utils.listdir_nohidden(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)

        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        unique_colors = np.unique(mask.reshape(-1,3), axis=0)[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = []
        for uc in unique_colors:
            mask_unique = np.where(np.all(mask == uc, axis=2), 1, 0)
            # print('mask unique shape: ', mask_unique.shape)
            # Check unique colours in mask
            # print(np.unique(mask_unique.reshape(-1, 1), axis=0))
            masks.append(mask_unique)

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


