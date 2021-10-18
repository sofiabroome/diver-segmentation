import av
import time
import torch
import numpy as np

from data_parser import WebmDataset, Mp4Dataset
from torchvision import transforms


class VideoFramesForSegmentation(torch.utils.data.Dataset):

    def __init__(self, root, json_file_input, json_file_labels, clip_size,
                 nclips, step_size, is_val=False, get_item_id=False, is_test=False):
        # self.dataset_object = WebmDataset(json_file_input, json_file_labels,
        #                                   root, is_test=is_test)
        self.dataset_object = Mp4Dataset(json_file_input, json_file_labels,
                                         root, is_test=is_test)
        self.json_data = self.dataset_object.json_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.root = root
        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val
        self.get_item_id = get_item_id

    def __getitem__(self, index):

        item = self.json_data[index]

        # Open video file
        reader = av.open(item.path)  # Takes around 0.005 seconds.

        try:
            imgs = []
            imgs = [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]  # 0.5 s.
        except (RuntimeError, ZeroDivisionError) as exception:
            print('{}: WEBM reader cannot open {}. Empty '
                  'list returned.'.format(type(exception).__name__, item.path))
        # preprocess = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        preprocess = transforms.Compose([
            transforms.ToTensor()
        ])
        imgs = [preprocess(img) for img in imgs]
        evenly_spaced_frame_indices = np.round(np.linspace(0, len(imgs) - 1, self.clip_size)).astype(int)
        imgs = [imgs[ind] for ind in evenly_spaced_frame_indices]

        # format data to torch
        data = torch.stack(imgs)
        if self.get_item_id:
            return data, item.id
        else:
            return data

    def __len__(self):
        return len(self.json_data)


if __name__ == '__main__':
    upscale_size = int(84 * 1.1)
    smth_root = '/local_storage/users/sbroome/something-something/20bn-something-something-v2/'

    root = '/Users/sbroome/Downloads/rgb/diving48_rgb_webm/'
    json_data_train = "/Users/sbroome/Downloads/Diving48_V2_train.json"
    json_data_val = "/Users/sbroome/Downloads/Diving48_V2_test.json"
    json_file_labels = "/Users/sbroome/Downloads/Diving48_vocab.json"

    # loader = VideoFramesForSegmentation(root=smth_root,
    #                      json_file_input=smth_root + "annotations/something-something-v2-train.json",
    #                      json_file_labels=smth_root + "annotations/something-something-v2-labels.json",
    #                      clip_size=36,
    #                      nclips=1,
    #                      step_size=1,
    #                      is_val=False,
    #                      )
    loader = VideoFramesForSegmentation(root=root,
                         json_file_input=json_data_train,
                         json_file_labels=json_file_labels,
                         clip_size=36,
                         nclips=1,
                         step_size=1,
                         is_val=False,
                         )
    # fetch a sample
    # data_item, target_idx = loader[1]
    # save_images_for_debug("input_images_2", data_item.unsqueeze(0))
    # print("Label = {}".format(loader.classes_dict[target_idx]))

    import time
    from tqdm import tqdm

    batch_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=10, shuffle=False,
        num_workers=2, pin_memory=True)

    start = time.time()
    for i, a in enumerate(tqdm(batch_loader)):
        import ipdb; ipdb.set_trace()
        if i == 0:
            b, t, c, h, w = a[0].shape
        a[0] = torch.reshape(a[0], (-1, c, h, w))
        if i > 10:
            break
        pass
    print("Size --> {}".format(a[0].size()))
    print(time.time() - start)
