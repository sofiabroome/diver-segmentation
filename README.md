# diver-segmentation

This repository contains code to 1) fine-tune a MaskRCNN to segment diver instances from the Diving48 dataset, and 2) code to perform inference with such a model, and save the segmented clips as videos. This is used in the paper [Recur, Attend or Convolve? Frame Dependency Modeling Matters for Cross-Domain Robustness in Action Recognition](https://arxiv.org/abs/2112.12175) by Broomé et al., arXiv 2112.12175.

The manually labelled frames can be downloaded [here](https://doi.org/10.7910/DVN/OXKE6E) from Harvard Dataverse. A trained model checkpoint is also available on the same page (search for `checkpoint` among the files).


Please cite our paper if you found this code or dataset useful for your work.

```
@article{broome2021recur,
      title={{Recur, Attend or Convolve? On Whether Frame Dependency Modeling Matters for Cross-Domain Robustness in Action Recognition}}, 
      author={Sofia Broomé and Ernest Pokropek and Boyu Li and Hedvig Kjellström},
      booktitle = {IEEE Winter Conference on Applications in Computer Vision (WACV)},
      month = {January}, 
      year={2023}
}
```


