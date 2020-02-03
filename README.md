# NerveCeg: Effective nerve segmentation with two-stage neural network

In nerve segmentation task, there was a critical shortcoming which affects very bad in the result — the problem nerve mask with no annotated mask made neural network biased. As the unannotated mask is more than 50% of the dataset, the trained neural network tended to draw blank masks on any nerve image. To solve this problem, we suggest a two-stage neural network which classifies and do segmentation task.

The first part is a binary classification task. The existence of a nerve mask is labeling criteria. By adding all values in a nerve mask, we can find out whether the mask is empty or not. In this code, we used EfficientNet-B4 for classifying nerve existence.

The second part is the nerve segmentation task. After confirming nerve presence, we segment nerve by U-Net, and others derived from U-Net. The reason for selecting U-Net and it's affiliated models is that the feature accumulation component gives a massive advantage in medical imagery segmentation.

![Architechture of NerveCeg](https://user-images.githubusercontent.com/40779417/73638667-3410ee00-46ae-11ea-9644-cc62242d5651.png)

## LICENSE

Check [LICENSE](https://github.com/kim-younghan/NerveCeg/blob/master/LICENSE).

## Installation

Please check Installation procedure in [INSTALL.md](https://github.com/kim-younghan/NerveCeg/blob/master/INSTALL.md).

## Quick Start

Follow the instructions in [QUICK_START.md](https://github.com/kim-younghan/NerveCeg/blob/master/QUICK_START.md).

## Acknowledgments

EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks by Mingxing Tan, Quoc V. Le

```plain
@misc{tan2019efficientnet,
    title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
    author={Mingxing Tan and Quoc V. Le},
    year={2019},
    eprint={1905.11946},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

U-Net: Convolutional Networks for Biomedical Image Segmentation by Olaf Ronneberger, Philipp Fischer, Thomas Brox

```plain
@misc{ronneberger2015unet,
    title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
    author={Olaf Ronneberger and Philipp Fischer and Thomas Brox},
    year={2015},
    eprint={1505.04597},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

Road Extraction by Deep Residual U-Net by Zhengxin Zhang, Qingjie Liu, Yunhong Wang

```plain
@article{Zhang_2018,
   title={Road Extraction by Deep Residual U-Net},
   volume={15},
   ISSN={1558-0571},
   url={http://dx.doi.org/10.1109/LGRS.2018.2802944},
   DOI={10.1109/lgrs.2018.2802944},
   number={5},
   journal={IEEE Geoscience and Remote Sensing Letters},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Zhang, Zhengxin and Liu, Qingjie and Wang, Yunhong},
   year={2018},
   month={May},
   pages={749–753}
}
```

Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation by Md Zahangir Alom, Mahmudul Hasan, Chris Yakopcic, Tarek M. Taha, Vijayan K. Asari

```plain
@misc{alom2018recurrent,
    title={Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation},
    author={Md Zahangir Alom and Mahmudul Hasan and Chris Yakopcic and Tarek M. Taha and Vijayan K. Asari},
    year={2018},
    eprint={1802.06955},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

Attention U-Net: Learning Where to Look for the Pancreas by Ozan Oktay, Jo Schlemper, Loic Le Folgoc, Matthew Lee, Mattias Heinrich, Kazunari Misawa, Kensaku Mori, Steven McDonagh, Nils Y Hammerla, Bernhard Kainz, Ben Glocker, Daniel Rueckert

```plain
@misc{oktay2018attention,
    title={Attention U-Net: Learning Where to Look for the Pancreas},
    author={Ozan Oktay and Jo Schlemper and Loic Le Folgoc and Matthew Lee and Mattias Heinrich and Kazunari Misawa and Kensaku Mori and Steven McDonagh and Nils Y Hammerla and Bernhard Kainz and Ben Glocker and Daniel Rueckert},
    year={2018},
    eprint={1804.03999},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

PyTorch CNN Visualizations by Utku Ozbulak

```plain
@misc{uozbulak_pytorch_vis_2019,
  author = {Utku Ozbulak},
  title = {PyTorch CNN Visualizations},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/utkuozbulak/pytorch-cnn-visualizations}},
  commit = {3460e7f014f52f4099c1a4864e1534de9cc901e7}
}
```
