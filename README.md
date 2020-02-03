# NerveCeg

Effective nerve segmentation with two-stage neural network

In nerve segmentation task, there was a critical shortcoming which affects very bad in the result — the problem nerve mask with no annotated mask made neural network biased. As the unannotated mask is more than 50% of the dataset, the trained neural network tended to draw blank masks on any nerve image. To solve this problem, we suggest a two-stage neural network which classifies and do segmentation task.

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
