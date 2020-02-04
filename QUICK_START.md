# Quick Start

Follow several steps to run the code

## Data Preparation

Download data from [Kaggle](https://www.kaggle.com/c/ultrasound-nerve-segmentation/) and distribute it as shown in csv file.

## To train(EfficientNet training)

```plain
cd classification
python train.py     # put proper arguments
```

## To run(EfficientNet running)

```plain
cd classification
python test.py
```

## To train(U-Net training)

```plain
cd segmentation
python train.py     # put proper arguments
```

## To run(U-Net running)

```plain
cd segmentation
python run.py
```
