## DCGAN (Deep Convolutional Generative Adversarial Network)

This is a pytorch implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434).The following is the architecture of the network.

![alt tag](imgs/nw.png)

### Prerequisites

- Python 3.5 3.6
- torch == 1.2.0 (or higher)
- torchvision == 0.4.0 (or higher)

You can go to the official website of [pytorch](https://pytorch.org/get-started/locally/#windows-package-manager) to install torch/torchvision correctly.

### Usage

1. Download the dataset that you want this network to be trained on and put the data set into a `data` folder. You can also specify the path of your dataset with `dataRoot` argument.   
2. Run $python train.py$ with any arguments you want.

The following is the list of the arguments with explaination:

- **--batchSize**: (int) The batch size of your training set. The default value is `16`.
- **--imgSize**  : (int) Size of your output images. The default value is `64`.
- **--nz**       : (int) 

