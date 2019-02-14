# Autoencoder
Trains autoencoder on MNIST.

AE models are tied-weight (weights are shared between encoder and decoder).

## Requirements
Chainer, OpenCV

```bash
$ pip install chainer opencv-python
```

##  How to run
```bash
$ python ae.py [options]
```

You can read help with `-h` option.

```bash
$ python ae.py -h
usage: ae.py [-h] [-b BATCHSIZE] [-e EPOCH] [-g GPU] [--graph GRAPH] [--cnn]
             [--lam LAM] [-m MODEL] [-r RESULT]

optional arguments:
  -h, --help            show this help message and exit
  -b BATCHSIZE, --batchsize BATCHSIZE
                        batchsize
  -e EPOCH, --epoch EPOCH
                        iteration
  -g GPU, --gpu GPU     GPU ID
  --graph GRAPH         computational graph
  --cnn                 CNN
  --lam LAM             weight decay
  -m MODEL, --model MODEL
                        model file name
  -r RESULT, --result RESULT
                        result directory
```

