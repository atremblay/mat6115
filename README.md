# MAT6115 - Dynamical Systems


## Semester Project

[Reverse engineering recurrent networks for sentiment classification reveals line attractor dynamics](https://arxiv.org/abs/1906.10720)

First clone the project
`git clone https://github.com/atremblay/mat6115.git`

cd into the directory
`cd mat6115`

Install it in editable mode (or not, up to you)
`pip install -e .`

Then some command line tools are available

Run the model training
`mat6115 train --n_layers 3 --cuda 1 --rnn_type gru`

Run the artifacts
`mat6115 artifacts -l gru_3layer_100 --rnn_layer 1  --fixed_point --unique_fixed_point --pca`

```
❯ mat6115 -h
usage: mat6115 [-h] {train,artifacts} ...

positional arguments:
  {train,artifacts}  Tools for semester project
    train            Train tool
    artifacts        Create artifacts for further analysis.

optional arguments:
  -h, --help         show this help message and exit
```


```
❯ mat6115 train -h
usage: mat6115 train [-h] [-s SAVE_PATH] [-d {imdb}] [--rnn_type {gru,rnn}] [--n_layers N_LAYERS]
                     [-e {charngram.100d,fasttext.en.300d,fasttext.simple.300d,glove.42B.300d,glove.840B.300d,glove.twitter.27B.25d,glove.twitter.27B.50d,glove.twitter.27B.100d,glove.twitter.27B.200d,glove.6B.50d,glove.6B.100d,glove.6B.200d,glove.6B.300d}] [--cuda CUDA]

optional arguments:
  -h, --help            show this help message and exit
  -s SAVE_PATH, --save_path SAVE_PATH
                        Path where to save all the files (default .)
  -d {imdb}, --dataset {imdb}
                        Supported dataset: imdb
  --rnn_type {gru,rnn}  Type of RNN layer
  --n_layers N_LAYERS
  -e {charngram.100d,fasttext.en.300d,fasttext.simple.300d,glove.42B.300d,glove.840B.300d,glove.twitter.27B.25d,glove.twitter.27B.50d,glove.twitter.27B.100d,glove.twitter.27B.200d,glove.6B.50d,glove.6B.100d,glove.6B.200d,glove.6B.300d}, --embedding {charngram.100d,fasttext.en.300d,fasttext.simple.300d,glove.42B.300d,glove.840B.300d,glove.twitter.27B.25d,glove.twitter.27B.50d,glove.twitter.27B.100d,glove.twitter.27B.200d,glove.6B.50d,glove.6B.100d,glove.6B.200d,glove.6B.300d}
                        Initial embedding to use for words
  --cuda CUDA           Cuda device to use
```


```
❯ mat6115 artifacts -h
usage: mat6115 artifacts [-h] -l LOAD_PATH [-d {imdb}]
                         [-e {charngram.100d,fasttext.en.300d,fasttext.simple.300d,glove.42B.300d,glove.840B.300d,glove.twitter.27B.25d,glove.twitter.27B.50d,glove.twitter.27B.100d,glove.twitter.27B.200d,glove.6B.50d,glove.6B.100d,glove.6B.200d,glove.6B.300d}]
                         [--rnn_layer RNN_LAYER] [--fixed_point] [--cuda CUDA] [--unique_fixed_point] [--pca]

optional arguments:
  -h, --help            show this help message and exit
  -l LOAD_PATH, --load_path LOAD_PATH
                        Path to the saved model. The artifacts will also be saved there.
  -d {imdb}, --dataset {imdb}
                        Supported dataset: imdb
  -e {charngram.100d,fasttext.en.300d,fasttext.simple.300d,glove.42B.300d,glove.840B.300d,glove.twitter.27B.25d,glove.twitter.27B.50d,glove.twitter.27B.100d,glove.twitter.27B.200d,glove.6B.50d,glove.6B.100d,glove.6B.200d,glove.6B.300d}, --embedding {charngram.100d,fasttext.en.300d,fasttext.simple.300d,glove.42B.300d,glove.840B.300d,glove.twitter.27B.25d,glove.twitter.27B.50d,glove.twitter.27B.100d,glove.twitter.27B.200d,glove.6B.50d,glove.6B.100d,glove.6B.200d,glove.6B.300d}
                        Initial embedding to use for words
  --rnn_layer RNN_LAYER
                        What RNN layer to use to create the artifacts
  --fixed_point
  --cuda CUDA           Cuda device to use
  --unique_fixed_point
  --pca
```

### Thanks to

[Original Fixed Point Finder](https://github.com/mattgolub/fixed-point-finder)

[Partial Pytorch port](https://github.com/tripdancer0916/pytorch-fixed-point-analysis)

[Tutorial on using TorchText](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb)
