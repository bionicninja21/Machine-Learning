import re

# pandas and numpy for dataframes and array manipulations
# tqdm as a progress
# matplotlib for plotting

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
tqdm_notebook().pandas()

from matplotlib import pyplot as plt

# usual PyTorch imports for tensor manipulations, neural networks and data processings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# import some sklearn utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# import keras tokenizing utilities
from keras.preprocessing import text, sequence

# import tensorboardX in case we want to log metrics to tensorboard (requires tensorflow installed - optional)
from tensorboardX import SummaryWriter

from graphviz import Digraph
from torchviz import make_dot


# import spacy for tokenization
import spacy

# fastText is a library for efficient learning of word representations and sentence classification
# https://github.com/facebookresearch/fastText/tree/master/python
# I use it with a pre-trained english embedding that you can fetch from the official website
import fastText