import os 
import logging
import pandas as pd
import random
import numpy as np
from tqdm import tqdm, tqdm_notebook
import matplotlib.pyplot as plt
import pickle
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from typing import List, Tuple, Dict

import torch
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing
from sklearn.preprocessing import LabelEncoder