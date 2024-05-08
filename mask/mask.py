import numpy as np 
import pandas as pd 
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import skimage.io
import matplotlib.pyplot as plt
import sys, os, random, glob, cv2, math

DATA_DIR = Path('/kaggle/input')
ROOT_DIR = Path('/kaggle/working')
import tensorflow as tf
tf.__version__
