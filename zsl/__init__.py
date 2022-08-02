from hierarchical_solver import *
from word_embeddings import *
from misc import *

import logging
import warnings

FORMAT = '%(levelname)-10s %(message)s'
logging.basicConfig(format=FORMAT, level = logging.INFO, filename = "ZSL.log" )

warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')