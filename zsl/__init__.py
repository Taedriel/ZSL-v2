from zsl import hierarchical_solver
from zsl import word_embeddings
from zsl import misc

import logging
import warnings

FORMAT = '%(levelname)-10s %(message)s'
logging.basicConfig(format=FORMAT, level = logging.INFO, filename = "ZSL.log" )

warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')