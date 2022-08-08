from .misc import *
from .word_embeddings import *
from .embeddings_solver import *

import logging

FORMAT = '%(levelname)-10s %(message)s'
logging.basicConfig(format=FORMAT, level = logging.INFO, filename = __name__ + ".log" )


NAME = 'ZSL-v2'

__version__ = "0.0.1"
VERSION = __version__

DESCRIPTION = ""

AUTHOR = "Prof. Jean-Christophe Nebel"
AUTHOR_EMAIL = "J.Nebel@kingston.ac.uk"
URL = ""
PROJECT_URLS = {
    'Documentation': "",
    'Source Code': "https://github.com/Taedriel/ZSL-v2",
}
LICENSE = ""

KEYWORDS = ["image-recognition", "zero shot learning", "few shot learning"]