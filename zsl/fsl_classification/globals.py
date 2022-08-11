from multiprocessing.spawn import import_main_path
from torch import nn
from torchvision.models import resnet50

from .constants import PATH_MODEL
from .siamese_network import *
from .classes_for_model import Trainer

cuda_ = False
backbone = resnet50(pretrained=True)
modules=list(backbone.children())[:-1]

enter = 2048
hidden = 2000

metric_cleaning = nn.Sequential(nn.Linear(enter, hidden), nn.ReLU(),
                               
                               nn.Linear(hidden, hidden), nn.ReLU(),
                               nn.Linear(hidden, hidden), nn.ReLU(),
                               nn.Linear(hidden, hidden), nn.ReLU(),
                               nn.Linear(hidden, hidden), nn.ReLU(),
                               nn.Linear(hidden, hidden), nn.ReLU(),
                               nn.Linear(hidden, hidden), nn.ReLU(),

                               nn.Linear(hidden, 1)) 

metric = nn.Sequential(nn.Linear(enter, hidden), nn.ReLU(),
                       
                       nn.Linear(hidden, hidden), nn.ReLU(),
                       nn.Linear(hidden, hidden), nn.ReLU(),
                       nn.Linear(hidden, hidden), nn.ReLU(),
                       nn.Linear(hidden, hidden), nn.ReLU(),
                       nn.Linear(hidden, hidden), nn.ReLU(),
                       nn.Linear(hidden, hidden), nn.ReLU(),

                       nn.Linear(hidden, 1))

if cuda_:
    model_cleaning = Siamese(modules, metric_cleaning, cuda_).cuda()
    model = Siamese(modules, metric, cuda_).cuda()

else:
    model_cleaning = Siamese(modules, metric_cleaning, cuda_)
    model = Siamese(modules, metric, cuda_)

training_model = Trainer(PATH_MODEL, model, cuda_, False)


