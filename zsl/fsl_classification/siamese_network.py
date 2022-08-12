from torch import nn, abs, flatten, sigmoid

from .utils_sn import *

enter = 2048
hidden = 2000

class Siamese(nn.Module):

  """
  I1 --> CNN --> u
                  \
                    |u-v| --> NN --> x --> s(x) --> L(.,.)
                  /
  I2 --> CNN --> v
  """

  def __init__(self, modules, metric, cuda_):
    super(Siamese, self).__init__()
    
    self.cuda_ = cuda_
    self.backbone = nn.Sequential(*modules).cuda() if cuda_ else nn.Sequential(*modules)
    self.metric = metric.cuda() if cuda_ else metric
    self.combination = lambda u, v: abs(u-v)      

  def get_vector(self, image):
    image = image.cuda() if self.cuda_ else image
    return flatten(self.backbone(image))

  def createCombinedVector(self, I1, I2):
    u = self.get_vector(I1)
    v = self.get_vector(I2)
    return self.combination(u, v), u, v 

  def forward(self, w):
    out = self.metric(w)
    out_normalized = sigmoid(out)

    return out_normalized