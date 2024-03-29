from torch import optim
from torch import save, load

from torch import Tensor
from typing import List, Tuple

class ModelUtils:

  def __init__(self):
    pass

  def get_optimizer(self, model):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return optimizer


  def get_images(self, image1 : Tensor, image2 : Tensor) -> Tuple[Tensor, Tensor]:
      
    """
    takes in image tensor and encapsulate them in one more dimension because models want (4, 0) tensor

    Parameters
    ----------
    Image1 / Image2 :
      Image to encapsulate

    Return
    ------
    both image encapsulated
    """

    if image1.dim() < 4:
      image1 = image1.unsqueeze(0)
    
    if image2.dim() < 4:
      image2 = image2.unsqueeze(0)

    return image1, image2

  def reset_model_param(self, model):

    for layer in model.children():
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

    return model


  #TODO CAN CHANGE THE RELOADING MODEL HERE
  def get_model_name(self, context : str, cuda_ : bool):

    valid_context = ["train", "clean"]
    if context in valid_context:
      return "SN6L2k_.pt" if cuda_ else "SN6L2k_noCuda.pt"


class ModelSaver:

  def __init__(self, path : str):
    self.PATH_MODEL = path

  def save_model(self, name : str, model_opti, epoch, loss_value):

    model, optimizer = model_opti[0], model_opti[1]

    save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss_value
    }, self.PATH_MODEL+name)

  def load_model(self, model_name, model, optimizer):
    
    checkpoint = load(self.PATH_MODEL+model_name)
    model.load_state_dict(checkpoint['model_state_dict'])  
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_ = checkpoint['epoch']
    loss_ = checkpoint['loss']

    return (epoch_, loss_), model, optimizer
