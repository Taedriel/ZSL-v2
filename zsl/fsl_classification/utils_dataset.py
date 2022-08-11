from torchvision import transforms
import PIL.Image
from random import randint

from torch import Tensor
from typing import List

imageResize = (224, 224)

def getNrandomClassesPaths(path : str, listClasses : List[str], n_way : int) -> List[str]:

  paths = []
  for i in range(0, n_way):
    ridx = randint(0, len(listClasses)-1)
    paths.append(path+listClasses[ridx]+"/")
    listClasses.remove(listClasses[ridx])

  return paths

def get_image_tensor(path : str, conversion_type : str) -> Tensor:

  valid_conversion = ["CUB", "IMG"]
  convert = None
  if conversion_type in valid_conversion:
    convert = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0, 0, 0), (1, 1, 1))
                                ])
  if conversion_type == "OMNI":
    convert = transforms.Compose([
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor(),
                                transforms.Normalize((0, 0, 0), (1, 1, 1))
                              ])

  return convert(PIL.Image.open(path).resize(imageResize))

