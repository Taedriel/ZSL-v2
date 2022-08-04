from torchvision import transforms
import PIL.Image
from random import randint

def getNrandomClassesPaths(path, listClasses, n_way):

  if n_way == len(listClasses):
    return listClasses

  paths = []
  for i in range(0, n_way):
    ridx = randint(0, len(listClasses)-1)
    paths.append(path+listClasses[ridx]+"/")
    listClasses.remove(listClasses[ridx])

  return paths

imageResize = (224, 224)


def getImageTensor(path, conversion_type):

  convert = None
  if conversion_type == "CUBLike":
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

