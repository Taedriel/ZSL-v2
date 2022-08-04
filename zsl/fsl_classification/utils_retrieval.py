from urllib import request
import tqdm
from os import makedirs
from shutil import rmtree
from selenium import webdriver
from bs4 import BeautifulSoup 

from .constants import *

def getParser(classeName):

  site = 'https://www.google.com/search?tbm=isch&q='+classeName

  chrome_options = webdriver.ChromeOptions()
  chrome_options.add_argument('--headless')
  chrome_options.add_argument('--no-sandbox')
  chrome_options.add_argument('--disable-dev-shm-usage')

  driver = webdriver.Chrome('chromedriver', options=chrome_options)
  driver.get(site)
  driver.execute_script("window.scrollBy(0, document.body.scrollHeight)")
  soup = BeautifulSoup(driver.page_source, 'html.parser')
  driver.close()

  return soup


"""
@desc retrieve 20 images for a specific class

@param path path to the folder that will contain the images
@param classeName name of the class 

@return the number of downloaded images
"""
def getClassImages(path, classeName):
  
  imagesNumber = 0
  soup = getParser(classeName)
  img_tags = soup.find_all("img", class_="rg_i")

  for index in range(0, len(img_tags)):
    try:
        request.urlretrieve(img_tags[index]['src'], path+classeName+"/"+str(classeName+str(index))+".jpg")
        imagesNumber+=1
    except Exception as e:
        pass

  return imagesNumber


def getImagesGoogle(classes):

  imagesNumber = 0
  rmtree(PATH_IMAGES, ignore_errors=False)
  makedirs(PATH_IMAGES)

  print("downloading images...")

  for classe in tqdm(classes):

    try:
      classeName = classe.replace(" ", "")
      makedirs(PATH_IMAGES+classeName)
      imagesNumber += getClassImages(PATH_IMAGES, classeName)
    except:
      pass

  return imagesNumber


def getClassesImagesURLLIB(classes, download=True):

  imagesNumber = 0

  if download:
    imagesNumber = getImagesGoogle(classes)
  
  print("\n"+str(imagesNumber) + " images were downloaded. " + str(imagesNumber/len(classes)) + " per classes")
