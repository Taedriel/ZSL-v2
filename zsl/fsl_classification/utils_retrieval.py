from urllib import request
from tqdm import tqdm
from os import makedirs
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup 
from chromedriver_py import binary_path 
import time

from .constants import *

def getParser(classeName):

  site = 'https://www.google.com/search?tbm=isch&q='+classeName

  chrome_options = webdriver.ChromeOptions()
  chrome_options.add_argument('--headless')
  chrome_options.add_argument('--no-sandbox')
  chrome_options.add_argument('--disable-dev-shm-usage')

  service_object = Service(binary_path)
  driver = webdriver.Chrome('chromedriver', options=chrome_options, service=service_object)
  driver.get(site)
  time.sleep(2)

  # accept cookies
  try:
    driver.find_element(By.CLASS_NAME, "VfPpkd-LgbsSe.VfPpkd-LgbsSe-OWXEXe-k8QpJ.VfPpkd-LgbsSe-OWXEXe-dgl2Hf.nCP5yc.AjY5Oe.DuMIQc.LQeN7.Nc7WLe").submit()
  except:
    pass

  time.sleep(3)
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
  print("downloading images...\n")
  for classe in tqdm(classes):

    try:
      classeName = classe.replace(" ", "")
      makedirs(PATH_IMAGES+classeName)
      imagesNumber += getClassImages(PATH_IMAGES, classeName)
    except Exception as e:
      pass

  return imagesNumber


def getClassesImagesURLLIB(classes, download=True):

  imagesNumber = 0

  if download:
    imagesNumber = getImagesGoogle(classes)
  
  print("\n"+str(imagesNumber) + " images were downloaded. " + str(imagesNumber/len(classes)) + " per classes", "\n")
