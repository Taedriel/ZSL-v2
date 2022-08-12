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

def get_parser(classe_name : str) -> BeautifulSoup:

  site = 'https://www.google.com/search?tbm=isch&q='+classe_name

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


def get_class_images(path : str, classe_name : List[str]) -> int:
  """
  retrieve 20 images for a specific class

  Parameters
  ----------
  path :
    path to the folder that will contain the images
  classe_name : 
    name of the class 

  Return
  ------
  the number of downloaded images
  """
  
  images_number = 0
  soup = get_parser(classe_name)
  img_tags = soup.find_all("img", class_="rg_i")

  for index in range(0, len(img_tags)):

    try:
      request.urlretrieve(img_tags[index]['src'], path+classe_name+"/"+str(classe_name+str(index))+".jpg")
      images_number+=1
    except Exception as e:
      pass

  return images_number


def get_images_from_google(classes : List[str]) -> int:

  images_number = 0
  print("downloading images...\n")
  for classe in tqdm(classes):

    try:
      classe_name = classe.replace(" ", "")
      makedirs(PATH_IMAGES+classe_name)
      images_number += get_class_images(PATH_IMAGES, classe_name)
    except Exception as e:
      pass

  return images_number


def get_classes_images_URLLIB(classes : List[str], download=True):

  images_number = 0

  if download:
    images_number = get_images_from_google(classes)
  
  print("\n"+str(images_number) + " images were downloaded. " + str(images_number/len(classes)) + " per classes", "\n")
