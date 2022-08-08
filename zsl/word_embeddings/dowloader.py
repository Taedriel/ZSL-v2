from os.path import exists, abspath
from os import system

import logging
log = logging.getLogger(__name__)

__all__ = ["Downloader"]

class Downloader:
    """Pre download compressed file, usually pre trained embedding and extract them in order to use them

    If the file is already present, it will not download anything
    """

    def __init__(self, base_addr : str, file_zipname : str):
        """download the file at the given address on internet. the full address is obtained by concatenating the address and the filename 

        Args:
            base_addr (str): the internet address of the server hosting the file
            file_zipname (str): the compressd name of the file. uncompressed name while be the same but without the extension
        """
        self.__address = base_addr
        self.__zip_filename = file_zipname
        self.__unzip_filename = "".join(file_zipname.split(".")[:-1])

    def download(self) -> str:
        """ download the embedding file and return a path to the file.
         If the file is already downloaded, only return the path
         """
        self.path = abspath(self.__unzip_filename)
        log.info(f"Checking the presence of {self.path}...")
        if exists(self.path):
            log.info("File found ! No download needed")
            return self.path

        log.info("File Not present, need to download")
        log.info(f"Starting to download {self.__address}{self.__unzip_filename}")
        try:
            ret = system(f"wget {self.__address}{self.__zip_filename}")
            if ret != 0:
                raise SystemError
        except:
            log.error(f"can't retrieve the file {self.__zip_filename}")
            raise SystemError(f"can't retrieve the file {self.__zip_filename}")

        log.info(f"Download finished")
        log.info(f"Starting unziping the file")

        ext = self.__zip_filename[-4:]
        if ext == ".zip":
            unzip_command = "unzip -p"
        elif ext == ".bz2":
            unzip_command = "bunzip2 -c"

        try:
            ret = system(f"{unzip_command} ./{self.__zip_filename} > {self.__unzip_filename}")
            if ret != 0:
                raise SystemError
        except:
            log.error(f"can't unzip the file {self.__zip_filename}")
            raise SystemError(f"can't unzip the file {self.__zip_filename}")

        log.info(f"Unzipping finished ! ({self.path})")
        return self.path