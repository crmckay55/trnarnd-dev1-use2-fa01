# helper_functions.py
# Chris McKay
# v1.0 2021-11-14
# Wrapper for azure.storage.blob to handle all blob-related tasks
# as well as image conversion, and json parsing

import io
import os
import logging, inspect
import uuid
import tempfile
import json

import numpy as np
from PIL import Image

from . import config as c
from azure.storage.blob import BlobClient


class BlobHandler:
    """
        Wrapper for BlobClient to handle the repetitive calls and app-specific transformations
    """

    def __init__(self, blob=None, extension=None):
        """
            Initialization of basic parameters of the class

            :param blob: path and filename of the blob to load or create.  If empty, randomly assigned
            :param extenstion: leave blank if blob supplied.  otherwise non-dotted extension (e.g. PNG)
        """

        self.__connection = c.SOURCE_CONNECTION
        self.__container = c.SOURCE_CONTAINER
        self.local_path = None
        self.file_contents = None

        if blob == None:
            blob = f'results/{str(uuid.uuid4())}.{extension}'

        self.blob_name = blob
        self.__create_client__()
        logging.info(self.__class__.__name__ + '.' + inspect.stack()[0].function + f': client created for {self.blob_name}')
    
    
    def __create_client__(self):
        """
            internal process to create client object based on supplied environment variables and blob name
        """
        self.client = BlobClient.from_connection_string(
            conn_str=self.__connection,
            container_name=self.__container,
            blob_name=self.blob_name)


    def push_blob(self, content):
        """
            save content to blob storage
            :param content: object holding content (e.g. bytestream...I think?)
        """
        try:  # will throw if file exists already
            self.client.upload_blob(content)
            logging.info(self.__class__.__name__ + '.' + inspect.stack()[0].function + f': successfully uploaded {self.blob_name}')

        except Exception as e:  # delete blob then try again
            self.client.delete_blob()
            self.client.upload_blob(content)
            logging.info(self.__class__.__name__ + '.' + inspect.stack()[0].function + f': successfully uploaded {self.blob_name} after delete of existing')


    def get_blob(self):
        """
            Download the blob
        """
        self.file_contents = self.client.download_blob()
        logging.info(self.__class__.__name__ + '.' + inspect.stack()[0].function + f': successfully downloaded {self.blob_name}')
        
    def get_blob_all(self):
        self.file_contents = self.client.download_blob().readall()
        logging.info(self.__class__.__name__ + '.' + inspect.stack()[0].function + f': successfully downloaded {self.blob_name} of length {str(len(self.file_contents))}')

    def save_local(self, name):
        """
            Well, I can't figureout how to load the .pth model and pass to torch.load() because it takes a path, not content.
            So, I save the file locally (function app) and pass the path to the local storage vs. blob storage
            :param name: the name of the local file to save
            :return: the path/filename.ext of the local file
        """
        
        try:
            self.get_blob_all()
            
            temp_file_path = tempfile.gettempdir()
            temp_path = os.path.join(temp_file_path, name)
            logging.info(self.__class__.__name__ + '.' + inspect.stack()[0].function + f': arranged for path {temp_path}')
        
        except Exception as e:
            logging.error(self.__class__.__name__ + '.' + inspect.stack()[0].function + f': failed during setup to save local file to {temp_path}')

        try:
            with open(temp_path, 'wb') as f:
                f.write(self.file_contents)
            
            size = os.path.getsize(temp_path)
            logging.info(self.__class__.__name__ + '.' + inspect.stack()[0].function + f': file contents saved of size {size}')
            
        except Exception as e:
            logging.error(self.__class__.__name__ + '.' + inspect.stack()[0].function + f': file contents not saved: {e}')

        return temp_path


class ImageObject:
    """
        takes care of any conversions to data and images
    """


    def __init__(self, blob_stream): 
        """
            Converts the blobstream into a cv2 compatible np array.  don't ask me why I load as a PIL Image first...
            :param blob_stream: the loaded blob object
        """
        # open filestream and convert to cv2 picture array
        self.image = Image.open(io.BytesIO(blob_stream.content_as_bytes()))
        self.image = np.array(self.image)
        logging.info(self.__class__.__name__ + '.' + inspect.stack()[0].function + f': Converted blob stream into a np array')


    def convert_for_save(self):
        """
            Converts the image back into bytes for saving to a blob
            :return: object that can be saved to a blob
        """
        # converts image to bytes for upload to blob
        image = Image.fromarray(self.image)
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        byte_im = buf.getvalue()
        logging.info(self.__class__.__name__ + '.' + inspect.stack()[0].function + f': Converted cv2 image back into bytestream')

        return byte_im

def make_response(new_path, quantity, status): 
    """
        because I use this in __init__ several times, I make this to reduce the code.
        :return: json object of the approriately formed response to send back through the API to PowerAPP
    """
    response = {"new_path":new_path, "quantity":quantity, "status":status}
    response = json.dumps(response)
    return response