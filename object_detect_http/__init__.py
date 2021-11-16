# __init__.py
# Chris McKay
# v1.0 2021-11-14
# entry point for the http trigger, parsing the path to the source image and 
# starting the object detection processs.

import logging

import azure.functions as func

from .src import process
from .src import helper_functions


def main(req: func.HttpRequest) -> func.HttpResponse:
    
    logging.info('image_recognizer_http: function app triggered')
    
    image = req.params.get('image')
    if not image:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            image = req_body.get('image')

    if image:
        logging.info(f'image_recognizer_http: Found image with name {image}')
        try:
            # Pass in the image name, and get new image name back, with number of objects identified
            new_path, quantity, status = process.inference_image(image)  
            logging.info(f'returned to __init__ with {new_path}, {quantity}, {status}')
            
            json_response = helper_functions.make_response(new_path, quantity, status)

        except Exception as e:
            json_response = helper_functions.make_response(image, 0, f"image {image} passed in but function failed: {e}")
            
        return func.HttpResponse(json_response)

    else:
        logging.error(f'found no image in {image}')
        
        json_response = helper_functions.make_response(image, 0, "no image name passed in")
        return func.HttpResponse(json_response)


