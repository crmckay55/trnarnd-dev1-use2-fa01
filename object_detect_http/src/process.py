import logging, inspect

from . import config
from . import helper_functions as hf
from . import inference


def inference_image(img):
    
    source = hf.BlobHandler(blob=img)
    source.get_blob()
    logging.info(inspect.stack()[0].function + f': opened source image from blob with {source.blob_name} of length {str(len(source.file_contents))}')

    sink = hf.BlobHandler(extension='PNG')
    logging.info(inspect.stack()[0].function + f': opened sink with filename {sink.blob_name}')

    model = hf.BlobHandler(config.MODEL_LOCATION + config.MODEL_NAME)
    local_model_path = model.save_local('model.pth')
    logging.info(inspect.stack()[0].function + f': loaded model {config.MODEL_LOCATION}{config.MODEL_NAME} and saved to {local_model_path}')

    # convert raw blob input to cv2 image
    image_to_process = hf.ImageObject(source.file_contents)
    logging.info(inspect.stack()[0].function + f': converted image from raw blob to cv2')

    status='successful '
    try:
        image_to_process.image, number_of_objects = inference.detect_objects(image_to_process.image, local_model_path)
    except Exception as e:
        number_of_objects = 0
        status = str(e)
    
    logging.info(inspect.stack()[0].function + f': inference complete, found {str(number_of_objects)} objects')
    
    bytes_for_save = image_to_process.convert_for_save()
    sink.push_blob(bytes_for_save)
    logging.info(inspect.stack()[0].function + f': saved final image, {sink.blob_name}, {number_of_objects}, {status}')

    return sink.blob_name, number_of_objects, status