import numpy as np
import cv2
import torch
import glob as glob
import logging, inspect
from .model import create_model
from . import config


def detect_objects(image, model_path):

    # set the computation device
    device = torch.device(config.COMPUTATION_DEVICE)
    logging.info(inspect.stack()[0].function + f': device set')
    

    # load the model and the trained weights
    model = create_model(num_classes=config.NUM_CLASSES).to(device)
    logging.info(inspect.stack()[0].function + f': created model, now loading {model_path}')

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(inspect.stack()[0].function + f': loaded model dict')

    except Exception as e:
        logging.error(inspect.stack()[0].function + f': error at model load: {e}')

    try:
        model.eval()
        logging.info(inspect.stack()[0].function + f': evaluated model')
    
    except Exception as e:
        logging.error(inspect.stack()[0].function + f': error at model eval: {e}')

    detection_threshold = 0.8

    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float)
    logging.info(inspect.stack()[0].function + f': adjusted image')

    try:
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cpu()  # INFO: changed from .cuda to .cpu
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image)

        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        logging.info(inspect.stack()[0].function + f': finished evaluating and moving to drawing boxes')

        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()

            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()

            # get all the predicited class names
            pred_classes = [config.CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]      # INFO: left as .cpu

            logging.info(inspect.stack()[0].function + f': all box preamble complete')

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 2)
                cv2.putText(orig_image, pred_classes[j] + ': ' + str('{:.2%}'.format(scores[j])),        # INFO: added score to box, increased font size, formatted percent
                            (int(box[0]), int(box[1] - 5)),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, .4, (0, 255, 0),
                            1, lineType=cv2.LINE_4)
            
            logging.info(inspect.stack()[0].function + f': box drawing complete')
        
        else:
            draw_boxes = []

    except Exception as e:
        draw_boxes = []
        logging.error(inspect.stack()[0].function + f': error after model load: {e}')

    # return the updated image, and the number of boxes drawn
    return orig_image, len(draw_boxes)
