import os, time, json
import cv2
import glob
import sagemaker
import numpy as np
from gluoncv import data, utils
from mxnet import gluon, image, nd
from pprint import pprint


endpoint_key = 'yolo3-mobilenet10-coco-detection-2020-05-25-15-53-18'
predictor = sagemaker.predictor.RealTimePredictor(endpoint=endpoint_key, content_type='image/jpeg')

classes = ['car', 'motorcycle', 'bus', 'truck', 'traffic light']

def detect(pic, predictor):
    """elementary function to send a picture to a predictor"""
    
    with open(pic, 'rb') as image:
        f = image.read()

    tensor = nd.array(json.loads(predictor.predict(f)))
    box_ids, scores, bboxes = tensor[:,:,0], tensor[:,:,1], tensor[:,:,2:]
    return box_ids, scores, bboxes

CONFIDENCE_THRESHOLD = .5

classes = {
    "0": "car",
    "1": "motorcycle",
    "2": "bus",
    "3": "truck",
    "4": "traffic light"
}

def draw_boxes(img, box_ids, scores, bboxes):
    for i in range(len(bboxes[0])):
        box = bboxes[0, i]
        if scores[0, i] >= CONFIDENCE_THRESHOLD:
            coco_class = str(int(box_ids[0, i]))
            if coco_class in classes.keys():
                label = classes[coco_class]
                cv2.rectangle(
                    img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1
                )
                cv2.putText(
                    img,
                    label + " " + str(scores[0, i]),
                    (box[0], box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    lineType=cv2.LINE_AA,
                )
    return img


def img_detect():
    pic_folder = "/home/victor/Data/benchmarks/kitti/data_object_image_2/testing/image_2/"
    img_names = glob.glob(pic_folder + "*.png")
    for img_name in img_names[:10]:
        path = os.path.join(pic_folder, img_name)
        
        start_time = time.time()
        box_ids, scores, bboxes = detect(path, predictor)
        print("--- %s seconds ---" % (time.time() - start_time))
        
        box_ids, scores, bboxes = box_ids.asnumpy(), scores.asnumpy(), bboxes.asnumpy()
        _, img = data.transforms.presets.yolo.load_test(path)
        img = draw_boxes(img, box_ids, scores, bboxes)
        cv2.imshow("image", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    img_detect()