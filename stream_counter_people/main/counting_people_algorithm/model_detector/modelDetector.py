import cv2
import os
# MOBILENET SSD
# list of class labels MobileNet SSD was trained to detect
CLASSES_MobileNet_SSD = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

path_protocol_model_SSD = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"
path_weight_model_SSD="mobilenet_ssd/MobileNetSSD_deploy.caffemodel"

class model_detector:
    def __init__(self,path_protocol,path_weight,classes):
        self.m_path_protocol_model=path_protocol
        self.m_path_weight_model=path_weight
        self.m_classes=classes

    def load_model_detector(self):
        model = cv2.dnn.readNetFromCaffe(os.path.join(os.getcwd(),self.m_path_protocol_model),
                                         os.path.join(os.getcwd(),self.m_path_weight_model))
        return model

    def get_classes(self):
        return self.m_classes