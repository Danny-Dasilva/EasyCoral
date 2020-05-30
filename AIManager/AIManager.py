from edgetpu.classification.engine import ClassificationEngine
from edgetpu.detection.engine import DetectionEngine
import edgetpu
from threading import Thread
import re

class AIManager():
    def __init__(self):
        self.ai_type = "detection" #not always true
        self.engines = {}
        self.frameBuffer = {}
        self.enabled = True
        self.t = Thread(target=self.run_models)
        self.t.start()
        self.new_data = False
        self.dataBuffer = []
        self.labelsArray = {}

    def load_labels(self,path):
        LABEL_PATTERN = re.compile(r'\s*(\d+)(.+)')
        with open(path, 'r', encoding='utf-8') as f:
            lines = (LABEL_PATTERN.match(line).groups() for line in f.readlines())
            return {int(num): text.strip() for num, text in lines}
    
    def analyze_frame(self,model_type,frame,tag):
        if(self.enabled):
            self.create_engine(model_type)
            self.frameBuffer[tag] = (frame, model_type)

    def create_engine(self,model_type):
        if model_type["path"] not in self.engines.keys():
            self.engines[model_type["path"]] = model_type["engine"](model_type["path"],'/dev/apex_0')
            self.labelsArray[model_type["path"]] = self.load_labels(model_type["label"])

    def run_models(self):
        while True:
            keys = list(self.frameBuffer)
            if keys:
                key = keys[0]
                frame, model_type = self.frameBuffer[key]
                if(model_type["modelType"]=="detect"):
                    objs = self.engines[model_type["path"]].detect_with_image(frame)#add arguments
                    labels = self.labelsArray[model_type["path"]]
                    self.new_data = True
                    del self.frameBuffer[key]
                    tempArray = []
                    for obj in objs:
                        tempArray.append({"box":obj.bounding_box.flatten().tolist(),"score":obj.score,"label":labels[obj.label_id]})
                    tempDict = {}
                    tempDict[key] = tempArray 
                    self.dataBuffer.append(tempDict)
                if(model_type["modelType"]=="classify"):
                    objs = self.engines[model_type["path"]].classify_with_image(frame)#add arguments
                    labels = self.labelsArray[model_type["path"]]
                    self.new_data = True
                    del self.frameBuffer[key]
                    tempArray = []
                    for obj in objs:
                        tempArray.append({"score":obj[1],"label":labels[obj[0]]})
                    tempDict = {}
                    tempDict[key] = tempArray 
                    self.dataBuffer.append(tempDict)
                
    def __bool__(self):
        return self.new_data

    def getData(self):
        self.new_data = False
        if(self.dataBuffer):
            data = self.dataBuffer[0]
            del self.dataBuffer[0]
            return(data)

class AIModels:
    detectFace = {"modelType":"detect","engine":DetectionEngine,"path":"/home/mendel/EasyCoral/AIManager/models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite","label":"/home/mendel/EasyCoral/AIManager/models/face_labels.txt","size":(640,480)}
    detectFRC = {"modelType":"detect","engine":DetectionEngine,"path":"/home/mendel/EasyCoral/AIManager/models/mobilenet_v2_edgetpu_red.tflite","label":"/home/mendel/EasyCoral/AIManager/models/field_labels.txt","size":(600,600)}
    classifyRandom = {"modelType":"classify","engine":ClassificationEngine,"path":"/home/mendel/EasyCoral/AIManager/models/mobilenet_v2_1.0_224_quant_edgetpu.tflite","label":"/home/mendel/EasyCoral/AIManager/models/imagenet_labels.txt","size":(640,480)}
