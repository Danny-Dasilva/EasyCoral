from threading import Thread
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.detection.engine import DetectionEngine
import edgetpu
import time
from time import sleep
import re
import os
dirname, filename = os.path.split(os.path.abspath(__file__))

class AI:
    def __init__(self, tpu_path):
        self.tpu_path = tpu_path
        self.models = []
        self.engines = {}
        self.labels = {}
        self.run_thread = Thread(target=self.run_models)
        self.run_thread.daemon = True
        self.run_thread.start()

    def add_model(self, model_type):
        engine, label = self.create_engine(model_type)
        model = AIModel(model_type, engine, label)
        self.models.append(model)
        return model

    def load_labels(self,path):
        LABEL_PATTERN = re.compile(r'\s*(\d+)(.+)')
        with open(path, 'r', encoding='utf-8') as f:
            lines = (LABEL_PATTERN.match(line).groups() for line in f.readlines())
            return {int(num): text.strip() for num, text in lines}
    
    def create_engine(self,model_type):
        if model_type["path"] not in self.engines.keys():
            self.engines[model_type["path"]] = model_type["engine"](model_type["path"], self.tpu_path)
            self.labels[model_type["path"]] = self.load_labels(model_type["label"])
        return(self.engines[model_type["path"]], self.labels[model_type["path"]])
    
    def run_models(self):
        while True:
            for model in self.models:
                model.run()
            sleep(0.0001)

class AIModel():
    def __init__(self, model_type, engine, label):
        self.labels = label
        self.res = model_type["size"]
        self.engine = engine
        self.frame = None
        self.listeners = []
        self.run_func = model_type["runFunc"]

    def add_listener(self, func):
        self.listeners.append(func)

    def remove_listener(self, func):
        target = 0
        for idx, listener in enumerate(self.listeners):
            if(listener == func):
                target = idx
        del(self.listeners[target])
    
    def remove_listener(self, func):
        target = 0
        for idx, listener in enumerate(self.listeners):
            if(listener == func):
                target = idx
        del(self.listeners[target])
    
    def data(self, data):
        self.frame = data

    def run(self):
        if self.frame is not None:
            results = self.run_func(self.frame, self.engine, self.labels)
            self.frame = None
            self.send_data(results)

    def send_data(self, data):
        for listener in self.listeners:
            listener(data)
    
class ModelType:
    def run_classify(self, frame, engine, labels):
        start = time.monotonic()
        objs = engine.classify_with_input_tensor(frame)#add arguments
        inference_time = time.monotonic() - start
        tempArray = []
        for obj in objs:
            tempArray.append({"score":obj[1],"label":labels[obj[0]],"inference_time":inference_time})
        return(tempArray)
        
    def run_detect(self, frame, engine, labels):
        start = time.monotonic()
        objs = engine.detect_with_input_tensor(frame)
        inference_time = time.monotonic() - start
        tempArray = []
        for obj in objs:
            tempArray.append({"box":obj.bounding_box.flatten().tolist(),"score":obj.score,"label":labels[obj.label_id],"inference_time":inference_time})
        return(tempArray)

    detectFace = {"modelType":"detect","engine":DetectionEngine,"path":f"{dirname}/models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite","label":f"{dirname}/models/face_labels.txt","size":(320,320),"runFunc":run_detect}
    detectFRC = {"modelType":"detect","engine":DetectionEngine,"path":f"{dirname}/models/mobilenet_v2_edgetpu_red.tflite","label":f"{dirname}/models/field_labels.txt","size":(300,300),"runFunc":run_detect}
    classifyRandom = {"modelType":"classify","engine":ClassificationEngine,"path":f"{dirname}/models/mobilenet_v2_1.0_224_quant_edgetpu.tflite","label":f"{dirname}/models/imagenet_labels.txt","size":(224,224),"runFunc":run_classify}


# class DetectFace(ModelType):
#     modelType = "detect"
#     engine = DetectionEngine
#     path = f"{dirname}/models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite"
#     label = f"{dirname}/models/face_labels.txt"
#     size = (320,320)
#     runFunc = ModelType.run_detect


# class detectFRC(ModelType):
#     modelType = "detect"
#     engine = DetectionEngine
#     path = f"{dirname}/models/mobilenet_v2_edgetpu_red.tflite"
#     label = f"{dirname}/models/field_labels.txt"
#     size = (300,300)
#     runFunc = ModelType.run_detect

class TPUType:
    DEVBOARD = "/dev/apex_0"