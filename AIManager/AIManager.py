from edgetpu.classification.engine import ClassificationEngine
from edgetpu.detection.engine import DetectionEngine
import edgetpu
from threading import Thread
import re
import time
from codetiming import Timer
t = Timer()
class AI():
    def __init__(self, tpu):
        self.tpu = tpu
        self.engines = {}
        self.labels = {}
        self.data_classes = []
        self.t = Thread(target=self.run_models)
        self.t.daemon = True
        self.t.start()

    def add(self, model_type, camera):
        self.create_engine(model_type)
        pipe = camera.add(size=(model_type["size"][0], model_type["size"][1]), frame_rate=30)
        data_class = AIData(pipe, model_type, self)
        self.data_classes.append(data_class)
        return(data_class)

    def load_labels(self,path):
        LABEL_PATTERN = re.compile(r'\s*(\d+)(.+)')
        with open(path, 'r', encoding='utf-8') as f:
            lines = (LABEL_PATTERN.match(line).groups() for line in f.readlines())
            return {int(num): text.strip() for num, text in lines}
    
    def create_engine(self,model_type):
        if model_type["path"] not in self.engines.keys():
            self.engines[model_type["path"]] = model_type["engine"](model_type["path"], self.tpu)
            self.labels[model_type["path"]] = self.load_labels(model_type["label"])

    def run_models(self):
        while True:
            for data_class in self.data_classes:
                if(data_class.new_frame()):
                    data_class.analyze_frame()
            time.sleep(0.0001)


class AIData:
    def __init__(self, pipeline, model_type, AI_class):
        self.pipeline = pipeline
        self.model_type = model_type
        self.engine = AI_class.engines[model_type["path"]]
        self.labels = AI_class.labels[model_type["path"]]
        self.run = model_type["runFunc"]
        self.is_data = False
        self.data = None
        self.raw_data = None

    def __bool__(self):
        return self.is_data

    def new_frame(self):
        return bool(self.pipeline)

    def analyze_frame(self):
        raw_results = results = self.run(self.pipeline.image(), self.engine, self.labels)
        self.is_data = True
        self.data = results
        self.raw_data = raw_results

    def raw(self):
        self.is_data = False
        return(self.raw_data)

    def array(self):
        self.is_data = False
        return(self.data)

    def svg(self):
        self.is_data = False
        return "in progress"


class AIModels:
    def run_classify(frame, engine, labels):
            objs = engine.classify_with_input_tensor(frame)#add arguments
            tempArray = []
            for obj in objs:
                tempArray.append({"score":obj[1],"label":labels[obj[0]]})
            return(objs,tempArray)
        
    def run_detect(frame, engine, labels):
        objs = engine.detect_with_input_tensor(frame)
        tempArray = []
        for obj in objs:
            tempArray.append({"box":obj.bounding_box.flatten().tolist(),"score":obj.score,"label":labels[obj.label_id]})
        return(objs,tempArray)
    
    detectFace = {"modelType":"detect","engine":DetectionEngine,"path":"/home/mendel/EasyCoral/AIManager/models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite","label":"/home/mendel/EasyCoral/AIManager/models/face_labels.txt","size":(320,320),"runFunc":run_detect}
    detectFRC = {"modelType":"detect","engine":DetectionEngine,"path":"/home/mendel/EasyCoral/AIManager/models/mobilenet_v2_edgetpu_red.tflite","label":"/home/mendel/EasyCoral/AIManager/models/field_labels.txt","size":(300,300),"runFunc":run_detect}
    classifyRandom = {"modelType":"classify","engine":ClassificationEngine,"path":"/home/mendel/EasyCoral/AIManager/models/mobilenet_v2_1.0_224_quant_edgetpu.tflite","label":"/home/mendel/EasyCoral/AIManager/models/imagenet_labels.txt","size":(224,224),"runFunc":run_classify}