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
        self.frameBuffer = {}
        self.enabled = True
        self.t = Thread(target=self.run_models)
        self.t.daemon = True
        self.t.start()
        self.new_data = False
        self.dataBuffer = {}
        self.labelsArray = {}
        self.tag_model_type = {}
        self.data_classes = []
        
    def add(self, model_type, camera):
        self.create_engine(model_type)
        pipe = camera.add(pipeline_type=Pipelines.RGB, size=(model_type["size"][0], model_type["size"][1]), frame_rate=30)
        data_class = AIData(pipe, model_type)
        self.data_classes.append(data_class)
        return(data_class)

    def load_labels(self,path):
        LABEL_PATTERN = re.compile(r'\s*(\d+)(.+)')
        with open(path, 'r', encoding='utf-8') as f:
            lines = (LABEL_PATTERN.match(line).groups() for line in f.readlines())
            return {int(num): text.strip() for num, text in lines}
    
    # def add_camera_pipeline(self, model_type, tag):
    #     self.create_engine(model_type)
    #     self.tag_model_type[tag] = model_type

    # def analyze_pipeline_frame(self, frame, tag):
    #     if(self.enabled):
    #         self.frameBuffer[tag] = (frame, self.tag_model_type[tag])
    
    # def analyze_frame(self,model_type,frame,tag):
    #     if(self.enabled):
    #         self.create_engine(model_type)
    #         self.frameBuffer[tag] = (frame, model_type)
    
    def create_engine(self,model_type):
        if model_type["path"] not in self.engines.keys():
            self.engines[model_type["path"]] = model_type["engine"](model_type["path"],self.tpu)
            self.labelsArray[model_type["path"]] = self.load_labels(model_type["label"])

    def run_models(self):
        while True:
            keys = list(self.frameBuffer)
            if keys:
                key = keys[0]
                frame, model_type = self.frameBuffer[key]
                results = model_type["runFunc"](frame, self.engines[model_type["path"]], self.labelsArray[model_type["path"]])
                self.new_data = True
                del self.frameBuffer[key]
                self.dataBuffer[key] = results
            time.sleep(0.0001)
            
    def data(self, tag):
        if tag in self.dataBuffer.keys():
            return True
        else:
            return False

    def get_data(self, tag):
        data = self.dataBuffer[tag]
        del self.dataBuffer[tag]
        return(data)


class AIData:
    def __init__(self, pipeline, model_type):
        self.pipeline = pipeline
        self.model_type = model_type

    def __bool__(self):
        return bool(self.pipeline)

    def get_frame(self):
        return(self.model_type, self.pipeline.image)

class AIModels:
    def run_classify(frame, engine, labels):
            objs = engine.classify_with_input_tensor(frame)#add arguments
            tempArray = []
            for obj in objs:
                tempArray.append({"score":obj[1],"label":labels[obj[0]]})
            return(tempArray)
        
    def run_detect(frame, engine, labels):
        #t.start()
        objs = engine.detect_with_input_tensor(frame)
        #t.stop()
        tempArray = []
        for obj in objs:
            tempArray.append({"box":obj.bounding_box.flatten().tolist(),"score":obj.score,"label":labels[obj.label_id]})
        return(tempArray)
    
    detectFace = {"modelType":"detect","engine":DetectionEngine,"path":"/home/mendel/EasyCoral/AIManager/models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite","label":"/home/mendel/EasyCoral/AIManager/models/face_labels.txt","size":(320,320),"runFunc":run_detect}
    detectFRC = {"modelType":"detect","engine":DetectionEngine,"path":"/home/mendel/EasyCoral/AIManager/models/mobilenet_v2_edgetpu_red.tflite","label":"/home/mendel/EasyCoral/AIManager/models/field_labels.txt","size":(300,300),"runFunc":run_detect}
    classifyRandom = {"modelType":"classify","engine":ClassificationEngine,"path":"/home/mendel/EasyCoral/AIManager/models/mobilenet_v2_1.0_224_quant_edgetpu.tflite","label":"/home/mendel/EasyCoral/AIManager/models/imagenet_labels.txt","size":(224,224),"runFunc":run_classify}