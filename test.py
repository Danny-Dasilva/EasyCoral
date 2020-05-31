from CameraManager.TPUCameraManager import CameraManager, GStreamerPipelines
from AIManager.AIManager import AIManager, AIModels
from edgetpu.detection.engine import DetectionEngine
camMan = CameraManager()
aiMan = AIManager()

CSICam = camMan.newCam(0)
USBCam = camMan.newCam(1)

FRCCSI = CSICam.addPipeline(GStreamerPipelines.RGB,AIModels.detectFRC["size"],30,"aisink")
FaceCSI = CSICam.addPipeline(GStreamerPipelines.RGB,AIModels.detectFace["size"],30,"ai2sink")
RandomCSI = CSICam.addPipeline(GStreamerPipelines.RGB,AIModels.classifyRandom["size"],30,"ai3sink")

FRCUSB = USBCam.addPipeline(GStreamerPipelines.RGB,AIModels.detectFRC["size"],30,"aiusbsink")
FaceUSB = USBCam.addPipeline(GStreamerPipelines.RGB,AIModels.detectFace["size"],30,"ai2usbsink")
RandomUSB = USBCam.addPipeline(GStreamerPipelines.RGB,AIModels.classifyRandom["size"],30,"ai3usbsink")

CSICam.startPipeline()
USBCam.startPipeline()

def run_detect(frame, engine, labels):
        objs = engine.detect_with_image(frame)#add arguments
        tempArray = []
        for obj in objs:
            tempArray.append({"box":obj.bounding_box.flatten().tolist(),"score":obj.score,"label":labels[obj.label_id]})
        return(tempArray)

while True:
    #aiMan.analyze_frame(AIModels.detectFRC, FRCCSI.getImage(), "FRCCSI") if(FRCCSI) else None
    #aiMan.analyze_frame({"modelType":"detect","engine":DetectionEngine,"path":"/home/mendel/EasyCoral/AIManager/models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite","label":"/home/mendel/EasyCoral/AIManager/models/face_labels.txt","size":(640,480),"runFunc":run_detect}, FaceCSI.getImage(), "FaceCSI") if(FaceCSI) else None
    #aiMan.analyze_frame(AIModels.classifyRandom, RandomCSI.getImage(), "RandomCSI") if(RandomCSI) else None

    #aiMan.analyze_frame(AIModels.detectFRC, FRCUSB.getImage(), "FRCUSB") if(FRCUSB) else None
    aiMan.analyze_frame(AIModels.detectFace, FaceUSB.getImage(), "FaceUSB") if(FaceUSB) else None
    #aiMan.analyze_frame(AIModels.classifyRandom, RandomUSB.getImage(), "RandomUSB") if(RandomUSB) else None

    print(aiMan.getData()) if(aiMan) else None