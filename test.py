from CameraManager.TPUCameraManager import CameraManager, GStreamerPipelines
from AIManager.AIManager import AIManager, AIModels

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

while True:
    aiMan.analyze_frame(AIModels.detectFRC, FRCCSI.getImage(), "FRCCSI") if(FRCCSI) else None
    aiMan.analyze_frame(AIModels.detectFace, FaceCSI.getImage(), "FaceCSI") if(FaceCSI) else None
    aiMan.analyze_frame(AIModels.classifyRandom, RandomCSI.getImage(), "RandomCSI") if(RandomCSI) else None

    aiMan.analyze_frame(AIModels.detectFRC, FRCUSB.getImage(), "FRCUSB") if(FRCUSB) else None
    aiMan.analyze_frame(AIModels.detectFace, FaceUSB.getImage(), "FaceUSB") if(FaceUSB) else None
    aiMan.analyze_frame(AIModels.classifyRandom, RandomUSB.getImage(), "RandomUSB") if(RandomUSB) else None

    print(aiMan.getData()) if(aiMan) else None