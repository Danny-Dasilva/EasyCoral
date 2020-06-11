from EasyCamera.EasyCamera import Camera, Pipelines
from AIManager.AIManager import AIManager, AIModels
from time import sleep
import os

CSI = Camera(0)
USB = Camera(1)

AIMan = AIManager()

CSI.add(pipeline_type=Pipelines.H264, size=(640,480), frame_rate=30, tag="csi_h264")
USB.add(pipeline_type=Pipelines.H264, size=(640,480), frame_rate=30, tag="usb_h264")

CSI.add_AI(AI_class=AIMan, model_type=AIModels.detectFace, frame_rate=30, tag="csi_face")
USB.add_AI(AI_class=AIMan, model_type=AIModels.detectFace, frame_rate=30, tag="usb_face")

CSI.start()
if os.path.exists(f'/dev/video{str(USB.device)}'):
    USB.start()

while True:

    #if(CSI.data("csi_face")):
    #    print(CSI.get_image("csi_face"), "CSI Face Data")
    
    print(CSI.get_fps("csi_face"), "CSI Face")

    if(AIMan.data("csi_face")):
        print(AIMan.get_data("csi_face"), "CSI Face AI")
    #if(AIMan.data("usb_face")):
    #    print(AIMan.get_data("usb_face"), "USB Face AI")
    
    sleep(0.0001)