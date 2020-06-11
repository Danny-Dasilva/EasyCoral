from EasyCamera.EasyCamera import Camera, Pipelines
from AIManager.AIManager import AI, AIModels
from EasyStreaming.EasyServer import EasyServer, ServerType
from time import sleep

CSI = Camera(device=0)
USB = Camera(device=1)
TPU = AI(tpu="/dev/apex_0")

#CSI_H264 = CSI.add(pipeline_type=Pipelines.H264, size=(640,480), frame_rate=30)
#USB_H264 = USB.add(pipeline_type=Pipelines.H264, size=(640,480), frame_rate=30)

CSI_DETECT = TPU.add(model_type=AIModels.detectFace, camera=CSI)
USB_DETECT = TPU.add(model_type=AIModels.detectFace, camera=USB)
# CSI.add_AI(AI_class=AIMan, model_type=AIModels.detectFace, frame_rate=30, tag="csi_face")
# USB.add_AI(AI_class=AIMan, model_type=AIModels.detectFace, frame_rate=30, tag="usb_face")

CSI.start()
USB.start()

reef_server = EasyServer(server_type=ServerType.Reef, csi_h264=CSI_H264, usb_h264=USB_H264)

while True:
    # if(CSI_H264):
    #     print(bytes(CSI_H264))
    if(CSI_DETECT):
        print(CSI_DETECT.array())
    if(USB_DETECT):
        print(USB_DETECT.array())
    # if(CSI.data("csi_face")):
    #     print(CSI.get_image("csi_face"), "CSI Face Data")
    
    # print(CSI.get_fps("csi_face"), "CSI Face")

    # if(AIMan.data("csi_face")):
    #     print(AIMan.get_data("csi_face"), "CSI Face AI")
    # if(AIMan.data("usb_face")):
    #     print(AIMan.get_data("usb_face"), "USB Face AI")
    
    sleep(0.0001)