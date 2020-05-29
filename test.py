from CameraManager.TPUCameraManager import CameraManager, GStreamerPipelines

camMan = CameraManager()
CSICam = camMan.newCam(0)
USBCam = camMan.newCam(1)
USB2Cam = camMan.newCam(2)
detect = detectAI()
AICSI = CSICam.addPipeline(GStreamerPipelines.RGB,detect.size,30,"aisink")
AIUSB = USBCam.addPipeline(GStreamerPipelines.RGB,detect.size,30,"aisink")
s
H264 = CSICam.addPipeline(GStreamerPipelines.H264,(640,480),30,"h264sink")
CSICam.startPipeline()
while True:
    if(H264):
        server.updateFrame(H264)
        print(bytes(H264))
    if(AICSI):
        detect.newFrame(AICSI,"CSI")
    if(AIUSB):
        detect.newFrame(AIUSB,"")
    if(detect.resp):
        print(detect.resp)