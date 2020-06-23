from EasyCamera import Camera, CameraType, PipelineType
from EasyAI import AI, TPUType, ModelType
from EasyStreaming.StreamingServer import StreamingServer
from EasyStreaming.overlay import DetectConvert, ClassifyConvert

from time import sleep
import json

#These three lines need to constantly check for changes to JSON file, currently only checks once on startup
f = open('/home/mendel/EasyCoral/Easy_Coral/EasyStreaming/assets/json/enableAI.json')
inefficient = json.load(f)
enable_AI = inefficient["status"]

reef_server = StreamingServer()
csi_cam = Camera(device_path=CameraType.CSI)
dev_board = AI(tpu_path=TPUType.DEVBOARD)

def array_to_svg(ai_data):
    global reef_server
    svg = DetectConvert.convert(ai_data)
    reef_server.send_overlay(svg) #gives svg to server
    #print(ai_data)

csi_H264 = csi_cam.add_pipeline(size=(640, 480), frame_rate=30, pipeline_type=PipelineType.H264) #INPUT: None OUTPUT: H264 Frame
csi_H264.add_listener(reef_server.write_csi) #send H264 Frame to server

if enable_AI == "on":
    FRC_csi = dev_board.add_model(ModelType.detectFRC) #INPUT: RGB Frame OUTPUT: AI data
    csi_rgb = csi_cam.add_pipeline(size=FRC_csi.res, frame_rate=30, pipeline_type=PipelineType.RGB) #INPUT: None OUTPUT: RGB Frame
    csi_rgb.add_listener(FRC_csi.data) #send RGB Frame to face ai engine
    FRC_csi.add_listener(array_to_svg) #send AI data to array to svg function

csi_cam.start()

while True:
    pass
    #print(csi_rgb.fps)
    sleep(0.033)