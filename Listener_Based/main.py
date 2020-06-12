from EasyCamera import Camera, CameraType, PipelineType
from EasyAI import AI, TPUType, ModelType
from EasyStreaming.EasyServer import EasyServer, ServerType

from time import sleep

reef_server = EasyServer(server_type=ServerType.Reef)
csi_cam = Camera(device_path=CameraType.CSI)
dev_board = AI(tpu_path=TPUType.DEVBOARD)

def array_to_svg(ai_data):
    global reef_server
    svg = convert(ai_data)
    reef_server.aidata(svg)
    print(ai_data)

face_csi = dev_board.add_model(ModelType.detectFace) #INPUT: RGB Frame OUTPUT: AI data
csi_H264 = csi_cam.add_pipeline(size=(640, 480), frame_rate=30, pipeline_type=PipelineType.H264) #INPUT: None OUTPUT: H264 Frame
csi_rgb = csi_cam.add_pipeline(size=face_csi.res, frame_rate=30, pipeline_type=PipelineType.RGB) #INPUT: None OUTPUT: RGB Frame


csi_H264.add_listener(reef_server.data) #send H264 Frame to server
csi_rgb.add_listener(face_csi.data) #send RGB Frame to face ai engine
face_csi.add_listener(array_to_svg) #send AI data to array to svg function

csi_cam.start()
while True:
    pass
    #print(csi_rgb.fps)
    sleep(0.033)