# Example file for using pip package

from EasyCamera import Camera, CameraType, PipelineType
from EasyAI import AI, TPUType, ModelType

from time import sleep

csi_cam = Camera(device_path=CameraType.CSI)
dev_board = AI(tpu_path=TPUType.DEVBOARD)

def print_data(sender, ai_data):
    print(ai_data)
    
face_ai = dev_board.add_model(ModelType.DetectFace) #INPUT: RGB Frame OUTPUT: AI data
csi_H264 = csi_cam.add_pipeline(size=(640, 480), frame_rate=30, pipeline_type=PipelineType.H264) #INPUT: None OUTPUT: H264 Frame
csi_rgb = csi_cam.add_pipeline(size=face_ai.res, frame_rate=30, pipeline_type=PipelineType.RGB) #INPUT: None OUTPUT: RGB Frame

csi_rgb += face_ai.process_data #send RGB Frame to ai 
csi_rgb -= face_ai.process_data
face_ai += print_data #send AI data to print function

csi_cam.start()

while True:
    pass
    #print(csi_rgb.fps)
    sleep(0.033)