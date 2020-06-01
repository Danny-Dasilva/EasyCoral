import gi
import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)

import enum
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import os
from threading import Thread
from time import sleep
import time
from PIL import Image

Gst.init(None)

class Camera:
    def __init__(self, device):
        self.device = device
        self.pipeline = str(Pipelines.SRC).format(self.device)
        self.watchdog_thread = Thread(target=self.camera_watchdog)
        self.pipeline_started = False
        self.pipeline_buffer = {}
        self.pipeline_data = {}
        self.pipeline_tags = []
        self.data_call_threads = []
        self.fps = {}
        self.parsed_pipeline = None

    def camera_watchdog(self):
        while True:
            if os.path.exists(f'/dev/video{str(self.device)}'):
                if not self.pipeline_started:
                    self.start_pipeline()
            else:
                if(self.pipeline_started):
                    self.stop_pipeline()
            sleep(0.25)

    def add(self, pipeline_type, size, frame_rate, tag):
        self.pipeline += str(pipeline_type).format(size[0],size[1], frame_rate, tag)
        self.pipeline_tags.append((tag, None))
        self.pipeline_data[tag] = False

    def add_AI(self, AI_class, model_type, frame_rate, tag):
        self.pipeline += str(Pipelines.RGB).format(model_type["size"][0],model_type["size"][1], frame_rate, tag)
        self.pipeline_tags.append((tag, AI_class))
        AI_class.add_camera_pipeline(model_type, tag)
        self.pipeline_data[tag] = False

    def start(self):
        if self.device is not 0:
            self.watchdog_thread.daemon = True
            self.watchdog_thread.start()
        else:
            self.start_pipeline()
        while(self.pipeline_buffer=={}):
            sleep(0.0001)
    
    def pipeline_data_call(self, AI_class, tag):
        sink = self.parsed_pipeline.get_by_name(tag)
        start = time.monotonic()
        count = 0
        total_time = 0
        while True:
            sample = sink.emit("pull-sample")
            if sample is not None:
                inference_time = time.monotonic() - start
                count+=1
                total_time+=inference_time
                if(count>100):
                    total_time = inference_time
                    count = 1
                self.fps[tag] = count/total_time
                start = time.monotonic()
                buf = sample.get_buffer()
                result, mapinfo = buf.map(Gst.MapFlags.READ)
                if AI_class is not None:
                    nparr = np.frombuffer(mapinfo.data, dtype=np.uint8)
                    AI_class.analyze_pipeline_frame(nparr, tag)
                self.pipeline_buffer[tag] = mapinfo.data
                self.pipeline_data[tag] = True
                
    def start_pipeline(self):
        self.pipeline_started = True
        self.parsed_pipeline = Gst.parse_launch(self.pipeline)
        for tag, AI_class in self.pipeline_tags:
            sink_thread = Thread(target=self.pipeline_data_call,args=(AI_class, tag,))
            sink_thread.daemon = True
            sink_thread.start()
            self.data_call_threads.append(sink_thread)
        self.parsed_pipeline.set_state(Gst.State.PLAYING)

    def stop_pipeline(self):
        self.pipeline_started = False
        self.parsed_pipeline.set_state(Gst.State.NULL)

    def data(self,tag):
        return self.pipeline_data[tag]
    
    def get_image(self, tag):
        self.pipeline_data[tag] = False
        nparr = np.frombuffer(self.pipeline_buffer[tag], dtype=np.uint8)
        return nparr

    def get_fps(self, tag):
        if tag in self.fps.keys():
            return self.fps[tag]
        else:
            return 0
        

class Pipelines(enum.Enum):
    SRC = "v4l2src device=/dev/video{0} ! tee name=t"
    H264 = " t. ! queue max-size-buffers=1 leaky=downstream ! video/x-raw,format=YUY2,width={0},height={1},framerate={2}/1 ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency threads=4 key-int-max=5 bitrate=1000 aud=False bframes=1 ! video/x-h264,profile=baseline ! h264parse ! video/x-h264,stream-format=byte-stream,alignment=nal ! appsink name={3} emit-signals=True max-buffers=1 drop=False sync=False"
    RGB = " t. ! queue ! glfilterbin filter=glbox ! video/x-raw,format=RGB,width={0},height={1},framerate={2}/1 ! appsink name={3}"
    MJPEG = " t. ! queue ! video/x-raw,format=YUY2,width={0},height={1},framerate={2}/1 ! jpegenc quality=20 ! appsink name={3} emit-signals=True"

    def __str__(self):
        return self.value