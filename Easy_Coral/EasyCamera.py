from time import sleep, monotonic
from os import path
from threading import Thread
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)


class Camera:
    def __init__(self, device_path):
        self.device_path = device_path
        self.pipeline_string = PipelineType.SRC.format(self.device_path)
        self.pipeline = None
        self.camera_running = False
        self.watchdog_thread = Thread(target=self.camera_watchdog)
        self.sinks = []
        self.sink_names = []

    def camera_watchdog(self):
        while True:
            if path.exists(self.device_path):
                if self.camera_running is False:
                    self.start_camera()
            else:
                if self.camera_running is True:
                    self.stop_camera()
            sleep(0.25)

    def add_pipeline(self, size=(640,480), frame_rate=30, pipeline_type=None, sink_name=None):
        if pipeline_type is None:
            pipeline_type = PipelineType.RGB
        
        if sink_name is None:
            self.sink_names.append(str(len(self.sinks)))
        else:
            self.sink_names.append(sink_name)

        self.pipeline_string += pipeline_type.format(size[0],size[1], frame_rate, self.sink_names[-1])
        sink = Sink(self, self.sink_names[-1], pipeline_type, size)
        self.sinks.append(sink)
        return self.sinks[-1]

    def start(self):
        if self.device_path is CameraType.CSI:
            self.start_camera()
        else:
            self.watchdog_thread.daemon = True
            self.watchdog_thread.start()

    def start_camera(self):
        self.camera_running = True
        self.pipeline = Gst.parse_launch(self.pipeline_string)
        for sink in self.sinks:
            sink_var = self.pipeline.get_by_name(sink.sink_name)
            sink.start_sink(sink_var)
        self.pipeline.set_state(Gst.State.PLAYING)

    def stop_camera(self):
        self.camera_running = False
        self.pipeline.set_state(Gst.State.NULL)


class Sink():
    def __init__(self, camera_class, sink_name, pipeline_type, size):
        self.camera_class = camera_class
        self.sink_name = sink_name
        self.pipeline_type = pipeline_type
        self.listeners = []
        self.pull_thread = None
        
        self.fps = 0
        self.size = size
        
    def add_listener(self, func):
        self.listeners.append(func)

    def remove_listener(self, func):
        target = 0
        for idx, listener in enumerate(self.listeners):
            if(listener == func):
                target = idx
        del(self.listeners[target])
    
    def sink_pull(self, sink):
        start = 0
        end = 0
        while True:
            sample = sink.emit("pull-sample")
            if sample is not None:
                buf = sample.get_buffer()
                result, mapinfo = buf.map(Gst.MapFlags.READ)
                
                if start is 0:
                    start = monotonic()
                else:
                    end = monotonic()
                    self.fps = int(1/(end-start))
                    start = end
                
                self.data = mapinfo.data
                self.send_data(self.data)
            sleep(0.0001)
    
    def start_sink(self, sink):
        self.pull_thread = Thread(target=self.sink_pull,args=(sink,))
        self.pull_thread.daemon = True
        self.pull_thread.start()

    def send_data(self, data):
        if self.pipeline_type is PipelineType.RGB:
            data = np.frombuffer(data, dtype=np.uint8)
        for listener in self.listeners:
            listener(data)


class PipelineType:
    SRC = "v4l2src device={0} ! tee name=t"
    H264 = " t. ! queue max-size-buffers=1 leaky=downstream ! video/x-raw,format=YUY2,width={0},height={1},framerate={2}/1 ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency threads=4 key-int-max=5 bitrate=1000 aud=False bframes=1 ! video/x-h264,profile=baseline ! h264parse ! video/x-h264,stream-format=byte-stream,alignment=nal ! appsink name={3} emit-signals=True max-buffers=1 drop=False sync=False"
    RGB = " t. ! queue ! glfilterbin filter=glbox ! video/x-raw,format=RGB,width={0},height={1},framerate={2}/1 ! appsink name={3}"
    MJPEG = " t. ! queue ! video/x-raw,format=YUY2,width={0},height={1},framerate={2}/1 ! jpegenc quality=20 ! appsink name={3} emit-signals=True"


class CameraType:
    CSI = "/dev/video0"
    USB = "/dev/video1"
    