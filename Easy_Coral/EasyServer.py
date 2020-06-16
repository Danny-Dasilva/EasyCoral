from .EasyStreaming.Reef.StreamingServer import StreamingServer
from .EasyStreaming.overlay import DetectConvert, ClassifyConvert

class EasyServer():
    def __init__(self, server_type, csi_h264 = None, usb_h264 = None):
        self.server = server_type(csi_h264, usb_h264)

    def data(self,data):
        self.server.write_csi(data)

    def aidata(self, data):
        self.server.send_overlay(data) #calls svg overlay function in streaming server
    
    def DetectConvert(data):
        return DetectConvert.convert(data)

    def ClassifyConvert(data):
        return ClassifyConvert.convert(data)


class ServerType:
    Reef = StreamingServer