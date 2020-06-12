from .streaming.Reef.StreamingServer import StreamingServer

class EasyServer():
    def __init__(self, server_type, csi_h264 = None, usb_h264 = None):
        self.server = server_type(csi_h264, usb_h264)

    def data(self,data):
        self.server.write_csi(data)

    def aidata(self,data):
        send_overlay(data) #calls svg overlay function in streaming server

class ServerType:
    Reef = StreamingServer