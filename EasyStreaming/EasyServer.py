from .streaming.Reef.StreamingServer import StreamingServer

class EasyServer():
    def __init__(self, server_type, csi_h264 = None, usb_h264 = None):
        server_type(csi_h264, usb_h264)

class ServerType:
    Reef = StreamingServer