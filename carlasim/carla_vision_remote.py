from carlasim.carla_ego_car import EgoCar
import numpy as np
import threading
from model.vision import Vision
from enum import Enum
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0') 
from gi.repository import Gst, GstApp

class CarlaVisionSender:
    _ego_car: EgoCar
    _host: str
    _port: int

    def __init__(self, ego: EgoCar, host: str, port: int) -> None:
        super().__init__()
        self._ego_car = ego
        self._host = host
        self._port = port

    def start(self) -> None:
        self._ego_car.stream_bev_rgb_camera_to(self._host, 
                                           self._port, 
                                           self._ego_car.bev_camera.width(), 
                                           self._ego_car.bev_camera.height(), 
                                           self._ego_car.bev_camera.fps())
    
    def destroy(self) -> None:
        self._ego_car.stop_stream_bev_rgb_camera()


class CarlaVisionReceiver (Vision):
    _listen_port: int
    _last_frame: np.array
    _running: bool
    _pipeline: any
    _frame_locked: bool

    def initialize_gst() -> None:
        Gst.init(None)

    def __init__(self, listen_port: int) -> None:
        super().__init__()
        self._listen_port = listen_port
        self._running = False
        self._frame_locked = False
        self.__open_stream()
    
    def destroy(self):
        self.__terminate_stream()

    def __bus_handler(bus, message):
        print (message)
    # handle the message
    

    def __on_frame_received(self, sink: GstApp.AppSink) -> None:
        if self._frame_locked:
            return Gst.FlowReturn.OK
        
        sample = sink.pull_sample()
        caps = sample.get_caps()

        # Get the actual data
        buffer = sample.get_buffer()
        
        # Get read access to the buffer data
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            print("Could not map buffer data!")
            return Gst.FlowReturn.ERROR

        self._last_frame = np.ndarray(
            (
                caps.get_structure(0).get_value('height'),
                caps.get_structure(0).get_value('width'),
                3
            ),
            buffer=buffer.extract_dup(0, buffer.get_size()), dtype=np.uint8)

        print (f"got new frame. Shape = {self._last_frame.shape}")

        buffer.unmap(map_info)
        return Gst.FlowReturn.OK
        
    def __open_stream(self) -> None:
        pipeline_string = f"udpsrc port={self._listen_port} " +\
                                "! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 "+\
                                "! rtph264depay ! decodebin ! videoconvert ! video/x-raw, format=RGB "+\
                                "! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        
        self._pipeline = Gst.parse_launch(pipeline_string)
        self._pipeline.set_state(Gst.State.PLAYING)
        bus = self._pipeline.get_bus()
        bus.connect('message', self.__bus_handler)
        bus.add_signal_watch()
        self._pipeline.get_by_name('sink').connect('new-sample', self.__on_frame_received)

    def __terminate_stream(self) -> None:
        self._pipeline.set_state(Gst.State.NULL)
        
 
    def read(self) -> np.array:
        self._frame_locked = True
        f = self._last_frame
        self._frame_locked = False
        return f
