import numpy as np

class Camera:
    __width: int
    __height: int
    __fov: int
    __fps: int
    
    def __init__(self, 
                 width: int,
                 height: int,
                 fov: float = 120.0,
                 fps: int = 30,
                 ):
            self.__width = width
            self.__height = height
            self.__fov = fov
            self.__fps = fps
    
    def read(self) -> np.ndarray:
        pass
    
    def width(self) -> int:
        return self.__width

    def height(self) -> int:
        return self.__height

    def fov(self) -> int:
        return self.__fov

    def fps(self) -> int:
        return self.__fps

