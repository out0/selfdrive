#
#  SETUP DEVICE address
#

DEVICE_ADDR = "/dev/ttyUSB0"

import curses
import ctypes

LIBNAME = "./libcrawler.so"
lib = ctypes.CDLL(LIBNAME)

lib.initialize.restype = ctypes.c_void_p
lib.initialize.argtypes = [ctypes.c_char_p]

lib.resetActuators.restype = ctypes.c_bool
lib.resetActuators.argtypes = [ctypes.c_void_p]

lib.stop.restype = ctypes.c_bool
lib.stop.argtypes = [ctypes.c_void_p]

lib.setPower.restype = ctypes.c_bool
lib.setPower.argtypes = [ctypes.c_void_p, ctypes.c_float]

lib.setSteeringAngle.restype = ctypes.c_bool
lib.setSteeringAngle.argtypes = [ctypes.c_void_p, ctypes.c_float]

class CrawlerEgoCar:
    _ego_controller: ctypes.c_void_p
    
    def __init__(self, device: str) -> None:
        super().__init__()
        self._ego_controller = lib.initialize(ctypes.c_char_p(
                device.encode('utf-8')))
    
    def set_power(self, power: float) -> bool:
        return lib.setPower(self._ego_controller, power)
        pass
    
    def set_brake(self, brake: float) -> bool:
        return lib.stop(self._ego_controller)
        pass
    
    def set_steering(self, angle: int) -> bool:
        return lib.setSteeringAngle(self._ego_controller, angle)
        pass
    
ego = CrawlerEgoCar(DEVICE_ADDR)

def showDataInfo(stdscr, power: float, angle: int) -> None:
    stdscr.addstr(3, 30, f"                    ")
    stdscr.addstr(4, 30, f"                    ")
    stdscr.addstr(3, 40, f"{power:.1f}")
    stdscr.addstr(4, 40, f"{angle}")

def main(stdscr):
    
    power: float = 0
    angle: int = 0
    
    stdscr.clear()
    curses.curs_set(0)
    
    stdscr.nodelay(1)  # Don't block I/O calls
    stdscr.timeout(100)  # Wait 100ms before trying to get input
    
    stdscr.addstr(1, 0, "      w")
    stdscr.addstr(2, 0, "      /\\")
    stdscr.addstr(3, 0, "      |")
    stdscr.addstr(4, 0, "a <---+----> d")
    stdscr.addstr(5, 0, "      |")
    stdscr.addstr(6, 0, "      \/")
    stdscr.addstr(7, 0, "      s")
    
    stdscr.addstr(3, 20, "Power: ")
    stdscr.addstr(4, 20, "Steering angle: ")
    stdscr.addstr(9, 0, "Press 'q' to reset actuators")
    stdscr.addstr(11, 0, "Press <ESC> to exit")
    
   
    while True:
        key = stdscr.getch()
        
        if key == ord('w'):
            power += 0.1
            if power > 1: power = 1
            ego.set_power(power)
           
        elif key == ord('a'):
            angle -= 5
            if angle < -40: angle = -40
            ego.set_steering(angle)
            
        elif key == ord('s'):
            power -= 0.1
            if power < -1: power = -1           
            if power == -0.0: power = 0
            ego.set_power(power)
            
        elif key == ord('d'):
            angle += 5
            if angle > 40: angle = 40
            ego.set_steering(angle)
            
        elif key == ord('q'):
            power = 0.0
            angle = 0
            ego.set_power(power)
            ego.set_steering(angle)
        
        elif key == 27:  # Escape key
            break
        
        stdscr.refresh()
        showDataInfo(stdscr, power, angle)

curses.wrapper(main)
