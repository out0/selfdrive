import sys
sys.path.append("../../")
import curses

from crawler.crawler_ego_car import CrawlerEgoCar

ego = CrawlerEgoCar("/dev/ttyACM0")


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
