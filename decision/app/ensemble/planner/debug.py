from pydriveless import Waypoint, angle
import os

class Debug:

    DEFAULT_LOG_FILE = "debug_path.log"

    def __init__(self):
        pass

    @classmethod
    def log_path (cls, path: list[Waypoint], file: str = DEFAULT_LOG_FILE) -> None:
        f = open(file, "w")
        for p in path:
            f.write(f"{p}\n")
        f.close()

    @classmethod
    def read_path(cls, file: str = DEFAULT_LOG_FILE) -> list[Waypoint]:
        f = open(file, "r")
        lines = f.readlines()
        res = []
        for l in lines:
            res.append(Waypoint.from_str(l))
        f.close()
        return res
    
if __name__ == "__main__":
    test_path = []
    for i in range(10):
        for j in range(10):
            test_path.append(Waypoint(i,j, angle.new_deg(10)))
    
    Debug.log_path(test_path)

    read_path = Debug.read_path()

    ok = True
    for i in range(len(test_path)):
        if test_path[i] != read_path[i]:
            ok = False
            break
    if ok:
        print("Debug class is ok.")
    else:
        print("Debug class is buggy, please check!")

    os.remove(Debug.DEFAULT_LOG_FILE)
    

