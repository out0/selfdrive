

LOGGING_LEVEL = 0
LOGGING_FILE = "run.log"

class Telemetry:
    
    @classmethod
    def log (cls, level: int, module: str, message: str) -> None:
        if level < LOGGING_LEVEL: return
        
        msg = f"[{module}] {message}"
        
        if LOGGING_FILE is None:
            print (msg)
        else:
            f = open(LOGGING_FILE, "a")
            f.write(f"{msg}\n")
            f.close()