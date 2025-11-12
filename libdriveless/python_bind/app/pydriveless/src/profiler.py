
import time
from inspect import getframeinfo, stack

class Profiler:
    _code_profile: dict

    def __init__(self):
        self._code_profile = {}
        self._enabled = True
        
    def _key(self, calling_file: str, func_name: str) -> str:
        return f"{calling_file}%{func_name}"

    def fn_measurement_start(self, func_name: str, stack_skip: int = 2) -> None:
        caller = getframeinfo(stack()[stack_skip][0])
        calling_fn = caller.function if func_name is None else func_name        
        calling_file = caller.filename.split('/')[-1]

        key = self._key(calling_file, calling_fn)

        if key not in self._code_profile:
            self._code_profile[key] = [
                time.time(),
                0,
                0
            ]
        else:
            self._code_profile[key][0] = time.time()

    def fn_measurement_finish(self, func_name: str, stack_skip: int = 2) -> tuple[str, float, float]:
        caller = getframeinfo(stack()[stack_skip][0])
        calling_fn = caller.function if func_name is None else func_name
        calling_file = caller.filename.split('/')[-1]

        key = self._key(calling_file, calling_fn)

        if key not in self._code_profile:
            return

        curr_time = 1000 * (time.time() - self._code_profile[key][0])

        self._code_profile[key][1] += curr_time
        self._code_profile[key][2] += 1
            
        
        return (calling_file, calling_fn, curr_time, self._code_profile[key][1])
        
    def _add_timing(self, calling_file: str, calling_fn: str, val: float) -> None:
        
        key = self._key(calling_file, calling_fn)

        if key not in self._code_profile:
            self._code_profile[key] = [
                -1,
                val,
                0
            ]
        else:
            self._code_profile[key][1] += val
        
        self._code_profile[key][2] += 1
    
    def _get_last_exec_time(self, file: str, func_name: str) -> float:
        key = self._key(file, func_name)

        if key not in self._code_profile:
            return -1
        return self._code_profile[key][0]
    
    def _get_last_accum_time(self, file: str, func_name: str) -> float:
        key = self._key(file, func_name)

        if key not in self._code_profile:
            return -1
        return self._code_profile[key][1]

    def _clear(self) -> None:
        self._code_profile.clear()

    def _print(self) -> None:
        print("\n")
        for key, val in self._code_profile.items():
            try:
                file, func = key.split('%', 1)
            except ValueError:
                continue
            accum = val[1]
            num_call = val[2]
            
            if num_call > 0:
                mean_exec = accum / num_call
            else:
                mean_exec = 0
            #print(f"{file}\t{func}\t{accum}\t{num_call}")
            print(f"{file}, {func}:  [{accum:.3f} ms], num calls: {num_call}, mean: {mean_exec:.2f} ms")


    @staticmethod
    def start(func_name: str = None) -> None:
        if not hasattr(Profiler, "_profiler"):
            Profiler._profiler = Profiler()

        if not Profiler._profiler._enabled:
            return

        Profiler._profiler.fn_measurement_start(func_name)

    @staticmethod
    def end(func_name: str = None) -> tuple[str, float, float]:
        if not hasattr(Profiler, "_profiler"):
            return

        if not Profiler._profiler._enabled:
            return

        return Profiler._profiler.fn_measurement_finish(func_name)

    @staticmethod
    def print() -> None:
        pass

    @staticmethod
    def clear() -> None:
        if not hasattr(Profiler, "_profiler"):
            return
        Profiler._profiler._clear()

    @staticmethod
    def exec(func, *args, **kwargs) -> None:
        if not hasattr(Profiler, "_profiler"):
            Profiler._profiler = Profiler()
        
        if not Profiler._profiler._enabled:
            return func(*args, **kwargs)

        time_init = time.time()
        res = func(*args, **kwargs)
        curr_time = 1000 * (time.time() - time_init)

        caller = getframeinfo(stack()[1][0])
        calling_file = caller.filename.split('/')[-1]

        Profiler._profiler._add_timing(calling_file, func.__name__, curr_time)


        #return (calling_file, func.__name__, curr_time, accum_timing)
        return res
    
    @staticmethod
    def get_last_exec_time (file: str, func_name: str) -> float:
        if not hasattr(Profiler, "_profiler"):
            return -1
        
        return Profiler._profiler._get_last_exec_time(file, func_name)

    @staticmethod
    def get_last_accum_time (file: str, func_name: str) -> float:
        if not hasattr(Profiler, "_profiler"):
            return -1
        
        return Profiler._profiler._get_last_accum_time(file, func_name)
    

    @staticmethod
    def print() -> None:
        if not hasattr(Profiler, "_profiler"):
            return -1
        Profiler._profiler._print()

    @staticmethod
    def enable() -> None:
        if not hasattr(Profiler, "_profiler"):
            return
        Profiler._profiler._enabled = True

    @staticmethod
    def disable() -> None:
        if not hasattr(Profiler, "_profiler"):
            return
        Profiler._profiler._enabled = False

    @staticmethod
    def enable_if(condition: bool) -> None:
        if not hasattr(Profiler, "_profiler"):
            return
        
        if not isinstance(condition, bool):
            raise TypeError("condition must be a boolean")
        
        Profiler._profiler._enabled = condition