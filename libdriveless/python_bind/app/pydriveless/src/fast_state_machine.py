from threading import Thread, Lock

class FastStateMachine:
    _state: int
    _thr: Thread
    _mtx: Lock
    _run: bool
    _state_map: dict
    __debug: bool

    def __init__(self, state_map: dict, debug: bool = False, initial_state: any = 0):
        self._run = True
        self.__debug = debug
        self._state_map = state_map
        self._state = initial_state
        self._mtx = Lock()
    
    def __del__(self) -> None:
        self.destroy()

    def start(self) -> None:
        self._thr = Thread(target=self._loop)
        self._thr.start()


    def destroy(self) -> None:
        if self._thr is None: return
        self._run = False
        if self._thr.is_alive():
            self._thr.join()
        self._thr = None
    
    def is_running(self) -> bool:
        return self._run
    
    def state(self) -> any:
        self._mtx.acquire(blocking=True)
        st = self._state
        self._mtx.release()
        return st
    
    def set_state(self, val: any) -> None:
        self._mtx.acquire(blocking=True)
        self._state = val
        self._mtx.release()

    def _loop(self) -> None:
        while(self._run):

            st = self.state()

            if st not in self._state_map:
                raise Exception(f"state {st} is not mapped in the state map ({len(self._state_map)} states), therefore it is invalid.")
            next_state_fn = self._state_map[st]
            if next_state_fn is None:
                if self.__debug:
                    print(f"[state {st}] warning: state function call in the state_map = 'None', no function will be called and the state is not going to be changed")
                continue
            
            new_state = getattr(self, next_state_fn)(st)
            if new_state is None:
                if self.__debug:
                    print(f"[state {st}] warning: state function {next_state_fn} returned None as next state")
            elif self.__debug:
                print (f"[state {st} to {new_state}] when calling {next_state_fn}")
            
            self.set_state(new_state)
