from pydriveless import Profiler
import unittest, time

class TestProfiler(unittest.TestCase):

    def test_init_end_context(self):
        Profiler.clear()
        i = time.time()
        Profiler.start()       
        for _ in range(10):
            time.sleep(0.1)
        file, fn, proc_time, total_accum_time = Profiler.end()
        i = 1000*(time.time() - i)

        self.assertEqual(file, "test_profiler.py")
        self.assertEqual(fn, "test_init_end_context")
        self.assertTrue((i - proc_time) < 100)
        self.assertTrue((i - total_accum_time) < 100)
        self.assertEqual(proc_time, total_accum_time)
        pass

    def test_init_end_context_partial_timing(self):
        Profiler.clear()
        for _ in range(10):
            i = time.time()
            Profiler.start()   
            time.sleep(0.1)
            file, fn, proc_time, total_accum_time = Profiler.end()
            p = 1000*(time.time() - i)
            self.assertTrue((p - proc_time) < 100)
            self.assertEqual(file, "test_profiler.py")
            self.assertEqual(fn, "test_init_end_context_partial_timing")
        
        i = 1000*(time.time() - i)

        total_accum_time = Profiler.get_last_accum_time("test_profiler.py", "test_init_end_context_partial_timing")
        
        
        self.assertTrue((i - total_accum_time) < 100)


    def test_exec_empty_param(self):
        Profiler.clear()
        res = Profiler.exec(lambda : time.sleep(0.5))
        self.assertIsNone(res)
        proc_time = Profiler.get_last_exec_time("test_profiler.py", "<lambda>")
        self.assertTrue((proc_time - 500) < 10)
        total_accum_time = Profiler.get_last_accum_time("test_profiler.py", "<lambda>")
        self.assertTrue((total_accum_time - 500) < 10)

    def test_exec_empty_param_many_calls(self):
        Profiler.clear()
        for _ in range(10):
            res = Profiler.exec(lambda : time.sleep(0.1))
            self.assertIsNone(res)
            proc_time = Profiler.get_last_exec_time("test_profiler.py", "<lambda>")
            self.assertTrue((proc_time - 100) < 10)
        
        total_accum_time = Profiler.get_last_accum_time("test_profiler.py", "<lambda>")
        self.assertTrue((total_accum_time - 1000) < 10)
        pass

    def call_method(p1, p2, p3) -> bool:
        return 2 * p1 + p2 + p3

    def test_exec_with_params(self):
        Profiler.clear()
        for _ in range(10):
            res = Profiler.exec(TestProfiler.call_method, 1, 2, 3)
            self.assertEqual(res, 7)
            proc_time = Profiler.get_last_exec_time("test_profiler.py", "call_method")
            self.assertTrue((proc_time - 100) < 10)
        
        total_accum_time = Profiler.get_last_accum_time("test_profiler.py", "call_method")
        self.assertTrue((total_accum_time - 1000) < 10)
        pass

    def func_a(self, p1) -> float:
        time.sleep(0.01)
        return Profiler.exec(self.func_b, p1)
    
    def func_b(self, p1) -> float:
        Profiler.start()
        res = 2.0 * p1
        Profiler.end()

        Profiler.start()
        res = 2.0 * p1
        Profiler.end()
        return res

    def test_print(self):
        Profiler.clear()
        for i in range(100):
            v = Profiler.exec(self.func_a, i)
            self.assertEqual(v, 2*i)
        Profiler.print()

    
    def test_print_disabled_profile(self):
        Profiler.clear()
        
        Profiler.disable()
        for i in range(100):
            v = Profiler.exec(self.func_a, i)
            self.assertEqual(v, 2*i)
        Profiler.print()

        #enable and execute again
        Profiler.enable()
        for i in range(100):
            v = Profiler.exec(self.func_a, i)
            self.assertEqual(v, 2*i)
        Profiler.print() 

    def test_enable_if(self):
        Profiler.clear()

        is_enabled = False

        Profiler.enable_if(is_enabled)
        for i in range(10):
            v = Profiler.exec(self.func_a, i)
            self.assertEqual(v, 2*i)
        Profiler.print()

        
        #enable and execute again
        is_enabled = True
        Profiler.enable_if(is_enabled)
        for i in range(100):
            v = Profiler.exec(self.func_a, i)
            self.assertEqual(v, 2*i)
        Profiler.print()          

if __name__ == "__main__":
    unittest.main()
