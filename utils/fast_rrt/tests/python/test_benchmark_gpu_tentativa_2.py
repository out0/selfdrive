import sys
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../")
sys.path.append("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive")
import unittest
from utils.fast_rrt.fastrrt import FastRRT
from test_utils import TestFrame, TestData, TestUtils
import time, math, numpy as np, os


MAX_STEERING_ANGLE = 40
VEHICLE_LENGTH_M = 5.412658774
TIMEOUT = -1

def get_test_data(file: str) -> TestData:
    return TestFrame(file).get_data_cuda()

class TestFastRRT(unittest.TestCase):
    
    
    def distance_between(p1: tuple[int, int, float], p2: tuple[int, int, float]) -> float:
        dx = p2[0] - p1[0]
        dz = p2[1] - p1[1]
        return math.sqrt(dx * dx + dz * dz)
    
    
    def hermite_interpolation(width: int, height: int, p1: tuple[int, int, float], p2: tuple[int, int, float]) -> list[tuple[int, int]]:     
        curve =[]
        d = TestFastRRT.distance_between(p1, p2)
        numPoints = int(round(d))
        
        p1x = p1[0]
        p1z = p1[1]
        p2x = p2[0] 
        p2z = p2[1]
        p1heading = p1[2]
        p2heading = p2[2]
        
        if numPoints <= 2:
            return [(p1x, p1z), (p2x, p2z)] 
        a1 = p1heading - (math.pi / 2)
        a2 = p2heading - (math.pi / 2)
        
        # Tangent vectors
        tan1 = (d * math.cos(a1), d * math.sin(a1))
        tan2 = (d * math.cos(a2), d * math.sin(a2))

        last_x = -1
        last_z = -1
        
        for i in range (numPoints):
            t = float(i) / (numPoints - 1)

            t2 = t * t
            t3 = t2 * t

            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2

            x = h00 * p1x + h10 * tan1[0] + h01 * p2x + h11 * tan2[0]
            z = h00 * p1z + h10 * tan1[1] + h01 * p2z + h11 * tan2[1]

            if (x < 0 or x >= width):
                continue
            if (z < 0 or z >= height):
                continue

            cx = int(round(x))
            cz = int(round(z))

            if (cx == last_x and cz == last_z):
                continue
            
            
            h00p = 6 * t2 - 6 * t
            h10p = 3*t2 - 4 * t + 1
            h01p = -6 * t2 + 6 * t
            h11p = 3*t2 - 2*t
            
            xp = h00p * p1x + h10p * tan1[0] + h01p * p2x + h11p * tan2[0]
            zp = h00p * p1z + h10p * tan1[1] + h01p * p2z + h11p * tan2[1]
            #h = math.degrees(math.pi/2 - math.atan2(-zp, xp))
            h = math.pi/2 - math.atan2(-zp, xp)

            curve.append((cx, cz, h))
            last_x = cx
            last_z = cz
        
        return curve

    
    def path_list_to_ndarray (curve: tuple[int, int, float]) -> np.ndarray:
        path = np.ndarray((len(curve), 3), dtype=np.float32)
        i = 0
        for p in curve:
            path[i, 0] = p[0]
            path[i, 1] = p[1]
            path[i, 2] = 0.0
            i += 1
        return path
    
    def test_small_cluttered(self):
        # Load the test image
        
        #data = TestUtils.timed_exec(get_test_data, "small_cluttered_2.png")
        #data = TestUtils.timed_exec(get_test_data, "test_scenarios/small_3.png")
        data = TestUtils.timed_exec(get_test_data, "custom1.png")
        
        rrt = FastRRT(
            width=data.width(),
            height=data.height(),
            perception_height_m=data.real_height(),
            perception_width_m=data.real_width(),
            max_steering_angle_deg=MAX_STEERING_ANGLE,
            vehicle_length_m=VEHICLE_LENGTH_M,
            timeout_ms=TIMEOUT,
            min_dist_x=22,
            min_dist_z=40,
            lower_bound_x=-1,
            lower_bound_z=-1,
            upper_bound_x=-1,
            upper_bound_z=-1,
            max_path_size_px=50.0,
            dist_to_goal_tolerance_px=15.0,
            libdir=None
        )

        # large
        # rrt.set_plan_data(
        #     data.frame.get_cuda_frame(),
        #     start=(data.start.x, data.start.z, math.radians(90.0)),
        #     goal=(data.goal.x, data.goal.z, math.radians(45.0)),
        #     velocity_m_s=5.0
        # )
        
        start = (data.start.x, data.start.z, 0.0)
        goal = (data.goal.x, data.goal.z, 0.0)  
        
        rrt.set_plan_data(
            data.frame.get_cuda_frame(),
            start=start,
            goal=goal,
            velocity_m_s=1.0
        )

        # rrt.set_plan_data(
        #     data.frame.get_cuda_frame(),
        #     start=(data.start.x, data.start.z, 0.0),
        #     goal=(1024, 0, math.radians(45.0)),
        #     velocity_m_s=5.0
        # )

        #os.remove("my_path.npy") if os.path.exists("my_path.npy") else None
        if not os.path.exists("my_path.npy"):
            start_time = time.time()
            loop_count = 0
            rrt.search_init()
            while not rrt.goal_reached() and rrt.loop(True):
                loop_count += 1
                #nodes = rrt.export_graph_nodes()
                #TestUtils.output_path_result(data.frame, nodes, "output1.png")
                pass
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"[Coarse path] total: {1000 * execution_time:.6f} ms, mean: {1000 * (execution_time/loop_count):.6f} ms/loop, num loops: {loop_count}")
            
                    
            # Check if the goal is reached
            self.assertTrue(rrt.goal_reached())
            
            path = rrt.get_planned_path(interpolate=False)
            TestUtils.output_path_result(data.frame, path, "output1.png")
            print ("path size: ", len(path))

            nodes = rrt.export_graph_nodes()
            np.save('my_path.npy', path)
            
        path = np.load('my_path.npy')
        ds = 10

        for _ in range(10):
            
            num_points = len(path)
            
            new_path = []
            p1 = 0
            for i in range(10):
                p2 = p1 + ds
                if p2 >= num_points:
                    break
                new_curve = TestFastRRT.hermite_interpolation(data.width(), data.height(), 
                                                (path[p1, 0], path[p1, 1], path[p1, 2]),
                                                (path[p2, 0], path[p2, 1], path[p2, 2]))
                new_path.extend(new_curve)
                p1 = p2
                
            
            TestUtils.output_2path_result(data.frame, path, TestFastRRT.path_list_to_ndarray(new_path) , "output1.png")
            
            path = TestFastRRT.path_list_to_ndarray(new_path)
            ds = int(round(num_points) / 10)
        
       
        # TestUtils.timed_loop_exec_count("Optimized path", 100, None, rrt.loop_optimize)       
            
        # path = rrt.get_planned_path(True)
        # if path is None:
        #     return
        # TestUtils.output_path_result(data.frame, path, "output1_optim.png")        
        # print ("path size: ", len(path))
    
    #https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
    
    def test_fast_rrt_with_custom1(self):
        return
        # Load the test image
        
        data = TestUtils.timed_exec(get_test_data, "custom3.png")
        
        rrt = FastRRT(
            width=data.width(),
            height=data.height(),
            perception_height_m=data.real_height(),
            perception_width_m=data.real_width(),
            max_steering_angle_deg=MAX_STEERING_ANGLE,
            vehicle_length_m=VEHICLE_LENGTH_M,
            timeout_ms=TIMEOUT,
            min_dist_x=22,
            min_dist_z=40,
            lower_bound_x=data.lower_bound.x,
            lower_bound_z=data.lower_bound.z,
            upper_bound_x=data.upper_bound.x,
            upper_bound_z=data.upper_bound.z,
            max_path_size_px=50.0,
            dist_to_goal_tolerance_px=15.0,
            libdir=None
        )


        rrt.set_plan_data(
            data.frame.get_cuda_frame(),
            start=(data.start.x, data.start.z, data.start.heading),
            goal=(data.goal.x, data.goal.z, data.goal.heading),
            velocity_m_s=1.0
        )
        
        
        TestUtils.timed_loop_exec("Coarse path", rrt.search_init, lambda: not rrt.goal_reached() and rrt.loop(True))
        
        # Check if the goal is reached
        self.assertTrue(rrt.goal_reached())
        
        path = rrt.get_planned_path(interpolate=True)
               
        print ("path size: ", len(path))
        TestUtils.output_path_result(data.frame, path, "output1.png")
       
        TestUtils.timed_loop_exec_count("Optimized path", 30, None, rrt.loop_optimize)       
            
        path = rrt.get_planned_path(False)
        TestUtils.output_path_result(data.frame, path, "output1.png")

if __name__ == "__main__":
    unittest.main()