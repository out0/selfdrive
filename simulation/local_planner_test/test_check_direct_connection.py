from pydriveless import TestUtils, Waypoint
from pydriveless import EgoParams, SearchParams, SearchFrame, angle
import math, numpy as np, cv2

HALF_PI = 0.5*math.pi

def direct_connection(frame: SearchFrame, 
                      outp: np.ndarray,
                      p1: Waypoint, 
                      p2: Waypoint,
                      min_dist: tuple[int, int], 
                      distance_to_goal_tolerance: float, 
                      max_curvature: float,
                      max_heading_error: float):
    width = frame.width()
    height = frame.height()

 
    max_dist_to_goal_squared = distance_to_goal_tolerance * distance_to_goal_tolerance;

    dx = p2.x - p1.x
    dz = p2.z - p1.z
    distance = math.sqrt(dx * dx + dz * dz)

    numPoints = int(1.5 * distance)

    a1 = p1.heading.rad() - HALF_PI
    a2 = p2.heading.rad() - HALF_PI

    #Tangent vectors
    tan1 = [distance * math.cos(a1), distance * math.sin(a1)]
    tan2 = [distance * math.cos(a2), distance * math.sin(a2)]

    last_x = -1
    last_z = -1
    last_heading = 0

    parentCost = frame.get_cost(p1.x, p1.z)
    nodeCost = parentCost

    for i in range(numPoints):
        t = (0.0 + i) / (numPoints - 1)
        t2 = t * t
        t3 = t2 * t

        # Hermite basis functions
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        px = h00 * p1.x + h10 * tan1[0] + h01 * p2.x + h11 * tan2[0];
        pz = h00 * p1.z + h10 * tan1[1]+ h01 * p2.z + h11 * tan2[1];

        if (px < 0 or px >= width):
            continue
        if (pz < 0 or pz >= height):
            continue

        cx = int(px)
        cz = int(pz)

        if (cx == last_x and cz == last_z):
            continue
        if (cx < 0 and cx >= width):
            continue
        if (cz < 0 and cz >= height):
            continue

        nodeCost += frame.get_cost(cx, cz) + 1

        
        t00 = 6 * t2 - 6 * t
        t10 = 3 * t2 - 4 * t + 1
        t01 = -6 * t2 + 6 * t
        t11 = 3 * t2 - 2 * t

        ddx = t00 * p1.x + t10 * tan1[0] + t01 * p2.x + t11 * tan2[0]
        ddz = t00 * p1.z + t10 * tan1[1] + t01 * p2.z + t11 * tan2[1]

        last_heading = math.atan2(ddz, ddx) + HALF_PI;

        d00 = 12 * t - 6
        d10 = 6 * t - 4
        d01 = -12 * t + 6
        d11 = 6 * t - 2

        dd2x = d00 * p1.x + d10 * tan1[0] + d01 * p2.x + d11 * tan2[0]
        dd2z = d00 * p1.z + d10 * tan1[1] + d01 * p2.z + d11 * tan2[1]

        outp[cz, cx, :] = [0, 255, 0]

        if (max_curvature > 0):
            k = abs(ddx * dd2z - ddz * dd2x) / pow(ddx * ddx + ddz * ddz, 1.5)
            if (k > max_curvature):
                print (f"max curvature exceeded in {cx}, {cz}")
                outp[cz, cx, :] = [0, 0, 255]
                #return (-1, -1, 0, 0.0)
        

        # Interpolated point
        last_x = cx
        last_z = cz

        # t = frame.get_traversability(cx, cz)
        # if t & 0x100 <= 0:
        #     print(f"[direct goal {p1} --> {p2} not ALL feasible at {cx}, {cz}, {last_heading}")
        #     outp[cz, cx, :] = [0, 0, 255]

        p = Waypoint(last_x, last_z, angle.new_rad(last_heading))
        if (not frame.check_feasible_path(min_dist, [p])):
            print(f"[direct goal {p1} --> {p2} not feasible at {p}")
            outp[cz, cx, :] = [0, 0, 255]
            
            #return (-1, -1, 0, 0.0)

    if (numPoints <= 0):
        return (-1, -1, 0, 0.0)
    
    if (abs(last_heading - p2.heading.rad()) > max_heading_error):
        return (-1, -1, 0, 0.0)

    dx = p2.x - last_x
    dz = p2.z - last_z
    if ((dx * dx + dz * dz) > max_dist_to_goal_squared):
        return (-1, -1, 0, 0.0)

    

    return (float(last_x), float(last_z), last_heading, nodeCost);


if __name__ == "__main__":
    min_dist = (20, 20)
    distance_to_goal_tolerance = 15
    max_curvature = 5
    max_heading_error = 5
    conf = TestUtils.read_config("map_cost_8")
    frame: SearchFrame = TestUtils.build_cuda_frame(conf)
    frame.process_safe_distance_zone(min_dist, True)

    f = TestUtils.export_safe_distance_frame(frame, "output2.png")

    p1 = Waypoint(245, 746, angle.new_deg(-45.7501950796617))
    p2 = Waypoint(442, 450, angle.new_deg(160.00000900563518))


    res = direct_connection(frame, f,
                      p1, 
                      p2,
                      min_dist, 
                      distance_to_goal_tolerance, 
                      max_curvature,
                      max_heading_error)
    
    cv2.imwrite("output2.png", f)
    
    print(res)
    pass

            