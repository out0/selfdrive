import math
from pydriveless import Waypoint, angle

def pre_compute(goal: Waypoint, max_steering_angle: angle, wheelbase_px: float):
   
    steer = math.tan(max_steering_angle.rad())
    beta = math.atan(steer / 2)
    curvature = abs(math.cos(beta) * steer / (2 * wheelbase_px))
    R = 1/curvature

    s = math.sin(goal.heading.rad())
    c = math.cos(goal.heading.rad())


    left = (goal.x - R * s, goal.z + R * c)
    right = (goal.x + R * s, goal.z - R * c)

    return (left, right, R*R)

def outside(goal: Waypoint, left, right, r_squared, x, z) -> bool:
    dx_l = x - left[0]
    dx_r = x - right[0]

    dy_l = z - left[1]
    dy_r = z - right[1]

    dr_hipotenuse = dx_r * dx_r + dy_r * dy_r
    dl_hipotenuse = dx_l * dx_l + dy_l * dy_l

    return dr_hipotenuse >= r_squared or dl_hipotenuse >= r_squared


def check_not_reachable(goal: Waypoint, max_steering_angle: angle, wheelbase_px: float, x: int, z: int) -> bool:
    #(left, right, r_squared) = pre_compute(goal, max_steering_angle, wheelbase_px)

    steer = math.tan(max_steering_angle.rad())
    beta = math.atan(steer / 2)
    curvature = abs(math.cos(beta) * steer / (2 * wheelbase_px))
    R = 1/curvature
    R = R
    Rsq = R*R
  
    rad = goal.heading.rad()
    dx = math.sin(rad)
    dy = -math.cos(rad)
    p = (goal.x + R * dx, goal.z + R * dy)

    po = (p[0] - goal.x, p[1] - goal.z)

    pr = (-po[1], po[0])
    pl = (po[1], -po[0])

    pr = (pr[0] + goal.x, pr[1] + goal.z)
    pl = (pl[0] + goal.x, pl[1] + goal.z)

    if (x == 0 and z == 0):
        print(pr)
        print(pl)

    dx = pr[0] - x
    dz = pr[1] - z
    dist = dx * dx + dz * dz
    if abs(dist - Rsq) <= 2: return True

    dx = pl[0] - x
    dz = pl[1] - z
    dist = dx * dx + dz * dz
    if abs(dist - Rsq) <= 2: return True

    return False