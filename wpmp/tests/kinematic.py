import math

def kinematic_curve(
    plane_dim,           # (width, height)
    start,                # (x, y)
    heading,
    steering_angle,
    max_path_size,
    wheelbase_px,
    cb,                   # callback(result, cx, cz, heading) -> float
    result
):
    heading = heading - (math.pi/2)
    width, height = plane_dim
    x, z = float(start[0]), float(start[1])

    steer = math.tan(steering_angle)
    dt = 0.1
    beta = math.atan(steer / 2)
    curvature = dt * math.cos(beta) * steer / (2 * wheelbase_px)

    max_size = int(max_path_size) + 1
    size = 0
    last_x = int(start[0])
    last_z = int(start[1])

    curve_cost = 0.0

    while max_path_size <= 0 or size < max_size:
        x += dt * math.cos(heading + beta)
        z += dt * math.sin(heading + beta)
        heading += curvature

        cx = int(x)
        cz = int(z)

        if cx == last_x and cz == last_z:
            continue

        if cx < 0 or cx >= width or cz < 0 or cz >= height:
            break

        size += 1

        point_cost = cb(result, cx, cz, heading)

        if point_cost < 0:
            return -1

        curve_cost += point_cost

    return curve_cost