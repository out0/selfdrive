import carla

rolename = "hero"

client = carla.Client("localhost", 2000)
world = client.get_world()
hero = next(filter(lambda a: a.attributes.get('role_name') == rolename, world.get_actors()).__iter__())

def imu_callback(name, data: carla.IMUMeasurement):
    print("{}, frame {}, accel: ({}, {}, {})".format(name, data.frame_number, data.accelerometer.x, data.accelerometer.y, data.accelerometer.z))

sensors = []

for i in [0, 1]:
    bp = blueprint = world.get_blueprint_library().find('sensor.other.imu')
    imu_rolename = "imu{}".format(i)
    bp.set_attribute('role_name', imu_rolename)
    bp.set_attribute('sensor_tick', '0.01')
    transform = carla.Transform(carla.Location(0, 0, 0), carla.Rotation(i * 10, i * 20, i * 30))
    sensor = world.spawn_actor(bp, transform, attach_to=hero)
    sensors.append(sensor)

sensors[0].listen(lambda data: imu_callback("imu0", data))
sensors[1].listen(lambda data: imu_callback("imu1", data))