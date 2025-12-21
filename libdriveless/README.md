## LibDriveless

This is a C++ library with python bindings that implements basic features for self-driving projects.

Basic model classes:

| Class | Description |
|-------|-------------|
| angle | provides an abstraction to represent an angle that helps avoiding bugs related to radian/degree conversions along the code. |
| async_component | provides an async thread execution based on start/stop loop for code modularization |
| coord_conversion | converts coordinates between Waypoint <---> Map <---->  World (lat/lon) |
| cuda_frame | provides a width x height frame of any cuda type that implements fast clear() using the GPU and handles memory copy from RAM-GPU |
| cuda_ptr | provides a memory safe abstraction for CUDA data |
| interpolator | provides interpolation CPU functions (hermite and spline) |
| map_pose | Map coordinate representation and useful geometric methods |
| quaternion | Quaternion representation, implementing most basic math features. Supports CUDA execution for GPU quaternion computing |
| search_frame | Represents a frame that can be used as the Search Space for Path Planning. The data is represented in CUDA to allow using the GPU for code acceleration |
| search_params | Represents a data storage for basic search params and definitions that are used during planning |
| state | Abstraction that represents vehicle information in the state space |
| waypoint | Search space coordinate representation and useful geometric methods |
| world_pose | State space coordinate representation (GNSS) and useful geometric methods |
