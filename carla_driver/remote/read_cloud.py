import open3d

# Load and visualize the point cloud
pcd = open3d.io.read_point_cloud("point_cloud.ply")
open3d.visualization.draw_geometries([pcd])