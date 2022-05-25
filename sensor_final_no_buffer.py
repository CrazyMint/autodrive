# Imports
import time
import cv2
from lidar_final import CeptonLiDAR
from camera_final import ZedCameraSensor
import open3d as o3d
import numpy as np
##################################################################
# SENSOR INTERFACE CLASS
##################################################################


class Sensors:

    def __init__(self,
                 camera_resolution: str = '1080',
                 fps: int = 30,
                 camera_view: str = 'left',
                 include_depth: bool = True,
                 count: int = 5,
                 use_buffer: bool = False):

        # Initialize Camera
        self.camera = ZedCameraSensor(camera_resolution, fps, camera_view, include_depth=include_depth)

        self.include_depth = include_depth

        # start camera sensors
        self.camera.start()

        # get image sizes
        self.img_h = self.camera.image_height
        self.img_w = self.camera.image_width
        self.img_c = self.camera.num_channels

        # Initialize LiDAR
        self.lidar = CeptonLiDAR(count=count, use_buffer=use_buffer)
        self.lidar.start()

    def get_data(self):

        # get image data
        image_frame = self.camera.get_image_frame()

        # get depth data ## channel = None if we want to measure depth, to visualize can be same as image
        depth_image = self.camera.get_depth_map()

        lidar_out = self.lidar.get_accumulate_frame()
        pcd = np.vstack(lidar_out)


        data_dict = {

            'image': image_frame,
            'depth': depth_image,
            'point_cloud': pcd

        }

        return data_dict

    def close_sensors(self):
        self.lidar.exit()
        self.camera.exit()


if __name__ == "__main__":
    sensor = Sensors()
    time.sleep(0.5)

    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    reset = True



    while True:
        start = time.perf_counter()
        data_dict = sensor.get_data()
        print('Data Packet Recieved')
        print(f'image: ', data_dict['image'].shape)
        print(f'depth: ', data_dict['depth'].shape)
        print(f'lidar: ', data_dict['point_cloud'].shape)

        print(f'Total time = {time.perf_counter() - start}')

        cv2.imshow("image", data_dict['image'])


        pcd.points = o3d.utility.Vector3dVector(data_dict['point_cloud'])
        vis.update_geometry(pcd)
        if reset:
            vis.reset_view_point(True)
            reset = False
        vis.poll_events()
        vis.update_renderer()

        key = cv2.waitKey(30)

        if key == ord('q'):
            break

    sensor.close_sensors()



