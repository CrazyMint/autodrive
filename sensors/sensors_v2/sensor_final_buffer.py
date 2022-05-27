# Imports
import time
import cv2
from lidar_final import CeptonLiDAR
from camera_final import ZedCameraSensor
import numpy as np
import open3d as o3d
import os
## Buffer Code


##################################################################
# SENSOR INTERFACE CLASS
##################################################################


class Sensors:

    def __init__(self,
                 camera_resolution: str = '1080',
                 fps: int = 30,
                 camera_view: str = 'left',
                 include_depth: bool = True,                 
		 count: int = 1,
                 use_buffer: bool = True):

        # get image sizes
        self.img_h = 1080
        self.img_w = 1920
        self.img_c = 3
        self.count = 1
        self.max_depth = 10

    def get_data(self):
        # get image data
        image_frame = ZedCameraSensor.get_from_buffer(os.path.join('/home/osu/adc_software/python/Lidar/sensors_working_2022/sensors_working_v2/sensors','zed_image'),
                                                      image_shape=(self.img_h, self.img_w, self.img_c))

        # get depth data ## channel = None if we want to measure depth, to visualize can be same as image
        depth_image = ZedCameraSensor.get_from_buffer(os.path.join('/home/osu/adc_software/python/Lidar/sensors_working_2022/sensors_working_v2/sensors','zed_depth_map'),
                                                      image_shape=(self.img_h, self.img_w, self.img_c))

        # convert depth map to point cloud format
        depth_image = np.reshape(depth_image, (self.img_w * self.img_h, 3))
        depth_image = depth_image[:, 0:3]
        depth_image = np.array(depth_image[np.logical_not(np.isnan(depth_image[:, 0])), :])
        mask = depth_image[:, 2] < self.max_depth
        depth_image = depth_image[mask]

        # get the lidar point clods
        lidar_points = CeptonLiDAR.get_from_buffer(count=self.count)
        lidar_out = lidar_points

        pcd = np.vstack(lidar_out)

        #pcd = np.vstack((pcd,depth_image))

        data_dict = {

            'image': image_frame,
            'depth': depth_image,
            'point_cloud': (pcd)

        }

        return data_dict


if __name__ == "__main__":
    sensor = Sensors()
    time.sleep(2)

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

        #pcd1 = frame[2]
        #print(pcd1.shape)

        print(f'Total time = {time.perf_counter() - start}')

        cv2.imshow("image", data_dict['image'])
        #cv2.imshow("depth", data_dict['depth'])

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

