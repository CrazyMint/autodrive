# Imports
import time
import cv2
from lidar_final import CeptonLiDAR
from camera_final import ZedCameraSensor

##################################################################
# SENSOR INTERFACE CLASS
##################################################################


class Sensors:

    def __init__(self,
                 camera_resolution: str = '1080',
                 fps: int = 30,
                 camera_view: str = 'left',
                 include_depth: bool = True):

        # Initialize Camera
        self.camera = ZedCameraSensor(camera_resolution, fps, camera_view, include_depth=include_depth)

        self.include_depth = include_depth

        # start camera sensors
        self.camera.start()

        # get image sizes
        self.img_h = self.camera.image_height
        self.img_w = self.camera.image_width
        self.img_c = self.camera.num_channels
        print(self.img_h,self.img_w,self.img_c)

        # Initialize LiDAR
        self.lidar = CeptonLiDAR(pcap_path = None,
                                 visualize = False,
                                 sensor_number = 0,
                                 count = 2)
        self.lidar.start()

    def get_data(self):

        # get image data
        image_frame = ZedCameraSensor.get_from_buffer('zed_image',
                                                      image_shape=(self.img_h, self.img_w, self.img_c))

        # get depth data ## channel = None if we want to measure depth, to visualize can be same as image
        depth_image = ZedCameraSensor.get_from_buffer('zed_depth_map',
                                                      image_shape=(self.img_h, self.img_w, self.img_c))

        lidar_points = CeptonLiDAR.get_from_buffer()


        return image_frame, depth_image, lidar_points

    def close_sensors(self):
        self.lidar.exit()
        self.camera.exit()


if __name__ == "__main__":
    sensor = Sensors()

    while True:
        start = time.perf_counter()
        #frame = sensor.get_data()
        #print(f'Total time = {time.perf_counter() - start}')

        #pcd1,pcd2,pcd3 = frame[2]
        #print(pcd2.shape)
        #cv2.imshow("image", frame[0])
        #cv2.imshow("depth", frame[1])

        key = cv2.waitKey(30)

        if key == ord('q'):
            break

    sensor.close_sensors()
