# Imports
import time

import cepton_sdk
import cv2

# from cepton_sdk_redist.python.cepton_sdk.common import *
import open3d as o3d  # might need to remove
import os
import threading
from threading import Thread, Lock
from abc import ABC, abstractmethod
import numpy as np


##################################################################
# EXCEPTIONS
##################################################################


class LiDARConnectionError(Exception):
    """
    Exception Called When:
    1. Zed Camera Fails to Open -- try unplugging and replugging in camera
    """
    pass


##################################################################
# ABSTRACT BASE CLASS
##################################################################


class LiDARSensor(ABC):

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def get_frame(self):
        pass

    @abstractmethod
    def _update(self):
        pass

    @abstractmethod
    def exit(self):
        pass


##################################################################
# LiDAR SENSOR CLASS
##################################################################


class CeptonLiDAR(LiDARSensor):
    """
    CeptonLiDAR class is used to interact with the LiDAR Sensor
    Functionality Include:
        1. Initialize and Start the Sensor
        2. Continuously obtain real time point cloud information
        3. Visualize the real time point cloud information
        4. Write point cloud information to a buffer
        5. Read point cloud information from buffer
        6. Kill and Exit the sensor
    """

    def __init__(self,
                 pcap_path: str = None,
                 visualize: bool = False,
                 sensor_number: int = 0,
                 count: int = 5,
                 use_buffer: bool = True):
        """
        :param pcap_path:
        :param visualize:
        :param sensor_number:
        :param count:
        """
        # **** Write a small comment for each variable
        # ** so anyone can know what each one does later on

        self._read_path = pcap_path
        self.listener = None
        self.sensor = None
        self.started = None  # indicate if sensor has been started or not

        self.frame_pcd = o3d.geometry.PointCloud()
        self.np_points = None
        self.raw_points = None
        self.sensor_number = sensor_number  # specify the sensor number ie: lidar sensor 1, 2, ..

        self.visualize = visualize
        self.bird_eye_view = np.zeros([1, 1], dtype=np.uint8)
        self.count = count

        self.acc_np_points_list = [None] * self.count
        self.acc_np_points = np.zeros((self.count, 3))
        self.use_buffer = use_buffer
        self.acc_counter = 0

        self.read_lock = Lock()

    def start(self) -> None:
        """
        starts the LiDAR sensor
        :return: None
        """

        if self.started:
            print("[info] LiDAR Sensor: ON")
            return None

        try:
            # initialize the lidar sensor
            if self._read_path is not None:
                cepton_sdk.initialize(capture_path=self._read_path, enable_wait=True)
            else:
                cepton_sdk.initialize(enable_wait=True)

            # if visualize, do that in a separate thread
            if self.visualize:
                self.visualize_thread = Thread(target=self._visualize, args=())
                self.visualize_thread.start()

            # get sensor information
            self.sensor = cepton_sdk.Sensor.create_by_index(self.sensor_number)

            # get and update data
            self.listener = cepton_sdk.FramesListener()
            print(f"[info] connected to sensor {self.sensor.serial_number}")
            print('[info] LiDAR Sensor: ON')
            self.started = True

            self.thread = Thread(target=self._update, args=())
            self.thread.start()

        except:
            raise LiDARConnectionError('LiDAR Not Connected')

    @staticmethod
    def _scale_to_255(arr: np.array,
                      min: int,
                      max: int,
                      dtype=np.uint8) -> np.array:
        """
        :param arr:
        :param min:
        :param max:
        :param dtype:
        :return:
        """
        return (((arr - min) / float(max - min)) * 255.0).astype(dtype)

    def _visualize(self):
        """
        :return:
        """
        # set the dimensions
        left_right_range = (-40, 40)
        front_back_range = (-10, 40)

        while self.visualize:
            if self.np_points is not None:
                self.read_lock.acquire()

                # print("visualizing...")
                x_points = self.np_points[:, 1]
                y_points = -self.np_points[:, 0]
                z_points = self.np_points[:, 2]

                self.read_lock.release()

                # FILTER - To return only indices of points within desired cube
                # Three filters for: Front-to-back, side-to-side, and height ranges
                # Note left side is positive y axis in LIDAR coordinates
                f_filt = np.logical_and((x_points > front_back_range[0]), (x_points < front_back_range[1]))
                s_filt = np.logical_and((x_points > -left_right_range[1]), (x_points < -left_right_range[0]))
                filter = np.logical_and(f_filt, s_filt)
                indices = np.argwhere(filter).flatten()

                # KEEPERS
                x_points = x_points[indices]
                y_points = y_points[indices]
                z_points = z_points[indices]
                res = 0.1
                # CONVERT TO PIXEL POSITION VALUES - Based on resolution
                x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
                y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

                x_img -= int(np.floor(left_right_range[0] / res))
                y_img += int(np.ceil(front_back_range[1] / res))

                height_range = (-2, 0.5)  # bottom-most to upper-most

                # CLIP HEIGHT VALUES - to between min and max heights
                pixel_values = np.clip(a=z_points,
                                       a_min=height_range[0],
                                       a_max=height_range[1])
                # RESCALE THE HEIGHT VALUES - to be between the range 0-255
                pixel_values = CeptonLiDAR._scale_to_255(pixel_values, min=height_range[0], max=height_range[1])

                x_max = 1 + int((left_right_range[1] - left_right_range[0]) / res)
                y_max = 1 + int((front_back_range[1] - front_back_range[0]) / res)
                im = np.zeros([y_max, x_max], dtype=np.uint8)

                # FILL PIXEL VALUES IN IMAGE ARRAY
                im[y_img, x_img] = 255
                self.bird_eye = im
                time.sleep(0.1)

    def get_frame(self) -> np.array:
        """
        :return:
        """
        self.read_lock.acquire()
        frame = self.np_points.copy()
        self.read_lock.release()

        return frame

    def get_accumulate_frame(self) -> list:
        """

        """
        self.read_lock.acquire()
        acc_frame = self.acc_np_points_list.copy()
        self.read_lock.release()

        return acc_frame

    def get_pcd(self) -> np.array:
        """
        :return:
        """
        self.read_lock.acquire()
        pcd = self.frame_pcd
        self.read_lock.release()

        return pcd

    def accumulate_frame(self):

        if self.count > 1:

            if self.acc_counter < self.count:
                self.acc_np_points_list[self.acc_counter] = self.np_points
                self.acc_counter += 1
            else:
                self.acc_counter = 0
                self.acc_np_points_list[self.acc_counter] = self.np_points
                self.acc_counter += 1


    def _update(self) -> None:
        """
        :return:
        """
        print('[info] updating point cloud')
        current_index = 0
        while self.started:
            # get a set of points
            points_dict = self.listener.get_points()
            # filter by serial number and get the raw data for given sensor
            raw_points = cepton_sdk.combine_points(points_dict[self.sensor.serial_number])
            # update x,y,z point cloud locations
            self.read_lock.acquire()
            self.raw_points = raw_points
            pcd_data = np.array(raw_points.positions)
            self.np_points = pcd_data  # np.array(raw_points.positions)
            if self.use_buffer:
                current_index = CeptonLiDAR.send_to_buffer(pcd_data, current_index, self.count)
            else:
                self.accumulate_frame()
            self.frame_pcd.points = o3d.utility.Vector3dVector(raw_points.positions)
            self.read_lock.release()

            time.sleep(0.08)

    def exit(self) -> None:
        """
        :return:
        """
        self.started = False
        self.visualize = False
        cv2.destroyAllWindows()

        del self.listener
        print('[info] sensor terminated')

    @staticmethod
    def mmap_write(filename: str,
                   data: np.array) -> None:
        """
        This function is used to write dato to memory map
        Typical point cloud output from the lidar is [19000, 3], but the system is variable
        If the point cloud size is bigger than the memory map size it causes an issue
        To solve this issue we create a memory map of size 60K (experimentally created size)
        .. and we use np.nan to seperate the real values from the lidar.
        When we read the files we need to filter out the np.nan numbers and only output the
        .. true point cloud points.
        :param filename:
        :param data:
        :return:
        """
        memmap_size = 50000
        # set up memmap
        if not os.path.exists(filename):
            pcd_path = np.memmap(filename, dtype='float64', mode='w+', shape=(memmap_size, 3))
        else:
            pcd_path = np.memmap(filename, dtype='float64', mode='r+', shape=(memmap_size, 3))

        # insert data
        print('initial data: ', data.shape[0])
        if data.shape[0] < memmap_size:
            # create and fill an dummy array of size 60K
            temp = np.empty((memmap_size, 3), dtype=np.float64)
            temp[:] = np.nan
            temp[:data.shape[0], 0:data.shape[1]] = data
            pcd_path[:] = temp[:]
        else:
            print('Point Cloud Exceeds Memory Map Allocated Size, ', data.shape[0])

    @staticmethod
    def mmap_read(filename: str) -> np.array:
        """
        TO DO:: HOW TO FILTER OUT np.NAN seems to not work well wth memorymap
        :param filename:
        :return:
        """

        if os.path.exists(filename):
            # read the file
            pcd_path = np.memmap(filename, dtype='float64', mode='c', shape=(50000, 3))
            # filter out numpy.nan values
            # pcd_path = pcd_path[np.logical_not(np.isnan(pcd_path))]

            mask = np.logical_not(np.isnan(pcd_path[:, 0]))

            pcd_path = pcd_path[mask, :]

            print("pcd_path_nest.shape", pcd_path.shape)

            # pcd_path = np.reshape(pcd_path, (int(pcd_path.shape[0] / 3), 3))
            return pcd_path

        else:
            raise FileNotFoundError

    @staticmethod
    def send_to_buffer(data_pcd: np.array,
                       current_index: int = 0,
                       count: int = 3) -> int:
        """
        :param data_pcd:
        :param current_index:
        :param count:
        :return:
        """
        start = time.perf_counter()

        # if the frame count needed > 1 write x many times
        if count > 1:
            index = current_index + 1
            if current_index == count:
                index = 1

            filename = 'pcd' + str(index)
            CeptonLiDAR.mmap_write(filename, data_pcd)

            return index

        elif count == 1:
            filename = 'pcd_latest'
            CeptonLiDAR.mmap_write(filename, data_pcd)

            return 0
        else:
            raise ValueError('count must be greater than or equal to 1')

        # print(f'[info] Total write time: {time.perf_counter() - start}')

    @staticmethod
    def get_from_buffer(count: int = 1) -> tuple:
        """
        :return:
        """
        start = time.perf_counter()

        if count > 1:

            # create storage
            pcd_storage = [None] * count

            for idx in range(count):

                filename = 'pcd' + str(idx + 1)
                pcd_storage[idx] = CeptonLiDAR.mmap_read(filename)

            return pcd_storage

        elif count == 1:
            pcd_latest = CeptonLiDAR.mmap_read('pcd_latest')

            return pcd_latest

        else:

            raise ValueError('check count')
        # print(f'Total time = {time.perf_counter() - start}')



    @staticmethod
    def get_latest_from_buffer() -> tuple:
        """
        :return:
        """
        start = time.perf_counter()

        # read 3 point clouds
        pcd_latest = CeptonLiDAR.mmap_read('pcd1')
        # print(f'Total time = {time.perf_counter() - start}')

        return pcd_latest


if __name__ == "__main__":
    lidar1 = CeptonLiDAR(visualize=True)
    lidar1.start()
    # time.sleep(5)
    #
    # file_tag = 1
    # save_path = "/home/autodrive/Lidar/ADC/python/Lidar/collect_data/"
    #
    # while True:
    #     bird_eye_view = CeptonLiDAR.get_from_buffer()
    #     print(lidar1.np_points.shape)
    #
    #     #cv2.imshow("bird_eye_view", bird_eye_view)

    key = cv2.waitKey(10)

    if key == ord('q'):
        lidar1.exit()
