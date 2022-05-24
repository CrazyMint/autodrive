
#############################################################
# Stand Alone Buffer Version
# these files will be able to function alone and still pick up all the files
# this version is just a backup incase we never get the other version to integrate well
#############################################################
import numpy as np
import time
import cv2
import open3d as o3d


def mmap_read(filename: str) -> np.array:
    """
    :param filename:
    :return:
    """

    pcd_path = np.memmap(filename, dtype='float64', mode='c', shape=(50000, 3))

    return pcd_path


def get_from_buffer() -> tuple:
    """
    if we are going to use this version we need to updat this file
    this file needs to have the mem
    :return:
    """
    start = time.perf_counter()

    # read 3 point clouds
    pcd_1 = mmap_read('/home/osu/adc_software/python/Lidar/sensors_working_2022/sensors/pcd1')
    pcd_2 = mmap_read('/home/osu/adc_software/python/Lidar/sensors_working_2022/sensors/pcd2')
    pcd_3 = mmap_read('/home/osu/adc_software/python/Lidar/sensors_working_2022/sensors/pcd3')
    print(f'Total time = {time.perf_counter() - start}')

    return pcd_1, pcd_2, pcd_3

def get_data():

    # get image data
    #image_frame = ZedCameraSensor.get_from_buffer('zed_image',
    #                                              image_shape=(self.img_h, self.img_w, self.img_c))

    # get depth data ## channel = None if we want to measure depth, to visualize can be same as image
    #depth_image = ZedCameraSensor.get_from_buffer('zed_depth_map',
    #                                             image_shape=(self.img_h, self.img_w, self.img_c))

    lidar_points = get_from_buffer()

    pcd1,pcd2,pcd3 = lidar_points
    pcd_all = np.vstack((pcd1,pcd2,pcd3))

    #mask = pcd_all[:,1] == 0

    #pcd_all = pcd_all[mask,:]
    #print(pcd_all.shape)

    return pcd_all




if __name__ == "__main__":

    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    reset = True

    while True:

        frame = get_data()
        #print(f'Total time = {time.perf_counter() - start}')



        #cv2.imshow("image", frame[0])
        #cv2.imshow("depth", frame[1])

        pcd.points = o3d.utility.Vector3dVector(frame)

        vis.update_geometry(pcd)
        if reset:
            vis.reset_view_point(True)
            reset = False
        vis.poll_events()
        vis.update_renderer()



        key = cv2.waitKey(100)

        if key == ord('q'):
            break
