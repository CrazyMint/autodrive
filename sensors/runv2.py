
#############################################################
# Stand Alone Buffer Version
# these files will be able to function alone and still pick up all the files
# this version is just a backup incase we never get the other version to integrate well
#############################################################
import numpy as np
import time
import cv2
import open3d as o3d
import os


def mmap_read_lidar(filename: str) -> np.array:
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

        #print("pcd_path_nest.shape", pcd_path.shape)

        # pcd_path = np.reshape(pcd_path, (int(pcd_path.shape[0] / 3), 3))
        return pcd_path

    else:
        raise FileNotFoundError


def get_from_buffer_lidar(count: int = 3) -> tuple:
    """
    :return:
    """
    start = time.perf_counter()

    if count > 1:

        # create storage
        pcd_storage = [None] * count

        for idx in range(count):

            filename = os.path.join('/home/osu/Models/AutoDriveYr1/highway_challenge/sensors/sensors', 'pcd' + str(idx + 1))
            pcd_storage[idx] = mmap_read_lidar(filename)

        return pcd_storage

    elif count == 1:
        filename = os.path.join('/home/osu/Models/AutoDriveYr1/highway_challenge/sensors/sensors', 'pcd_latest')
        pcd_latest = mmap_read_lidar('pcd_latest')

        return pcd_latest

    else:

        raise ValueError('check count')
    # print(f'Total time = {time.perf_counter() - start}')


def mmap_read_camera(filename: str, image_shape: tuple) -> np.array:
    """
    This function is used to read the image/depth map information from the memorymapped location
    :param filename: [string] the name of the file associated with the data
    :param image_shape: [tuple] shape of the image data to be retrieved
    :return:
    """
    if os.path.exists(filename):
        # grab and return the memory mapped data[ie: np.array]
        zed_img_path = np.memmap(filename, dtype='uint8', mode='c', shape=image_shape)
        return zed_img_path
    else:
        raise FileNotFoundError


def get_from_buffer_camera(filename: str, image_shape: tuple) -> tuple:
    """
    :return:
    """
    start = time.perf_counter()

    # read 3 point clouds
    filename = os.path.join('/home/osu/Models/AutoDriveYr1/highway_challenge/sensors/sensors', filename)
    image = mmap_read_camera(filename, image_shape)
    #print(f'Total time = {time.perf_counter() - start}')

    return image


def get_data(img_h=1080, img_w= 1920, img_c=3):

    # get image data
    image_frame = get_from_buffer_camera('zed_image', image_shape=(img_h, img_w, img_c))

    # get depth data ## channel = None if we want to measure depth, to visualize can be same as image
    depth_image = get_from_buffer_camera('zed_depth_map', image_shape=(img_h, img_w, img_c))

    lidar_points = get_from_buffer_lidar()

    pcd_all= lidar_points
    pcd_all = np.vstack(pcd_all)

    data_dict = {

        'image': image_frame,
        'depth': depth_image,
        'point_cloud': pcd_all

    }

    return data_dict




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
