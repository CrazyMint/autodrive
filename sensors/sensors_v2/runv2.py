
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


def get_from_buffer_lidar(count: int = 2) -> tuple:
    """
    :return:
    """
    start = time.perf_counter()

    if count > 1:

        # create storage
        pcd_storage = [None] * count

        for idx in range(count):

            filename = os.path.join(os.getcwd(), 'pcd' + str(idx + 1))
            pcd_storage[idx] = mmap_read_lidar(filename)

        return pcd_storage

    elif count == 1:
        filename = os.path.join(os.getcwd(), 'pcd_latest')
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
    if os.path.exists(filename) and os.path.basename(filename) == 'zed_image':
        # grab and return the memory mapped data[ie: np.array]
        zed_img_path = np.memmap(filename, dtype='uint8', mode='c', shape=image_shape)

        return zed_img_path

    elif os.path.exists(filename) and os.path.basename(filename) == 'zed_depth_map':

        # grab and return the memory mapped data[ie: np.array]
        zed_img_path = np.memmap(filename, dtype='float32', mode='c', shape=image_shape)

        return zed_img_path

    else:
        raise FileNotFoundError


def get_from_buffer_camera(filename: str, image_shape: tuple) -> tuple:
    """
    :return:
    """
    start = time.perf_counter()

    # read 3 point clouds
    filename = os.path.join(os.getcwd(), filename)
    image = mmap_read_camera(filename, image_shape)
    #print(f'Total time = {time.perf_counter() - start}')

    return image


def get_data(img_h=1080, img_w=1920, img_c=4):
    # 720 (720, 1280 )
    # 1080 (1080, 1920)
    # 2K (1242 2208)

    # get image data
    image_frame = get_from_buffer_camera('zed_image', image_shape=(img_h, img_w, img_c))[:, :, 0:3]

    # get depth data ## channel = None if we want to measure depth, to visualize can be same as image
    depth_image = get_from_buffer_camera('zed_depth_map', image_shape=(img_h, img_w, 4))[:, :, 0:3]
    depth_image = np.reshape(depth_image, (img_h * img_w, 3))

    #depth_image = np.array(depth_image[np.logical_not(np.isnan(depth_image[:, 0])), :])
    temp_nan = np.logical_not(np.isnan(depth_image[:, 0]))
    depth_image = depth_image[temp_nan]

    temp_nan = np.logical_not(np.isinf(depth_image[:, 0]))
    depth_image = depth_image[temp_nan]

    mask = depth_image[:, 2] < 10
    depth_image = depth_image[mask]

    lidar_points = get_from_buffer_lidar(2)

    pcd_all= lidar_points
    pcd_all = np.vstack(pcd_all)


    data_dict = {

        'image': image_frame,
        'depth': depth_image,
        'point_cloud': pcd_all

    }

    return data_dict


def save_image(path: str, filename: str, image: np.array) -> None:
    """
    Args:
        path:
        filename:
        image:

    Returns:
    """
    filename = os.path.join(path, filename)
    np.save(filename, image)


def save_depth(path: str,
               filename: str,
               image: np.array,
               image_shape: tuple,
               max_depth: int) -> None:
    """
    Args:
        path:
        filename:
        image:
        image_shape:
        max_depth:
    Returns:
    """

    filename = os.path.join(path, filename)
    image = np.reshape(image, (image_shape[0] * image_shape[1], 3))
    image = image[:, 0:3]
    image = np.array(image[np.logical_not(np.isnan(image[:, 0])), :])
    mask = image[:, 2] < max_depth
    image = image[mask]
    np.save(filename, image)


def save_lidar(path: str, filename: str, point_cloud: np.array) -> None:
    """
    Args:
        path:
        filename:
        point_cloud:
    Returns:
    """
    filename = os.path.join(path, filename)
    np.save(filename, point_cloud)




if __name__ == "__main__":

    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    reset = True

    counter = 0
    while True:

        frame = get_data()
        #print(f'Total time = {time.perf_counter() - start}')

        """
        # save image
        
        """

        image = frame["image"]

        #cv2.imshow("image", image)
        #print(frame['depth'].shape)

        #cv2.imshow("depth", frame["depth"])

        pcd.points = o3d.utility.Vector3dVector(frame["depth"])
        #pcd = pcd.voxel_down_sample(voxel_size=0.00005)

        vis.update_geometry(pcd)
        if reset:
            vis.reset_view_point(True)
            reset = False
        vis.poll_events()
        vis.update_renderer()



        key = cv2.waitKey(10)

        if key == ord('q'):
            break

        if key == ord('g'):
            image_filename = 'image' + str(counter)
            save_image(path='save_data/images', filename=image_filename, image=frame['image'])

            # save depth
            depth_filename = 'depth' + str(counter)
            save_depth(path='save_data/depth',
                       filename=depth_filename,
                       image=frame['depth'],
                       image_shape=(1080, 1920), # 2K (1242 2208)
                       max_depth=20)

            # save point cloud
            depth_filename = 'pcd' + str(counter)
            save_lidar(path='save_data/point_cloud', filename=depth_filename, point_cloud=frame['point_cloud'])

            counter += 1