"""Script to investigate the previous Radar frame as a baseline for the evaluation of RadarNeRF."""

import numpy as np
import argparse
import h5py
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from nerfstudio.model_components.gospa import calculate_gospa
from scipy.stats import wasserstein_distance_nd

import pyquaternion
from nuscenes.nuscenes import NuScenes as NuScenesDatabase
from nuscenes.utils.data_classes import RadarPointCloud

# from vod.configuration import KittiLocations
# from vod.frame import FrameDataLoader, FrameTransformMatrix

from zod import ZodSequences

ROOT_PATH = "/home/s0001331/"
NUSCENES_PATH = ROOT_PATH + "repos/data/nuscenes/"
VOD_PATH = ROOT_PATH + "repos/data/view-of-delft-dataset/ViewofDelft"
ZOD_PATH = ROOT_PATH + "mnt/staging/dataset_donation/round_2"

DATASET_SEQUENCE_NUMBER_FILL_MAP = {"nuscenes": 4, "vod": 2, "zod": 6}

#NUSCENES_RADARS = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK_LEFT", "BACK_RIGHT")
NUSCENES_RADARS = ("FRONT")

VOD_FRAME_STR_LEN = 5
VOD_DATA_FREQUENCY = 10
SEQUENCE_TO_FRAME_MAP = {"00": range(0, 544), "01": range(545, 1312),
                         "02": range(1314, 1803), "03": range(1803, 2200),
                         "04": range(2220, 2532), "05": range(2532, 2798),
                         "06": range(2798, 3277), "07": range(3277, 3575),
                         "08": range(3575, 3610), "09": range(3610, 4048),
                         "10": range(4049, 4387), "11": range(4387, 4652),
                         "12": range(4653, 5086), "13": range(6334, 6571),
                         "14": range(6571, 6759), "15": range(6759, 7543),
                         "16": range(7543, 7900), "17": range(7900, 8198),
                         "18": range(8198, 8481), "19": range(8482, 8749),
                         "20": range(8749, 9096), "21": range(9096, 9518),
                         "22": range(9518, 9776), "23": range(9776, 9930)}

def _parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence", "-s", type=int, default=103)

    parser.add_argument("--dataset", "-d", type=str, default="nuscenes")

    args = parser.parse_args()

    return args

def get_radar_scans(sequence: str, dataset: str):
    # if dataset == "nuscenes":
    #     return get_radar_scans_nuscenes(sequence)
    # elif dataset == "vod":
    #     return get_radar_scans_vod(sequence)
    if dataset == "zod":
        return get_radar_scans_zod(sequence)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")

def get_radar_scans_nuscenes(sequence: str):

    nusc = NuScenesDatabase(
        version="v1.0-mini",
        dataroot=NUSCENES_PATH,
        verbose=False,
    )

    scene = nusc.get("scene", nusc.field2token("scene", "name", "scene-" + sequence)[0])

    radar_pcs, poses, times = [], [], []
    first_sample = nusc.get("sample", scene["first_sample_token"])

    #for radar in NUSCENES_RADARS:
    radar = NUSCENES_RADARS
    for radar_data in _find_all_sample_data_nuscenes(nusc, first_sample["data"]["RADAR_" + radar]):
        calibrated_sensor_data = nusc.get("calibrated_sensor", radar_data["calibrated_sensor_token"])
        ego_pose_data = nusc.get("ego_pose", radar_data["ego_pose_token"])
        ego_pose = _rotation_translation_to_pose(ego_pose_data["rotation"], ego_pose_data["translation"])
        radar_pose = _rotation_translation_to_pose(
            calibrated_sensor_data["rotation"], calibrated_sensor_data["translation"]
        )
        pose = ego_pose @ radar_pose

        # load point clouds
        filename = NUSCENES_PATH + radar_data["filename"]
        pc = RadarPointCloud.from_file(filename).points.reshape((18, -1)).T.astype(np.float32)
        radar_pcs.append(pc)
        times.append(radar_data["timestamp"] / 1e6)
        poses.append(pose)

    return radar_pcs, poses, np.array(times, dtype=np.float64)

# def get_radar_scans_vod(sequence: str):

#     kitti_locations = KittiLocations(root_dir=VOD_PATH)
#     frames = SEQUENCE_TO_FRAME_MAP[sequence]

#     camera_times = np.zeros((len(frames)))
#     radar_scans = np.zeros((len(frames)), dtype=np.ndarray)
#     camera_to_odom = np.zeros((len(frames), 4, 4))
#     radar_to_cam = np.zeros((len(frames), 4, 4))
#     for frame in frames:
#         camera_times[frame - frames[0]] = (frame - frames[0]) / VOD_DATA_FREQUENCY
#         frame_str = str(frame).zfill(VOD_FRAME_STR_LEN)
#         frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_str)
#         transforms = FrameTransformMatrix(frame_data)
#         camera_to_odom[frame - frames[0]] = transforms.t_odom_camera
#         radar_to_cam[frame - frames[0]] = transforms.t_camera_radar
#         radar_scans[frame - frames[0]] = frame_data.radar_data

#     times = camera_times.astype(np.float64)
#     poses = (camera_to_odom @ radar_to_cam).astype(np.float32)


    return radar_scans, poses, times

def get_radar_scans_zod(sequence: str):

    zod_parser = ZodSequences(dataset_root=ZOD_PATH, version="full", mp=False)
    assert sequence in zod_parser.get_all_ids(), f"Sequence {sequence} not found in dataset"

    zod_seq = zod_parser[sequence]

    start_time = np.float64(zod_seq.info.start_time.timestamp() * 1e9)
    end_time = np.float64(zod_seq.info.end_time.timestamp() * 1e9)

    radar_file = ZOD_PATH + "/sequences/" + sequence + "/internal/radar_flr/" + "000005_romeo_FC_2022-08-30T09:46:09.180289Z.hdf5"

    radar_pcs, times, poses = [], [], []

    with h5py.File(radar_file, "r") as f:
        h5_data = f["data"]
        timestamps = np.array(h5_data["timestamp/nanoseconds/value"], dtype=np.float64)
        timestamps = timestamps[(timestamps > start_time) & (timestamps < end_time)]
        detections = h5_data["detections"][...]

        for scan_idx, timestamp in enumerate(timestamps):
            ranges_h5 = detections["range"][scan_idx]
            azimuths_h5 = detections["angle"][scan_idx]
            elevation_angles_h5 = detections["elevation_angle"][scan_idx]
            snrs_h5 = detections["amplitude"][scan_idx]
            rcss_h5 = detections["rcs"][scan_idx]
            range_rates_h5 = detections["range_rate"][scan_idx]
            mode_h5 = detections["mode"][scan_idx]
            quality_h5 = detections["quality"][scan_idx]
            lat_pos = h5_data["mounting_position/lat_pos/meters/value"][scan_idx]
            lon_pos = h5_data["mounting_position/long_pos/meters/value"][scan_idx]

            times.append((timestamp / 1e9))

            ego_pose = zod_seq.oxts.get_poses(times[-1])
            poses.append(ego_pose @ np.array([[1, 0, 0, lon_pos], [0, 1, 0, lat_pos], [0, 0, 1, 0], [0, 0, 0, 1]]))

            ranges, azimuths, elevation_angles, snrs, rcss, range_rates, mode, quality = [], [], [], [], [], [], [], []
            for index in range(len(ranges_h5)):
                ranges.append(ranges_h5[index][0][0])
                azimuths.append(azimuths_h5[index][0][0])
                elevation_angles.append(elevation_angles_h5[index][0][0])
                snrs.append(snrs_h5[index][0][0])
                rcss.append(rcss_h5[index][0][0])
                range_rates.append(range_rates_h5[index][0][0])
                mode.append(mode_h5[index][0][0])
                quality.append(quality_h5[index][0][0])

            x_positions = ranges * np.cos(elevation_angles) * np.cos(azimuths)
            y_positions = ranges * np.cos(elevation_angles) * np.sin(azimuths)
            z_positions = ranges * np.sin(elevation_angles)

            valid_indices = np.array(quality) < 3
            cloud = np.stack(
                [
                    x_positions,
                    y_positions,
                    z_positions,
                    snrs,
                    rcss,
                    range_rates,
                    mode,
                    quality,
                ],
                axis=-1,
            )[valid_indices]

            radar_pcs.append(cloud)

    return radar_pcs, poses, times

def get_metrics_between_radar_scans(radar_scans, poses, times):

        chamfers, gospas = [], []

        for i in range(1, len(radar_scans)):
            chamfer = chamfer_distance(radar_scans[i - 1][:, :3], radar_scans[i][:, :3])
            gospa = calculate_gospa(radar_scans[i - 1][:, :3], radar_scans[i][:, :3], 1, 1)[0]
            chamfers.append(chamfer)
            gospas.append(gospa)

        return chamfers, gospas

def _find_all_sample_data_nuscenes(nusc, sample_data_token):
    """Finds all sample data from a given sample token."""
    curr_token = sample_data_token
    sd = nusc.get("sample_data", curr_token)
    # Rewind to first sample data
    while sd["prev"]:
        curr_token = sd["prev"]
        sd = nusc.get("sample_data", curr_token)
    # Forward to last sample data
    all_sample_data = [sd]
    while sd["next"]:
        curr_token = sd["next"]
        sd = nusc.get("sample_data", curr_token)
        all_sample_data.append(sd)
    return all_sample_data

def _rotation_translation_to_pose(r_quat, t_vec):
    """Convert quaternion rotation and translation vectors to 4x4 matrix"""
    pose = np.eye(4)
    pose[:3, :3] = pyquaternion.Quaternion(r_quat).rotation_matrix
    pose[:3, 3] = t_vec
    return pose

def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist

def emd_distance(x, y):
    # Compute Wasserstein distance
    distance = wasserstein_distance_nd(x, y)
    return distance

def calculate_metrics_baseline(radar_scans, poses, times):
    chamfers, gospas = get_metrics_between_radar_scans(radar_scans, poses, times)
    print(f"Chamfer distances are between {np.min(chamfers)} (index {np.argmin(chamfers)}) and {np.max(chamfers)} (index {np.argmax(chamfers)})")
    print(f"GOSPA distances are between {np.min(gospas)} (index {np.argmin(gospas)}) and {np.max(gospas)} (index {np.argmax(gospas)})")

    plt.hist(chamfers, bins=100)
    plt.show()

    plt.hist(gospas, bins=100)
    plt.show()

def main():
    """Main function to evaluate RadarNeRF."""
    args = _parse_arguments()

    sequence = str(args.sequence).zfill(DATASET_SEQUENCE_NUMBER_FILL_MAP[args.dataset])

    radar_scans, poses, times = get_radar_scans(sequence, args.dataset)

    calculate_metrics_baseline(radar_scans, poses, times)

if __name__ == "__main__":
    main()