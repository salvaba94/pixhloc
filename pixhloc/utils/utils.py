import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np


def setup_logger():
    """Function to setup logging."""
    formatter = logging.Formatter(
        fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    logger = logging.getLogger()
    if not len(logger.handlers):
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False

    numexpr_logger = logging.getLogger("numexpr")
    numexpr_logger.setLevel(logging.ERROR)

    filelock_logger = logging.getLogger("filelock")
    filelock_logger.setLevel(logging.ERROR)

    urllib3_logger = logging.getLogger("urllib3")
    urllib3_logger.setLevel(logging.ERROR)

    h5py_logger = logging.getLogger("h5py")
    h5py_logger.setLevel(logging.ERROR)

    warnings.filterwarnings(
        "ignore", category=FutureWarning, module="transformers.models.vit.feature_extraction_vit"
    )

    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.data.dataloader")


def arr_to_str(a):
    return ";".join([str(x) for x in a.reshape(-1)])


def log_data_dict(data_dict: Dict[str, Any]):
    """Function to log data dictionary.

    Args:
        data_dict (Dict[str, Any]): Data dictionary.
    """
    logging.info("=" * 80)
    logging.info("DATA:")
    logging.info("=" * 80)
    for ds, ds_vals in data_dict.items():
        logging.info(ds)
        for scene in ds_vals.keys():
            logging.info(f"  {scene}: {len(data_dict[ds][scene])} imgs")


def get_data_from_dict(data_dir: str) -> Dict[str, Any]:
    """Function to get data from a dictionary.

    Args:
        data_dir (str): Path to data directory.

    Returns:
        Dict[str, Any]: Description of returned object.
    """
    data_dict = {}
    with open(os.path.join(data_dir, "sample_submission.csv"), "r") as f:
        for i, l in enumerate(f):
            # Skip header.
            if l and i > 0:
                image, dataset, scene, _, _ = l.strip().split(",")
                if dataset not in data_dict:
                    data_dict[dataset] = {}
                if scene not in data_dict[dataset]:
                    data_dict[dataset][scene] = []
                data_dict[dataset][scene].append(image)

    log_data_dict(data_dict)
    return data_dict


def get_data_from_dir(data_dir: str, mode: str) -> Dict[str, Any]:
    """Function to get data from a directory.

    Args:
        data_dir (str): Path to data directory.
        mode (str): Mode (train or test).

    Raises:
        ValueError: Invalid mode.

    Returns:
        Dict[str, Any]: Data dictionary.
    """
    if mode not in {"train", "test"}:
        raise ValueError(f"Invalid mode: {mode}")

    data_dict = {}
    datasets = [
        x
        for x in os.listdir(os.path.join(data_dir, mode))
        if os.path.isdir(os.path.join(data_dir, mode, x))
    ]
    for dataset in datasets:
        if dataset not in data_dict:
            data_dict[dataset] = {}

        image_dir = os.path.join(data_dir, mode, dataset, "images")
        scene = dataset
        data_dict[dataset][scene] = []
        for img in os.listdir(image_dir):
            data_dict[dataset][scene].append(os.path.join(mode, dataset, "images", img))

    log_data_dict(data_dict)
    return data_dict


def create_submission(out_results: Dict[str, Any], data_dict: Dict[str, Any], fname: str):
    """Function to create a submission file.

    Args:
        out_results (Dict[str, Any]): Estimated poses.
        data_dict (Dict[str, Any]): Data dictionary.
        fname (str): Output file name.
    """
    n_images_total = 0
    n_images_written = 0
    with open(fname, "w") as f:
        f.write("image_path,dataset,scene,rotation_matrix,translation_vector\n")
        for dataset in data_dict:
            res = out_results.get(dataset, {})
            for scene in data_dict[dataset]:
                scene_res = res[scene] if scene in res else {"R": {}, "t": {}}
                for image in data_dict[dataset][scene]:
                    n_images_total += 1
                    if image in scene_res:
                        # print(image)
                        R = np.array(scene_res[image]["R"]).reshape(-1)
                        T = np.array(scene_res[image]["t"]).reshape(-1)
                        n_images_written += 1
                    else:
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    f.write(f"{image},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n")

    logging.info(f"Written {n_images_written} of {n_images_total} images to submission file.")


def create_submission_mod(out_results: Dict[str, Any], data_dict: Dict[str, Any], fname: str, data_dir: str):

    import pandas as pd
    df = pd.read_csv(Path(data_dir).joinpath("sample_submission.csv"))
    with open(fname, 'w') as f:
        f.write('image_path,dataset,scene,rotation_matrix,translation_vector\n')
        for dataset in data_dict:
            if dataset in out_results:
                res = out_results[dataset]
            else:
                res = {}
            for scene in data_dict[dataset]:
                if scene in res:
                    scene_res = res[scene]
                else:
                    scene_res = {"R":{}, "t":{}}
                for image in data_dict[dataset][scene]:
                    prev_image = image
                    #image = image.split('/')[-1]
                    if image in scene_res:
                        R = scene_res[image]['R'].reshape(-1)
                        T = scene_res[image]['t'].reshape(-1)
                    else:
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    if(image not in df['image_path'].tolist()):
                        f.write(f'{prev_image},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n')
                    else:
                        f.write(f'{image},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n')





class DataPaths:
    def __init__(self, data_dir: str, output_dir: str, dataset: str, scene: str, mode: str):
        """Class to store paths.

        Args:
            data_dir (str): Path to data directory.
            output_dir (str): Path to output directory.
            dataset (str): Dataset name.
            scene (str): Scene name.
            mode (str): Mode (train or test).
        """
        if mode not in {"train", "test"}:
            raise ValueError(f"Invalid mode: {mode}")

        self.input_dir = Path(f"{data_dir}/{mode}/{dataset}")
        self.scene_dir = output_dir / dataset / scene
        self.image_dir = self.scene_dir / "images"
        self.dataset = dataset
        self.scene = scene

        # TODO: Uncomment if reference model is desired
        # self.reference_model = self.input_dir / "smf" # There is a typo with the data (should be sfm)
        # if not self.reference_model.exists():
        #     self.reference_model = None
        self.reference_model = None

        self.sfm_dir = self.scene_dir / "sparse"
        self.pairs_path = self.scene_dir / "pairs.txt"
        self.features_retrieval = self.scene_dir / "features_retrieval.h5"
        self.features_path = self.scene_dir / "features.h5"
        self.matches_path = self.scene_dir / "matches.h5"

        # for rotation matching
        self.rotated_image_dir = self.scene_dir / "images_rotated"
        self.rotated_features_path = self.scene_dir / "features_rotated.h5"

        # for image cropping
        self.cropping_dir = self.scene_dir / "cropping"
        self.cropping_pairs_path = self.cropping_dir / "pairs.txt"
        self.cropping_features_retrieval = self.cropping_dir / "features_retrieval.h5"
        self.cropping_features_path = self.cropping_dir / "features.h5"
        self.cropping_matches_path = self.cropping_dir / "matches.h5"

        # for pixsfm
        self.cache = output_dir / "cache"

        # create directories
        self.scene_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.sfm_dir.mkdir(parents=True, exist_ok=True)
        self.rotated_image_dir.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.cropping_dir.mkdir(parents=True, exist_ok=True)
