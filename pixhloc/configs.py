from hloc import match_features

configs = {
    "sift": {
        "features": {
            "model": {"name": "dog"},
            "options": {
                "first_octave": -1,
                "peak_threshold": 0.00667,  # 0.00667, # 0.01,
            },
            "output": "feats-sift",
            "preprocessing": { "grayscale": False, "resize_max": 1600},
        },
        "matches": match_features.confs["NN-ratio"],
    },
    "loftr": {
        "features": None,
        "matches": {
            "output": "matches-loftr",
            "model": {"name": "loftr", "weights": "outdoor"},
            "preprocessing": { "grayscale": False, "resize_max": 840, "dfactor": 8 },  # 1024,
            "max_error": 1,  # max error for assigned keypoints (in px)
            "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
        },
    },
    "xfeat": {
        "features": {
            "model": {"name": "xfeat" },
            "preprocessing": { "grayscale": False, "resize_max": 1600 },
            "output": "feats-xfeat",
            "top_k": 4096,
            "semi_dense": True
        },
        "matches": {
            "model": {"name": "cosine_mlp" },
            "preprocessing": { "grayscale": False, "resize_max": 1600 },
            "output": "matches-cosine-mlp",
            "top_k": 4096,
            "semi_dense": True
        },
    },
    "disk": {
        "features": {
            "output": "feats-disk",
            "model": {
                "name": "disk",
                "max_keypoints": 5000,
            },
            "preprocessing": {
                "grayscale": False,
                "resize_max": 1600,
            },
        },
        "matches": {
            "output": "matches-disk-lightglue",
            "model": {
                "features": "disk",
                "name": "lightglue",
                "weights": "disk_lightglue",
                "filter_threshold": 0.1,
                "width_confidence": -1,
                "depth_confidence": -1,
                "mp": True
            },
        },
    },
    "diskh": {
        "features": {
            "output": "feats-disk",
            "model": {
                "name": "disk",
                "max_keypoints": 5000,
            },
            "preprocessing": {
                "grayscale": False,
                "resize_max": 1600,
            },
        },
        "matches": {
            "output": "matches-disk-lightglue",
            "model": {
                "features": "disk",
                "name": "lightglue",
                "weights": "disk_lightglue",
                "filter_threshold": 0.2,
                "width_confidence": -1,
                "depth_confidence": -1,
                "mp": True
            },
        },
    },
    "diskl": {
        "features": {
            "output": "feats-disk",
            "model": {
                "name": "disk",
                "max_keypoints": 5000,
            },
            "preprocessing": {
                "grayscale": False,
                "resize_max": 1600,
            },
        },
        "matches": {
            "output": "matches-disk-lightglue",
            "model": {
                "features": "disk",
                "name": "lightglue",
                "filter_threshold": 0.01,
                "width_confidence": -1,
                "depth_confidence": -1,
                "mp": True
            },
        },
    },
    "disk2k": {
        "features": {
            "output": "feats-disk2k",
            "model": {
                "name": "disk",
                "max_keypoints": 2048,
            },
            "preprocessing": {
                "grayscale": False,
                "resize_max": 1600,
            },
        },
        "matches": {
            "output": "matches-disk2k-lightglue",
            "model": {
                "features": "disk",
                "name": "lightglue",
                "filter_threshold": 0.1,
                "width_confidence": -1,
                "depth_confidence": -1,
                "mp": True
            },
        },
    },
    "aliked": {
        "features": {
            "output": "feats-alikedn16",
            "model": {
                "name": "aliked",
                "model_name": "aliked-n16",  # 'aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'
                "max_num_keypoints": 4096,
                "detection_threshold": 0.0,
                "force_num_keypoints": False,
            },
            "preprocessing": {
                "resize_max": 1600,
                # "resize_force": True,
            },
        },
        "matches": {
            "output": "matches-aliked-lightglue",
            "model": {
                "features": "aliked",
                "name": "lightglue",
                "filter_threshold": 0.1,
                "width_confidence": -1,
                "depth_confidence": -1,
                "mp": True
            },
        },
    },
    "aliked2k": {
        "features": {
            "output": "feats-aliked2k",
            "model": {
                "name": "aliked",
                "model_name": "aliked-n16",  # 'aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'
                "max_num_keypoints": 2048,
                "detection_threshold": 0.0,
                "force_num_keypoints": False,
            },
            "preprocessing": {
                "resize_max": 1600,
                # "resize_force": True,
            },
        },
        "matches": {
            "output": "matches-aliked2k-lightglue",
            "model": {
                "features": "aliked",
                "name": "lightglue",
                "filter_threshold": 0.1,
                "width_confidence": -1,
                "depth_confidence": -1,
                "mp": True
            },
        },
    },
    "aliked2kh": {
        "features": {
            "output": "feats-aliked2k",
            "model": {
                "name": "aliked",
                "model_name": "aliked-n16",  # 'aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'
                "max_num_keypoints": 2048,
                "detection_threshold": 0.0,
                "force_num_keypoints": False,
            },
            "preprocessing": {
                "resize_max": 1600,
                # "resize_force": True,
            },
        },
        "matches": {
            "output": "matches-aliked2k-lightglue",
            "model": {
                "features": "aliked",
                "name": "lightglue",
                "filter_threshold": 0.2,
                "width_confidence": -1,
                "depth_confidence": -1,
                "mp": True
            },
        },
    },
    "aliked2kl": {
        "features": {
            "output": "feats-aliked2k",
            "model": {
                "name": "aliked",
                "model_name": "aliked-n16",  # 'aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'
                "max_num_keypoints": 2048,
                "detection_threshold": 0.0,
                "force_num_keypoints": False,
            },
            "preprocessing": {
                "resize_max": 1600,
                # "resize_force": True,
            },
        },
        "matches": {
            "output": "matches-aliked2k-lightglue",
            "model": {
                "features": "aliked",
                "name": "lightglue",
                "filter_threshold": 0.01,
                "width_confidence": -1,
                "depth_confidence": -1,
                "mp": True
            }
        }
    }
}
