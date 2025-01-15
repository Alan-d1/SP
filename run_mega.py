import logging
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.datasets import get_dataset
from gluefactory.models.cache_loader import CacheLoader
from gluefactory.settings import DATA_PATH, EVAL_PATH
from gluefactory.utils.export_predictions import export_predictions
from gluefactory.visualization.viz2d import plot_cumulative
from gluefactory.eval.eval_pipeline import EvalPipeline
from gluefactory.eval.io import get_eval_parser, load_model, parse_eval_args
from gluefactory.eval.utils import eval_matches_epipolar, eval_poses, eval_relative_pose_robust


# 假设这个是从 gluefactory.eval.megadepth1500 中导入的 MegaDepth1500Pipeline
from gluefactory.eval.megadepth1500 import MegaDepth1500Pipeline

if __name__ == "__main__":

    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(MegaDepth1500Pipeline.default_conf)

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = MegaDepth1500Pipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    pprint(s)

    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
