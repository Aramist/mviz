from pathlib import Path

import numpy as np

from . import utils
from .viz import make_full_vid

if __name__ == "__main__":
    import argparse

    import h5py

    ap = argparse.ArgumentParser()
    ap.add_argument("output_path", type=Path)
    ap.add_argument("assessment", type=Path)
    ap.add_argument("video_tracks", type=Path)
    ap.add_argument("vocalization_segments", type=Path)
    ap.add_argument("--assignments", type=Path, required=False)
    args = ap.parse_args()

    assessment_data = utils.load_assess_file(args.assessment)
    with h5py.File(args.video_tracks, "r") as ctx:
        video_tracks = ctx["tracks"][:]
    vocalization_segments = utils.load_segments_file(args.vocalization_segments)
    if args.assignments is not None:
        vocalization_assignments = np.load(args.assignments)

    make_full_vid(
        args.output_path,
        assessment_data,
        video_tracks,
        vocalization_segments,
        vocalization_assignments,
        do_parallel=True,
    )
