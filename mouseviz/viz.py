from pathlib import Path
from typing import Optional

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from . import utils

# mpl.use("Agg")

RED_COLOR = (1.0, 0.0, 0.0, 1.0)
RED_SHADOW_COLOR = (0.5, 0, 0, 0.5)
GREEN_COLOR = (0.0, 1.0, 0.0, 1.0)
GREEN_SHADOW_COLOR = (0, 0.5, 0, 0.5)
BLUE_COLOR = (0.0, 0.0, 1.0, 1.0)
BLUE_SHADOW_COLOR = (0, 0, 0.5, 0.5)
GRAY_SHADOW_COLOR = (0.5, 0.5, 0.5, 1)


mouse_colors = [
    GREEN_COLOR,
    RED_COLOR,
    BLUE_COLOR,
]  # todo: Add more colors for more mice
shadow_colors = [
    GREEN_SHADOW_COLOR,
    RED_SHADOW_COLOR,
    BLUE_SHADOW_COLOR,
]  # todo: Add more colors for more mice


def failed_frame(width: int, height: int) -> np.ndarray:
    """A placeholder frame to draw if the visualization raises for whatever reason.

    Args:
        width (int): Video frame width
        height (int): Video frame height

    Returns:
        np.ndarray: A uchar BGR image (height, width, 3)
    """

    im = np.full((height, width, 3), 127, dtype=np.uint8)
    msg = "Failed to render frame :("
    (text_width, text_height), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

    cv2.putText(
        im,
        msg,
        (width // 2 - text_width // 2, height // 2 + text_height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def visualize_single_frame(
    mice_joints: np.ndarray,
    frame_assignment: Optional[int],
    arena_dims_mm: np.ndarray,
    pred_mean_mm: Optional[np.ndarray],
    pred_cov_mm2: Optional[np.ndarray],
) -> tuple[mpl.pyplot.Figure, mpl.axes.Axes]:
    """Visualizes a single frame of the video with the mice and vocalocator predictions.

    Args:
        mice_joints (np.ndarray): The 3D coordinates of the mice skeletons in mm. Shape (n_mice, n_nodes, 3)
        frame_assignment (Optional[int]): The assignment of the vocalization to a mouse. None if no vocalization is present.
        arena_dims_mm (np.ndarray): The dimensions of the arena in mm. Shape (3,)
        pred_mean_mm (Optional[np.ndarray]): The mean of the 6D prediction in mm. Shape (6,)
        pred_cov_mm2 (Optional[np.ndarray]): The covariance of the 6D prediction in mm^2. Shape (6, 6)

    Returns:
        tuple[mpl.pyplot.Figure, mpl.axes.Axes]: The figure and axes objects containing the arena visualization
    """
    fig, ax = utils.create_arena_axes(arena_dims_mm / 1000)

    legend_position = (0.075, 0.9)  # just trust me bro
    num_mice = mice_joints.shape[0]
    h_width, h_length, _ = arena_dims_mm / 1000 / 2

    # Add invisible points to the plot to create legend labels
    ax.scatter(1e4, 1e4, 1e4, color="b", label="Prediction", alpha=1)
    ax.plot([1e4, 1e4], [1e4, 1e4], [1e4, 1e4], color="b", label="Predicted Elevation")
    ax.plot(
        [1e4, 1e4],
        [1e4, 1e4],
        [1e4, 1e4],
        color="b",
        linestyle="--",
        label="Predicted Orientation",
    )
    ax.plot(
        [1e4, 1e4],
        [1e4, 1e4],
        [1e4, 1e4],
        color=(0, 0, 1, 0.3),  # low alpha blue
        label="95% Confidence Set",
    )

    for mouse_idx in range(num_mice):
        mouse_frame = mice_joints[mouse_idx]
        utils.draw_mouse_3d(mouse_frame, ax, color=mouse_colors[mouse_idx])

        # Add invisible points to the plot to create legend labels
        ax.scatter(
            1e4,
            1e4,
            1e4,
            color=mouse_colors[mouse_idx],
            label=f"Mouse {mouse_idx + 1}",
            alpha=1,
        )
        ax.plot(
            [1e4, 1e4],
            [1e4, 1e4],
            [1e4, 1e4],
            color=mouse_colors[mouse_idx],
            label=f"Mouse {mouse_idx + 1} Elevation",
        )
        # plot a shadow of the mouse locations
        ax.scatter(*mouse_frame[0, :2], 0, color=shadow_colors[mouse_idx])
        # Plot the elevation of each mouse along an edge of the wireframe
        ax.plot(
            [h_width, h_width - 0.03],
            [-h_length, -h_length],
            [mouse_frame[0, 2], mouse_frame[0, 2]],
            color=mouse_colors[mouse_idx],
        )
        ax.plot(
            [h_width, h_width],
            [-h_length, -h_length + 0.03],
            [mouse_frame[0, 2], mouse_frame[0, 2]],
            color=mouse_colors[mouse_idx],
        )

    if pred_mean_mm is None:
        ax.legend(bbox_to_anchor=legend_position, loc="upper right", fontsize="small")
        return fig, ax

    # Code related to plotting the prediction ########################################
    # The smallest input value for `frame_assignment` is -1, add 1 so we can use it as a list index
    frame_assignment = frame_assignment + 1
    contour_color = (
        GRAY_SHADOW_COLOR,
        *shadow_colors[:num_mice],
        GRAY_SHADOW_COLOR,
    )[frame_assignment]

    # Marginalize over the x and y dimensions
    contour_mm = utils.get_confidence_set_contours(
        pred_mean_mm[:2],
        pred_cov_mm2[:2, :2],
        arena_dims=arena_dims_mm[:2],
    )[:-1]

    centered_contour_m = (contour_mm - pred_mean_mm[:2]) / 1000
    contour_theta = np.arctan2(centered_contour_m[:, 1], centered_contour_m[:, 0])
    contour_r_m = np.linalg.norm(centered_contour_m, axis=1)
    # Sort the contour points by angle
    sorting = np.argsort(contour_theta)
    contour_theta = contour_theta[sorting]
    contour_r_m = contour_r_m[sorting]

    # resample the contour to have a more uniform distribution of points
    new_contour_theta_bins = np.linspace(-np.pi, np.pi, 46, endpoint=True)
    new_contour_theta_midpoints = (
        new_contour_theta_bins[1:] + new_contour_theta_bins[:-1]
    ) / 2

    # Wrap theta and r a bit to ensure smooth interpolation at -pi and pi
    contour_theta_wrap = np.insert(contour_theta, 0, contour_theta[-1] - 2 * np.pi)
    contour_theta_wrap = np.insert(contour_theta_wrap, -1, contour_theta[0] + 2 * np.pi)
    contour_r_wrap_m = np.insert(contour_r_m, 0, contour_r_m[-1])
    contour_r_wrap_m = np.insert(contour_r_wrap_m, -1, contour_r_m[0])

    new_contour_radii_m = np.interp(
        new_contour_theta_midpoints, contour_theta_wrap, contour_r_wrap_m
    )

    _, angle_pdf = utils.estimate_angle_pdf(
        pred_6d_mean=pred_mean_mm,
        pred_6d_cov=pred_cov_mm2,
        n_samples=5000,
        theta_bins=new_contour_theta_bins,
    )

    utils.plot_contour_with_angular_pdf(
        ax=ax,
        center=pred_mean_mm[:2] / 1000,
        radius=new_contour_radii_m,
        theta=new_contour_theta_midpoints,
        pdf=angle_pdf,
        dist_height=0.02,  # This height looked good in practice
        color=contour_color,
    )
    ax.legend(bbox_to_anchor=legend_position, loc="upper right", fontsize="small")
    return fig, ax


def parallel_wrapper(width, height, *args, **kwargs):
    """Wrapper for visualize_single_frame to allow parallelization with joblib. Ensures figures never cross
    processes."""
    fig, ax = visualize_single_frame(*args, **kwargs)
    if fig is None:
        return failed_frame(width, height)
    else:
        im = utils.get_frame_buffer(fig)
        plt.close(fig)
        return im


def make_full_vid(
    output_path: Path,
    assessment_data: dict,
    video_tracks: np.ndarray,
    vocalization_segments: np.ndarray,
    vocalization_assignments: Optional[np.ndarray],
    *,
    do_parallel: bool = False,
    video_sr: int = 150,
) -> None:
    """Generates a video showing all mice and 6D vocalocator predictions.

    Args:
        output_path (Path): Path to save the video. Should have .mp4 extension.
        assessment_data (dict): The dictionary output by utils.load_assess_file(Path)
        video_tracks (np.ndarray): The full video tracks in mm. Origin expected at center of arena. Shape (n_frames, n_mice, n_nodes, 3). Expects 15 nodes
        vocalization_segments (np.ndarray): The start and end times of each vocalization in seconds. Shape (n_vocalizations, 2)
        vocalization_assignments (Optional[np.ndarray]): The assignment of each vocalization to a mouse. Shape (n_vocalizations,), dtype=int
        do_parallel (bool, optional): Use joblib to parallelize video generation across available cpu cores. Defaults to False.
        video_sr (int, optional): Rate of video tracks. Defaults to 150.
    """

    # Check validity of inputs
    if len(video_tracks.shape) == 3:
        video_tracks = video_tracks[
            :, None, ...
        ]  # Add a singleton dimension for the mice identity axis
    if video_tracks.shape[-2] != 15:
        raise ValueError(
            f"Expected 15 nodes in the mouse skeleton, found {video_tracks.shape[-2]}"
        )
    if video_tracks.shape[-1] != 3:
        raise ValueError(
            f"Expected 3D coordinates for each node, found {video_tracks.shape[-1]}D"
        )
    if vocalization_segments.shape[1] != 2:
        raise ValueError(
            f"Expected 2 columns in vocalization_segments, found {vocalization_segments.shape[1]}"
        )
    if (
        vocalization_assignments is not None
        and vocalization_assignments.shape[0] != vocalization_segments.shape[0]
    ):
        raise ValueError(
            "vocalization_assignments must have the same number of rows as vocalization_segments"
        )

    video_duration = video_tracks.shape[0] / video_sr
    # Indices within the full video for each frame we will generate
    video_frame_idx = (np.arange(0, video_duration, 1 / 30) * video_sr).astype(int)
    video_frame_idx = np.clip(
        video_frame_idx, 0, video_tracks.shape[0] - 1
    )  # Ensure we don't go out of bounds
    vox_segments_frames = (vocalization_segments[:, 0] * video_sr).astype(int)
    mice_joints = video_tracks[
        video_frame_idx, ...
    ]  # (n_frames_in_viz, n_mice, n_nodes, 3)

    # For each frame, contains the index of the vocalization that will be displayed (if any)
    vox_idx_per_frame = []
    for frame_idx in video_frame_idx:
        # Number of vocalizations emitted from t=0 to now
        num_vox_called = (vox_segments_frames <= frame_idx).sum()
        # Index of most recent vocalization in vox_segments
        vox_idx = num_vox_called - 1
        if (
            frame_idx - vox_segments_frames[vox_idx] < 35
        ):  # this controls how long the vocalization is displayed
            vox_idx_per_frame.append(vox_idx)
        else:
            vox_idx_per_frame.append(None)

    arena_dims_mm = assessment_data["arena_dims_mm"]
    pred_means_mm = [
        assessment_data["pred_means_mm"][vox_idx] if vox_idx is not None else None
        for vox_idx in vox_idx_per_frame
    ]
    pred_covs_mm2 = [
        assessment_data["pred_cov_6d_mm"][vox_idx] if vox_idx is not None else None
        for vox_idx in vox_idx_per_frame
    ]
    assignments_per_frame = [
        vocalization_assignments[vox_idx] if vox_idx is not None else None
        for vox_idx in vox_idx_per_frame
    ]

    fig, ax = visualize_single_frame(
        mice_joints[0],
        assignments_per_frame[0],
        arena_dims_mm,
        pred_means_mm[0],
        pred_covs_mm2[0],
    )

    if fig is None:
        raise ValueError("Failed to render the first frame. Double check the inputs.")
    video_height, video_width, _ = utils.get_frame_buffer(fig).shape
    plt.close(fig)
    del fig, ax

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,  # skipping 4/5 frames at 150 fps
        (video_width, video_height),
        isColor=True,
    )

    if do_parallel:
        fig_generator = Parallel(n_jobs=-2, return_as="generator")(
            [
                delayed(parallel_wrapper)(
                    video_width,
                    video_height,
                    mj,
                    assn,
                    arena_dims_mm,
                    pm,
                    pcov,
                )
                for mj, assn, pm, pcov in zip(
                    mice_joints, assignments_per_frame, pred_means_mm, pred_covs_mm2
                )
            ]
        )

        for n, frame in tqdm(enumerate(fig_generator), total=len(video_frame_idx)):
            timeline = utils.make_timeline(
                n / 30.0,
                vocalization_times_sec=vocalization_segments,
                video_duration=video_duration,
                video_dims=(video_width, video_height),
                assignments=vocalization_assignments,
            )
            frame[: timeline.shape[0]] = timeline
            writer.write(frame)
    else:
        for n, (mj, assn, pm, pcov) in tqdm(
            enumerate(
                zip(mice_joints, assignments_per_frame, pred_means_mm, pred_covs_mm2)
            ),
            total=len(video_frame_idx),
        ):
            fig, ax = visualize_single_frame(
                mj,
                assn,
                arena_dims_mm,
                pm,
                pcov,
            )
            if fig is None:
                frame = failed_frame(video_width, video_height)
            else:
                frame = utils.get_frame_buffer(fig)
                plt.close(fig)

            timeline = utils.make_timeline(
                n / 30.0,
                vocalization_times_sec=vocalization_segments,
                video_duration=video_duration,
                image_width=video_width,
                image_height=video_height,
                assignments=vocalization_assignments,
            )
            frame[: timeline.shape[0]] = timeline
            writer.write(frame)
    writer.release()


def demo(
    assessment_data: dict,
    video_tracks: np.ndarray,
    vocalization_segments: np.ndarray,
    vocalization_assignments: Optional[np.ndarray],
    *,
    video_sr: int = 150,
) -> None:
    """Generates a video showing all mice and 6D vocalocator predictions.

    Args:
        output_path (Path): Path to save the video. Should have .mp4 extension.
        assessment_data (dict): The dictionary output by utils.load_assess_file(Path)
        video_tracks (np.ndarray): The full video tracks in mm. Origin expected at center of arena. Shape (n_frames, n_mice, n_nodes, 3). Expects 15 nodes
        vocalization_segments (np.ndarray): The start and end times of each vocalization in seconds. Shape (n_vocalizations, 2)
        vocalization_assignments (Optional[np.ndarray]): The assignment of each vocalization to a mouse. Shape (n_vocalizations,), dtype=int
        do_parallel (bool, optional): Use joblib to parallelize video generation across available cpu cores. Defaults to False.
        video_sr (int, optional): Rate of video tracks. Defaults to 150.
    """

    # Check validity of inputs
    if len(video_tracks.shape) == 3:
        video_tracks = video_tracks[
            :, None, ...
        ]  # Add a singleton dimension for the mice identity axis
    if video_tracks.shape[-2] != 15:
        raise ValueError(
            f"Expected 15 nodes in the mouse skeleton, found {video_tracks.shape[-2]}"
        )
    if video_tracks.shape[-1] != 3:
        raise ValueError(
            f"Expected 3D coordinates for each node, found {video_tracks.shape[-1]}D"
        )
    if vocalization_segments.shape[1] != 2:
        raise ValueError(
            f"Expected 2 columns in vocalization_segments, found {vocalization_segments.shape[1]}"
        )
    if (
        vocalization_assignments is not None
        and vocalization_assignments.shape[0] != vocalization_segments.shape[0]
    ):
        raise ValueError(
            "vocalization_assignments must have the same number of rows as vocalization_segments"
        )

    video_duration = video_tracks.shape[0] / video_sr
    # Indices within the full video for each frame we will generate
    video_frame_idx = (np.arange(0, video_duration, 1 / 30) * video_sr).astype(int)
    video_frame_idx = np.clip(
        video_frame_idx, 0, video_tracks.shape[0] - 1
    )  # Ensure we don't go out of bounds
    vox_segments_frames = (vocalization_segments[:, 0] * video_sr).astype(int)
    mice_joints = video_tracks[
        video_frame_idx, ...
    ]  # (n_frames_in_viz, n_mice, n_nodes, 3)

    # For each frame, contains the index of the vocalization that will be displayed (if any)
    vox_idx_per_frame = []
    for frame_idx in video_frame_idx:
        # Number of vocalizations emitted from t=0 to now
        num_vox_called = (vox_segments_frames <= frame_idx).sum()
        # Index of most recent vocalization in vox_segments
        vox_idx = num_vox_called - 1
        if (
            frame_idx - vox_segments_frames[vox_idx] < 35 and vox_idx > 0
        ):  # this controls how long the vocalization is displayed
            vox_idx_per_frame.append(vox_idx)
        else:
            vox_idx_per_frame.append(None)

    arena_dims_mm = assessment_data["arena_dims_mm"]
    pred_means_mm = [
        assessment_data["pred_means_mm"][vox_idx] if vox_idx is not None else None
        for vox_idx in vox_idx_per_frame
    ]
    pred_covs_mm2 = [
        assessment_data["pred_cov_6d_mm"][vox_idx] if vox_idx is not None else None
        for vox_idx in vox_idx_per_frame
    ]
    assignments_per_frame = [
        vocalization_assignments[vox_idx] if vox_idx is not None else None
        for vox_idx in vox_idx_per_frame
    ]

    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    idx_of_vox = 0
    while True:
        video_frame_idx = int(vocalization_segments[idx_of_vox].mean() * 30)
        print(
            f"frame_idx: {video_frame_idx}, vox_idx: {vox_idx_per_frame[video_frame_idx]}"
        )

        test_fig, _ = visualize_single_frame(
            mice_joints[video_frame_idx],
            assignments_per_frame[video_frame_idx],
            arena_dims_mm,
            pred_means_mm[video_frame_idx],
            pred_covs_mm2[video_frame_idx],
        )
        if test_fig is None:
            idx_of_vox += 1
            continue
        frame = utils.get_frame_buffer(test_fig)
        plt.close(test_fig)

        timeline = utils.make_timeline(
            video_frame_idx / 30,
            vocalization_times_sec=vocalization_segments,
            video_duration=video_duration,
            video_dims=(frame.shape[1], frame.shape[0]),
            assignments=vocalization_assignments,
        )
        frame[: timeline.shape[0]] = timeline

        cv2.imshow("test", frame)
        key = cv2.waitKey(0)

        if key == ord("l"):
            idx_of_vox += 1
        elif key == ord("h"):
            idx_of_vox -= 1
        elif key == ord("s"):
            cv2.imwrite(f"frame_{idx_of_vox}.png", frame)
        elif key == ord("q"):
            break

        idx_of_vox = np.clip(idx_of_vox, 0, len(vox_idx_per_frame) - 1)
        print(idx_of_vox)


if __name__ == "__main__":
    import argparse

    import h5py

    ap = argparse.ArgumentParser()
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

    demo(
        assessment_data,
        video_tracks,
        vocalization_segments,
        vocalization_assignments,
    )

    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
