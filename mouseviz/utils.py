import json
from pathlib import Path
from typing import Optional

import cv2
import h5py
import matplotlib as mpl
import matplotlib.path as mplpath
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch
from shapely.geometry import Polygon


def softplus(x: np.ndarray):
    return np.log1p(np.exp(x))


def eval_1d_gaussian(points: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-((points - mu) ** 2) / (2 * sigma**2))


def draw_mouse_3d(
    mouse_joints: np.ndarray,
    ax: mpl.axes.Axes,
    color: str = "r",
    linewidth: int = 2,
    edges_only: bool = True,
) -> None:
    """Given the positions of all nodes in the mouse, draw a wireframe of the mouse in 3D.

    Args:
        mouse_joints (np.ndarray): The positions of all nodes in the mouse (15, 3)
        ax (mpl.axes.Axes): The axes to draw the wireframe on.
        color (str, optional): Color of rendered mouse. Defaults to "r".
        linewidth (int, optional): Stroke width of rendered mouse. Defaults to 2.
        edges_only (bool, optional): If False, will draw points at joints in addition to edges between joints. Defaults to True.

    Raises:
        ValueError: if mouse_joints does not have shape (15, 3)
    """

    # Check shape
    if mouse_joints.shape != (15, 3):
        raise ValueError(
            f"Expected mouse_joints to have shape (15, 3), got {mouse_joints.shape}"
        )

    # This is the order of the nodes in the track
    (
        nose,
        r_ear,
        l_ear,
        tail_0,
        tail_4,
        head,
        trunk,
        tail_1,
        tail_2,
        tail_3,
        l_shoulder,
        r_shoulder,
        l_hip,
        r_hip,
        neck,
    ) = mouse_joints

    if not edges_only:
        for n, p in enumerate(mouse_joints):
            if n == 0:
                ax.plot(*p, color=color, marker="o", markersize=2)
            else:
                ax.plot(
                    *p,
                    color=color,
                    marker="o",
                    markersize=2,
                    fillstyle="none",
                    markeredgecolor=color,
                )

    lines = [
        [nose, head, neck, trunk, tail_0, tail_1, tail_2, tail_3, tail_4],
        [nose, l_ear, head],
        [nose, r_ear, head],
        [neck, l_shoulder],
        [neck, r_shoulder],
        [trunk, l_hip],
        [trunk, r_hip],
    ]
    lines = [np.array(l) for l in lines]

    for line in lines:
        ax.plot(*line.T, color=color, linewidth=linewidth)


def create_arena_axes(dims: np.ndarray) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Given the dimensions of the arena, create a figure and axes for 3d plotting.

    Args:
        dims (np.ndarray): The dimensions of the arena in meters (3,)

    Returns:
        tuple[mpl.figure.Figure, mpl.axes.Axes]: The figure and axes for 3d plotting.
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=240)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_xlim([-dims[0] / 2, dims[0] / 2])
    ax.set_ylim([-dims[1] / 2, dims[1] / 2])
    ax.set_zlim(0, dims[2])

    ax.set_aspect("equal")
    ax.set_proj_type("persp")
    # draw the wireframe
    ax.plot(
        [-dims[0] / 2, dims[0] / 2, dims[0] / 2, -dims[0] / 2, -dims[0] / 2],
        [-dims[1] / 2, -dims[1] / 2, dims[1] / 2, dims[1] / 2, -dims[1] / 2],
        [0, 0, 0, 0, 0],
        color="k",
    )

    ax.plot(
        [-dims[0] / 2, dims[0] / 2, dims[0] / 2, -dims[0] / 2, -dims[0] / 2],
        [-dims[1] / 2, -dims[1] / 2, dims[1] / 2, dims[1] / 2, -dims[1] / 2],
        [dims[2], dims[2], dims[2], dims[2], dims[2]],
        color="k",
    )

    ax.plot(
        [dims[0] / 2, dims[0] / 2],
        [-dims[1] / 2, -dims[1] / 2],
        [0, dims[2]],
        color="k",
    )

    ax.plot(
        [dims[0] / 2, dims[0] / 2], [dims[1] / 2, dims[1] / 2], [0, dims[2]], color="k"
    )

    ax.plot(
        [-dims[0] / 2, -dims[0] / 2],
        [-dims[1] / 2, -dims[1] / 2],
        [0, dims[2]],
        color="k",
    )

    ax.plot(
        [-dims[0] / 2, -dims[0] / 2],
        [dims[1] / 2, dims[1] / 2],
        [0, dims[2]],
        color="k",
    )

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.grid(False)

    return fig, ax


def get_frame_buffer(figure: mpl.figure.Figure) -> np.ndarray:
    """Generates a BGRA image compatible opencv from a matplotlib figure

    Args:
        figure (mpl.figure.Figure): Figure to render

    Returns:
        np.ndarray: uchar image buffer of the figure in BGR format (height, width, 3)
    """

    figure.canvas.draw()
    buf = figure.canvas.buffer_rgba()
    # physical=True accounts for DPI scaling on high-resolution displays
    width, height = figure.canvas.get_width_height(physical=True)
    return np.frombuffer(buf, np.uint8).reshape(height, width, 4)[..., 2::-1]  # BGR


def make_xy_grid(arena_dims: np.ndarray, render_dims: np.ndarray) -> np.ndarray:
    """Generates a grid of points for evaluating a PMF. Places origin at the center of the arena. Striding
    along dimension 0 increases the y-coordinate, and striding along dimension 1 increases the x-coordinate.

    Args:
        arena_dims (np.ndarray | tuple[float, float]): Dimensions of the arena in desired units
        render_dims (np.ndarray | tuple[int,int]): Dimensions of the grid to generate (integer) (x_res, y_res)

    Returns:
        grid (np.ndarray): Grid of points for evaluating a PMF (render_dims[1], render_dims[0], 2)
    """
    test_points = np.stack(
        np.meshgrid(
            np.linspace(-arena_dims[0] / 2, arena_dims[0] / 2, render_dims[0]),
            np.linspace(-arena_dims[1] / 2, arena_dims[1] / 2, render_dims[1]),
        ),
        axis=-1,
    )
    return test_points


def get_confidence_set(pdf: np.ndarray, confidence_level: float) -> np.ndarray:
    """Get the confidence set for the given pdf. Makes no assumptions about
    the location of the origin in the arena.

    Args:
        pdf (np.ndarray): mass function of the distribution (y_res, x_res)
        confidence_level (float): confidence level at which the set will be computed

    Returns:
        np.ndarray: boolean array of the same shape as pdf containing the confidence set
    """

    orig_shape = pdf.shape
    flat_pdf = pdf.flatten()
    sorted_indices = np.argsort(flat_pdf)[::-1]  # big to small
    cumsum = np.cumsum(flat_pdf[sorted_indices])
    idx_in_confidence_set = sorted_indices[cumsum < confidence_level]
    confidence_set = np.zeros_like(flat_pdf, dtype=bool)
    confidence_set[idx_in_confidence_set] = True
    return confidence_set.reshape(orig_shape)


def eval_pdf(points: np.ndarray, mean_2d: np.ndarray, cov_2d: np.ndarray) -> np.ndarray:
    """Evaluate the multivariate normal pdf at points. Assumes the points and the mean are in the same coordinate system.
    In the case of a singular covariance matrix, the pdf will be a delta function at the mean.

    Args:
        points (np.ndarray): (n_points, 2)
        mean_2d (np.ndarray): (2,)
        cov_2d (np.ndarray): (2, 2)

    Returns:
        np.ndarray: The pdf evaluated at the given points
    """
    points_orig_shape = points.shape[:-1]
    points = points.reshape(-1, 2)
    diff = points - mean_2d
    try:
        precision = np.linalg.inv(cov_2d)
        exp_term = -0.5 * np.einsum("ij,jk,ik->i", diff, precision, diff)
        # Prevent overflow
        exp_term -= exp_term.max()
        probs = np.exp(exp_term)
        probs /= probs.sum()  # Normalize within arena bounds
        return probs.reshape(*points_orig_shape)
    except np.linalg.LinAlgError:
        # If the covariance matrix is singular, return a delta function at the mean
        closest_point_idx = np.argmin(np.linalg.norm(points - mean_2d, axis=-1))
        probs = np.zeros(points.shape[0])
        probs[closest_point_idx] = 1
        return probs.reshape(*points_orig_shape)


def get_confidence_set_contours(
    model_mean: np.ndarray, model_2d_cov: np.ndarray, arena_dims: np.ndarray
) -> np.ndarray:
    """Get the confidence set contours from the model's raw output. Assumes the model's output is
    in a coordinate frame which places the orgin at the center of the arena and in meters.
    The contours returned will be in the same coordinate frame.
    Assumes arena dims and cov are in mm

    Args:
        model_mean (np.ndarray): Mean of the model's output distribution (2,)
        model_2d_cov (np.ndarray): Covariance matrix of the model's output distribution (2, 2)
        arena_dims (np.ndarray): Dimensions of the arena in mm (3,)

    Returns:
        np.ndarray: _description_
    """
    """Get the pdf from the model's raw output. Assumes the model's output is
    in a coordinate frame which places the orgin at the center of the arena and in meters.
    The contours returned will be in the same coordinate frame.
    Assumes arena dims and cov are in mm
    """
    model_mean = model_mean  # origin at center of arena
    test_points = make_xy_grid(arena_dims[:2], (200, 200))  # origin at center of arena
    pdf = eval_pdf(test_points, model_mean, model_2d_cov)
    confidence_set = get_confidence_set(pdf, 0.95)  # Shape (200, 200), boolean
    confidence_set = confidence_set.astype(np.uint8) * 255
    contour = cv2.findContours(
        confidence_set, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0][0].squeeze()
    # Convert from pixels in the render back to mm in the arena
    contour = contour * (arena_dims[:2] / 200) - arena_dims[:2] / 2  # origin at center
    # Make it a closed curve
    contour = np.concatenate([contour, contour[:1]])
    return contour


def get_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame


def estimate_angle_pdf(
    pred_6d_mean: np.ndarray,
    pred_6d_cov: np.ndarray,
    n_samples: int = 1000,
    theta_bins: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimates a pmf of the yaw angle based on the 6d covariance matrix of the
    nose and head positions.

    Args:
        pred_6d_mean (np.ndarray): Head and nose mean positions (6,)
        pred_6d_cov (np.ndarray): 6d covariance matrix of the head and nose positions (6, 6)
        n_samples (int, optional): Number of samples to use in the histogram. Defaults to 1000.
        theta_bins (Optional[np.ndarray], optional): angle histogram bin edges. Defaults to 45 bins from -pi to pi.

    Returns:
        tuple[np.ndarray, np.ndarray]: The bin edges and the histogram of the angles
    """
    if theta_bins is None:
        theta_bins = np.linspace(-np.pi, np.pi, 45)

    gaussian_rv = np.random.multivariate_normal(
        mean=pred_6d_mean, cov=pred_6d_cov, size=n_samples
    )
    angles = np.arctan2(
        gaussian_rv[:, 1] - gaussian_rv[:, 4], gaussian_rv[:, 0] - gaussian_rv[:, 3]
    )
    angle_pdf, _ = np.histogram(angles, bins=theta_bins, density=False)
    angle_pdf = angle_pdf / angle_pdf.sum()
    return theta_bins, angle_pdf


def compute_covs(raw_outputs: np.ndarray, arena_dims: np.ndarray) -> np.ndarray:
    """Computes the covariance matrix from the raw output of the model.

    Args:
        raw_outputs (np.ndarray): Row output of the model (B, n_outputs)
        arena_dims (np.ndarray): Dimensions of the arena in mm

    Returns:
        np.ndarray: Covariance matrix (mm^2)
    """
    raw_outputs = raw_outputs
    n_dims = 6
    L = np.zeros((raw_outputs.shape[0], n_dims, n_dims))
    # embed the elements into the matrix
    idxs = np.tril_indices(n_dims)
    L[:, idxs[0], idxs[1]] = raw_outputs[:, n_dims:]
    # apply softplus to the diagonal entries to guarantee the resulting
    # matrix is positive definite
    new_diagonals = softplus(np.diagonal(L, axis1=-2, axis2=-1))  # (batch, 2)
    L[:, np.arange(n_dims), np.arange(n_dims)] = new_diagonals

    scale = arena_dims.max() / 2
    L = L * scale  # convert from arb. to mm
    # Compute covariance
    covs = np.einsum("bik,bjk->bij", L, L)
    return covs


def load_assess_file(assess_file: Path) -> dict[str, np.ndarray]:
    """Load the assess file and return the contents as a dictionary"""

    with h5py.File(assess_file, "r") as f:
        # model_config = json.loads(f.attrs["model_config"])
        # arena_dims_mm = np.array(model_config["DATA"]["ARENA_DIMS"])
        arena_dims_mm = np.array([615, 615, 425], dtype=float)

        # A recent change mmade the output shape (nnode, ndim) instead of (nnode*ndim) for each batch element
        dset_size, nnode, ndim = f["point_predictions"].shape
        pred_means_mm = f["point_predictions"][:].reshape(dset_size, nnode * ndim)
        raw_output = f["raw_model_output"][:]
        pred_cov_6d_mm = compute_covs(raw_output, arena_dims_mm)

    return {
        "arena_dims_mm": arena_dims_mm,
        "pred_means_mm": pred_means_mm,
        "pred_cov_6d_mm": pred_cov_6d_mm,
        "raw_output": raw_output,
    }


def load_segments_file(segments_file: Path) -> np.ndarray:
    """Loads vocalization segments from a csv or npy file.

    Args:
        segments_file (Path): Path to segments

    Returns:
        np.ndarray: An (N, 2) array containing the start and end times of each vocalization, in seconds
    """

    if segments_file.suffix == ".npy":
        return np.load(segments_file)

    df = pd.read_csv(segments_file)
    return df[["start", "stop"]].values


def plot_contour_with_angular_pdf(
    ax: mpl.axes.Axes,
    center: np.ndarray,
    radius: np.ndarray,
    theta: np.ndarray,
    pdf: np.ndarray,
    dist_height: float,
    color="blue",
) -> None:
    """Plots the contour of the x-y marginal distribution and the angular pdf as a ring around the contour.

    Args:
        ax (mpl.axes.Axes): Axes to plot on
        center (np.ndarray): Mean of the distribution
        radius (np.ndarray): Radius of the polar representation of the contour
        theta (np.ndarray): Angles of the polar representation of the contour
        pdf (np.ndarray): Angular pdf. Bin edges given by theta
        dist_height (float): Distance from the distribution peak and the contour
        color (str, optional): Color of the contour. Defaults to "blue".
    """

    def ring_coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(ob.coords)
        codes = np.ones(n, dtype=mplpath.Path.code_type) * mplpath.Path.LINETO
        codes[0] = mplpath.Path.MOVETO
        return codes

    def pathify(polygon):
        # Convert coordinates to path vertices. Objects produced by Shapely's
        # analytic methods have the proper coordinate order, no need to sort.
        vertices = np.concatenate(
            [np.asarray(polygon.exterior.xy).T]
            + [np.asarray(r.xy).T for r in polygon.interiors]
        )
        codes = np.concatenate(
            [ring_coding(polygon.exterior)]
            + [ring_coding(r) for r in polygon.interiors]
        )
        return mplpath.Path(vertices, codes)

    # plot the distribution peak
    peak_direction = np.array(
        [np.cos(theta[pdf.argmax()]), np.sin(theta[pdf.argmax()])]
    )
    # peak_line_start = center + peak_direction * radius[pdf.argmax()]
    peak_line_start = center
    peak_line_end = center + peak_direction * (radius[pdf.argmax()])
    ax.plot(
        [peak_line_start[0], peak_line_end[0]],
        [peak_line_start[1], peak_line_end[1]],
        color=color[:3],
        lw=1,
        linestyle="--",
    )

    inner_geom_points = (
        np.stack([np.cos(theta), np.sin(theta)], axis=1) * radius[:, None]
        + center[None, :]
    )
    inner_geom = Polygon(inner_geom_points)

    pdf_rescaled = pdf / pdf.max() * dist_height + radius + 0.005
    unit_vecs = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    outer_ring_points = unit_vecs * pdf_rescaled[:, None] + center
    outer_ring_geom = Polygon(outer_ring_points)

    difference = pathify(outer_ring_geom.difference(inner_geom))
    patch = PathPatch(difference, facecolor=color, edgecolor=None, lw=0)
    ax.add_patch(patch)
    art3d.pathpatch_2d_to_3d(patch, z=0, zdir="z")


cached_timeline = None


def make_timeline(
    t: float,
    vocalization_times_sec: np.ndarray,
    video_duration: float,
    video_dims: tuple[int, int],
    assignments: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Makes a timeline for the current frame of the video. The timeline will show a raster-plot of
    the vocalizations, optionally colored by the provided assignments.

    Args:
        t (float): Current time in the video
        vocalization_times_sec (np.ndarray): Times of all vocalizations in the video (s)
        video_length (float): Duration of the video (s)
        video_dims (tuple[int, int]): Dimensions of the video (width, height)
        assignments (Optional[np.ndarray], optional): Identity of the mouse that emitted each vocalization, if known.
          Values of -1 indicate unassigned vocalizations. 0 or 1 signify the mouse that emitted the vocalization. 2
          signifies an ambiguous identity. Defaults to None.

    Returns:
        np.ndarray: An image (BGR, uchar) of the timeline (timeline_height, video_width, 3)
    """
    global cached_timeline
    image_width, image_height = video_dims
    timeline_width = int(image_width * 0.8)
    timeline_start_x = int(image_width * 0.1)
    timeline_end_x = timeline_start_x + timeline_width
    timeline_height = int(image_height * 0.05)
    margin_y = int(image_height * 0.05) // 2
    stroke_width = 1

    if cached_timeline is not None:
        timeline_image = cached_timeline.copy()
    else:
        vocalization_raster_locations = np.interp(
            vocalization_times_sec,
            [0, video_duration],
            [timeline_start_x, timeline_end_x],
        ).astype(int)

        timeline_image = np.full(
            (timeline_height + 2 * margin_y, image_width, 3), 255, dtype=np.uint8
        )
        # draw border
        timeline_image[
            margin_y : margin_y + stroke_width, timeline_start_x:timeline_end_x, :
        ] = 0
        timeline_image[
            -stroke_width - margin_y : -margin_y, timeline_start_x:timeline_end_x, :
        ] = 0
        timeline_image[
            margin_y:-margin_y, timeline_start_x : timeline_start_x + stroke_width, :
        ] = 0
        timeline_image[
            margin_y:-margin_y, timeline_end_x - stroke_width : timeline_end_x, :
        ] = 0

        if assignments is None:
            assignments = np.full_like(vocalization_raster_locations, -1)
        # draw vocalization times
        unassigned_color = (127, 127, 127)  # gray (bgr)
        mouse_1_color = (0, 255, 0)  # green (bgr)
        mouse_2_color = (0, 0, 255)  # red (bgr)
        both_color = (127, 127, 127)  # gray (bgr)

        all_colors = [unassigned_color, mouse_1_color, mouse_2_color, both_color]
        bar_height = timeline_height - 2 * stroke_width
        bar_start = margin_y + stroke_width
        # Each assignment category occupies 1/3 of the bar's vertical space
        color_regions = [
            (bar_start, bar_start + bar_height / 3),
            (bar_start + bar_height / 3, bar_start + 2 * bar_height / 3),
            (bar_start + 2 * bar_height / 3, bar_start + bar_height),
            (bar_start, bar_start + bar_height / 3),
        ]
        for n, loc in enumerate(vocalization_raster_locations):
            color = all_colors[assignments[n] + 1]
            region = color_regions[assignments[n] + 1]
            timeline_image[int(region[0]) : int(region[1]), loc] = color

        cached_timeline = timeline_image.copy()

    # draw a line at the bottom of the box to show the current time
    current_time_loc = (
        np.interp(
            t,
            [0, video_duration],
            [timeline_start_x, timeline_end_x],
        )
        .astype(int)
        .item()
    )

    timeline_image[
        margin_y + stroke_width : -stroke_width - margin_y + 25,
        current_time_loc : current_time_loc + stroke_width,
    ] = (0, 0, 0)

    return timeline_image
