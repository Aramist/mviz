# Usage:

```shell
# Activate virtual environment of choice, then:
pip install -e .
python -m mouseviz output_path assessment video_tracks vocalization_segments [--assignments assignments_path]
```

## Arguments:
  - `output_path`: Where the output video should be written. Extension is expected to be `.mp4` and the parent directory is expected to exist
  - `assessment`: Path to the output of vocalocator.assess. Should be an HDF5 file
  - `video_tracks`: Path to an HDF5 file containing tracks for all mice for the entire duration of the original recording session.
  - `vocalization_segments`: Path to a numpy file (shape `(n, 2)`) or a csv file (with columns `start` and `stop`) containing the start and end times (in seconds) of all sound events.
  - `assignments_path`: Optional path to a numpy file (shape `(n,)`, integer dtype) containing assignments for each sound event. A value of `-1` at index `idx` indicates that the sound event at index `idx` did not lie within the predicted confidence set. A value in the interval `[1, n_mice]` indicates an assignment to the respective mouse. A value of `n_mice + 1` indicates the presence of multiple mice in the confidence set. 