# Three-tag plane pose

`Aruco_Test.py` estimates a plane's position and orientation relative to the
camera that took a still image. All lengths are in **millimeters**.

## Install and configure

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r ArucoTranslations/requirements.txt
```

Edit these constants at the top of `Aruco_Test.py`:

```python
tag_size = 50.0  # mm; replace 50.0 with your measured tag size
TAG_IDS = (0, 1, 2)
DICTIONARY = "DICT_4X4_50"
```

`tag_size` is the side of the **outer black square**, including its black border,
excluding any surrounding white margin. All three tags must have this size.
The size can also be overridden with `--tag-size 35` without editing the file.
The dictionary must match the one used to generate your tags. Use three unique
IDs and leave white space around each black square so detection can find it.

Attach all three tags flat on the **same plane you want to measure**, with their
centers forming a broad triangle. Avoid a line or a skinny triangle. If the tags
sit above the surface, the estimated plane is the tags' surface, so that offset
must be accounted for separately. All three tags and their corners must be
visible and sharp in the image. They can have different in-plane rotations.

## Calibrate the camera first

Metric pose needs the camera matrix (focal lengths and principal point) and
lens distortion coefficients. Tag size alone does not supply those parameters.
This script includes standard OpenCV **pinhole** calibration; very wide-angle
fisheye cameras need a separate fisheye calibration/undistortion workflow.

1. Print a chessboard and fix it to a flat, rigid surface. A board with **10 by
   7 squares** has **9 by 6 inner corners**. Measure one square's side in mm;
   the command below uses 25 mm as an example.
2. Take about **20-30 sharp photos** of the board with the camera you will use.
   Vary its tilt, distance and position so corners cover the center and edges
   of the image. Keep the entire grid of inner corners visible. Merely taking
   many identical front-facing views gives a weak calibration.
3. Keep the lens, focus, zoom, crop/aspect ratio and image-processing settings
   consistent with your target photo. Different pixel resolutions are resized
   automatically. Lock focus if possible. Recalibrate if the lens/settings
   change. For phone images, avoid switching lenses or using portrait
   effects, digital zoom or changing stabilization/cropping. Calibrate with the
   same photo pipeline used to produce the target image.
4. Put your photos in `ArucoTranslations/calibration/` and run:

```powershell
.\.venv\Scripts\python.exe ArucoTranslations/Aruco_Test.py calibrate --images "ArucoTranslations/calibration/*.jpg" --cols 9 --rows 6 --square-size 25 --output ArucoTranslations/camera_calibration.npz
```

The command needs at least 10 successfully detected boards. It saves
`camera_matrix`, `dist_coeffs`, `image_size` (width, height), and error statistics
in the `.npz` file. An existing standard OpenCV calibration can be used if you
save those first three arrays under the same names using `numpy.savez`.

Before detection, the script rotates portrait photos **90 degrees clockwise**
to landscape, then resizes all usable photos to one resolution. The first
usable photo defines that resolution, with its longest side capped at
**1600 pixels**; use `--max-dimension 1200` for a smaller cap. Smaller inputs
may be enlarged to match the shared resolution. Original files stay unchanged.
The calibration stores the processing resolution and method, and `estimate`
automatically applies them to your plane photo. Annotated images and camera
coordinates refer to this processed landscape image.

Images with a different aspect ratio (for example, 16:9 mixed with 4:3) are
skipped with an explanation. Resizing cannot undo a different camera crop,
lens, or zoom. Portrait rotation assumes a consistent phone/export convention;
if images were rotated in different directions or mirrored, first normalize
them into the same camera orientation. Prefer one phone orientation when
capturing future images. For older calibration files without processing
metadata, estimation still expects the original resolution and orientation.

The printed RMS errors measure reprojection error in pixels. Aim for well below
1 pixel, inspect unusually high-error views, and capture better images as
needed. Low reprojection error alone does not guarantee accurate distance:
validate against a ruler at several distances and tilts. Do not independently
undistort the target image before passing it to this script; it already passes
the original distortion model into the pose solver.

## Estimate from your image

```powershell
.\.venv\Scripts\python.exe ArucoTranslations/Aruco_Test.py estimate ArucoTranslations/plane.jpg --calibration ArucoTranslations/camera_calibration.npz --output ArucoTranslations/pose.json --annotated ArucoTranslations/annotated.jpg --show
```

Omit `--show` when no GUI is available. Paths are relative to your working
directory. To use other tag IDs or a dictionary, add, for example,
`--ids 10 20 30 --dictionary DICT_6X6_250`.

Without a measured layout, each tag's four corners and known size give a
separate pose estimate. The script fits a plane through the three estimated
3D tag centers. The plane origin is their centroid, +X runs from the first ID
toward the second, +Z is the plane normal facing toward the camera, and
`+Y = +Z cross +X`. The centroid is not necessarily your object's center.
With exactly three centers the plane fit has no redundant center observations;
depth noise can noticeably tilt the plane, especially for small/distant tags.

## More accurate option: measure the tag layout

If you can measure where the tags sit on the plane, supply `--layout` to fit
**all twelve corners in one pose solve**, instead of fitting three independent
centers. This usually gives a more stable pose and lets you place the origin
at a meaningful location, such as the center of your object.

Create JSON mapping each ID to its four **[x, y, z] corner positions in mm**
in one shared plane frame. All Z coordinates must be zero. Corner order must
be canonical **top-left, top-right, bottom-right, bottom-left as seen on the
unrotated marker design**, even if that tag is rotated on your plane. The
positions describe each tag's actual physical orientation, not its appearance
in the camera image. Choose orthogonal +X, +Y axes on the plane; +Z follows
the right-hand rule and may face toward or away from the camera.

For example, these are three 50 mm tags aligned with +X right, +Y up, centered
at (-100, 50), (100, 50), and (0, -100). These numbers are examples; replace them
with your measured positions. If `tag_size` changes, update the layout too.

```json
{
  "0": [[-125, 75, 0], [-75, 75, 0], [-75, 25, 0], [-125, 25, 0]],
  "1": [[75, 75, 0], [125, 75, 0], [125, 25, 0], [75, 25, 0]],
  "2": [[-25, -75, 0], [25, -75, 0], [25, -125, 0], [-25, -125, 0]]
}
```

Save your coordinates as `ArucoTranslations/layout.json`, then add
`--layout ArucoTranslations/layout.json` to the estimate command. The plane
origin and axes now follow your supplied coordinates. Orientations and origin
distances from the two modes need not match because they define different frames.

## Reading the output

The OpenCV camera frame has **+X right, +Y down, +Z forward**, with its origin at
the optical center. The output includes:

| Field | Meaning |
| --- | --- |
| `perpendicular_distance_mm` | Shortest distance from camera center to the infinite plane: `abs(normal dot translation)` |
| `distance_to_plane_origin_mm` | Straight-line distance to the chosen plane origin: `norm(translation)` |
| `plane_origin_depth_mm` | Origin's camera Z coordinate (optical-axis depth) |
| `plane_origin_camera_mm` | Plane origin's camera [X, Y, Z] coordinates |
| `plane_normal_camera` | Unit normal expressed in camera axes |
| `rotation_matrix_plane_to_camera` | Plane orientation; columns are plane X, Y, Z axes in camera coordinates |
| `rotation_vector_rad` | Rodrigues axis-angle vector, **not Euler angles** |
| `transform_plane_to_camera_mm` | 4x4 transform: `point_camera = R @ point_plane + t` |
| `camera_position_plane_mm` | Camera optical center expressed in plane coordinates: `-R.T @ t` |
| `plane_equation_camera` | Plane equation `normal dot point_camera + d_mm = 0` |
| `reprojection_rms_px` | Per-tag errors, or a joint error when using a measured layout |

The perpendicular foot may lie outside the physical object; this distance is
to the infinite plane, not necessarily to the nearest point on a finite panel.
Annotated axes are red X, green Y, blue Z, with axis length equal to `tag_size`.

A single image of a planar target can give two similar pose solutions,
especially with small or nearly front-facing tags. The script compares both
positive-depth planar solutions and warns if distinct orientations have nearly
equal errors. It cannot guarantee the physically correct choice from one image.
Use larger tags, a measured layout, a clearer oblique view, or additional views
when orientation is unstable. Printed size error also directly scales distance
error: a 2% size error produces roughly a 2% distance error.

## Verify the implementation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s ArucoTranslations -p "test_*.py" -v
```

The tests use known synthetic geometry and rendered tags. Real-camera accuracy
still needs validation with physical measurements.

OpenCV references: [camera calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html),
[ArUco detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html),
and [PnP pose conventions and planar solvers](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).
