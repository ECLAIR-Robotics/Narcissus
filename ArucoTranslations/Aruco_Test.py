"""Estimate a plane's metric pose from three coplanar ArUco markers.

Run with --help; see README.md for setup, calibration and coordinate conventions.
"""

import argparse
import glob
import json
from pathlib import Path
import sys

import cv2
import numpy as np


tag_size = 36.47  # mm: EDIT THIS to the printed tag's OUTER BLACK square side.
TAG_IDS = (0, 3, 6)  # Three distinct IDs, in frame-defining order.
DICTIONARY = "DICT_ARUCO_ORIGINAL"  # Must match your physical tags.
CALIBRATION_MAX_DIMENSION = 1600  # Pixels; longest side used during calibration.
IMAGE_PREPROCESSING = "landscape_clockwise_resize_v1"


def positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Expected a finite positive number.")
    return number


def marker_corners(size):
    """OpenCV's canonical TL, TR, BR, BL order, with local +Y upward."""
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def prepare_image(image, target_size=None, max_dimension=CALIBRATION_MAX_DIMENSION):
    """Normalize portrait to landscape, then resize without changing field of view.

    Portrait images are rotated 90 degrees clockwise. This assumes the same
    camera/lens/crop and a consistent phone orientation/export convention.
    """
    if image.shape[0] > image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    height, width = image.shape[:2]
    if target_size is None:
        if max_dimension < 64:
            raise ValueError("--max-dimension must be at least 64 pixels.")
        scale = min(1.0, max_dimension / max(width, height))
        target_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    target_width, target_height = map(int, target_size)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Invalid processing resolution in calibration.")
    if not np.isclose(width / height, target_width / target_height, rtol=0.005):
        raise ValueError(
            f"Image {width}x{height} has a different aspect ratio from "
            f"{target_width}x{target_height}. Use the same camera aspect ratio/crop; "
            "resizing cannot correct a different field of view."
        )
    if (width, height) != (target_width, target_height):
        interpolation = cv2.INTER_AREA if target_width < width else cv2.INTER_LINEAR
        image = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    return image


def read_pose_image(image_path, calibration_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    with np.load(calibration_path, allow_pickle=False) as data:
        if "image_preprocessing" in data:
            method = str(data["image_preprocessing"].item())
            if method != IMAGE_PREPROCESSING:
                raise ValueError(f"Unsupported calibration image preprocessing: {method}")
            image = prepare_image(image, tuple(data["image_size"].astype(int)))
    # Old calibration files retain their original resolution/orientation behavior.
    return image


def load_calibration(path, image_size):
    with np.load(path, allow_pickle=False) as data:
        matrix = np.asarray(data["camera_matrix"], np.float64)
        distortion = np.asarray(data["dist_coeffs"], np.float64)
        calibrated_size = tuple(data["image_size"].astype(int))
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("camera_matrix must be a finite 3x3 matrix.")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive.")
    if distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
        raise ValueError("Expected standard OpenCV pinhole distortion coefficients.")
    if calibrated_size != tuple(image_size):
        raise ValueError(
            f"Calibration resolution {calibrated_size} differs from image {image_size}. "
            "Use the same resolution, crop, orientation and camera settings as calibration."
        )
    return matrix, distortion.reshape(-1, 1)


def detect_tags(image, dictionary_name, required_ids):
    if not dictionary_name.startswith("DICT_") or not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    if any(i < 0 or i >= len(dictionary.bytesList) for i in required_ids):
        raise ValueError("A requested tag ID is outside the selected dictionary.")
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    detected = {}
    for tag_id, points in zip([] if ids is None else ids.flatten(), corners):
        tag_id = int(tag_id)
        if tag_id in required_ids:
            if tag_id in detected:
                raise ValueError(f"ID {tag_id} appears more than once; use unique IDs.")
            detected[tag_id] = np.asarray(points, np.float64).reshape(4, 2)
    missing = set(required_ids) - detected.keys()
    if missing:
        raise ValueError(f"Missing required tags: {sorted(missing)}. All three must be visible.")
    return detected


def solve_pose(object_points, image_points, matrix, distortion, square=False):
    """Compare both planar PnP solutions; return the best positive-depth pose."""
    flag = cv2.SOLVEPNP_IPPE_SQUARE if square else cv2.SOLVEPNP_IPPE
    solved, rotations, translations, _ = cv2.solvePnPGeneric(
        object_points, image_points, matrix, distortion, flags=flag
    )
    initial_poses = list(zip(rotations, translations)) if solved else []
    # IPPE can be singular for an exactly centered, front-facing square.
    # Homography-initialized iterative PnP provides a fallback for that case.
    iterative_ok, iterative_rotation, iterative_translation = cv2.solvePnP(
        object_points, image_points, matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if iterative_ok:
        initial_poses.append((iterative_rotation, iterative_translation))
    candidates = []
    for rotation, translation in initial_poses:
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            continue
        # Refine each candidate using all its corner observations.
        rotation, translation = cv2.solvePnPRefineLM(
            object_points, image_points, matrix, distortion, rotation.copy(), translation.copy()
        )
        rot_matrix, _ = cv2.Rodrigues(rotation)
        camera_points = (rot_matrix @ object_points.T).T + translation.reshape(3)
        if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0):
            continue
        projected, _ = cv2.projectPoints(object_points, rotation, translation, matrix, distortion)
        rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))))
        candidates.append((rms, rotation, translation))
    if not candidates:
        raise ValueError("No valid pose in front of the camera.")
    candidates.sort(key=lambda item: item[0])
    best = candidates[0]
    ambiguous = False
    for candidate in candidates[1:]:
        relative, _ = cv2.Rodrigues(best[1])
        other, _ = cv2.Rodrigues(candidate[1])
        angle = np.arccos(np.clip((np.trace(relative.T @ other) - 1) / 2, -1, 1))
        if angle > np.deg2rad(2) and candidate[0] - best[0] < 0.25:
            ambiguous = True
    return best[1], best[2], best[0], ambiguous


def check_triangle(centers):
    a, b, c = centers
    longest = max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b))
    twice_area = np.linalg.norm(np.cross(b - a, c - a))
    if longest <= 1e-9 or twice_area / longest**2 < 0.02:
        raise ValueError("Tag centers are collinear or nearly collinear; arrange a broad triangle.")


def estimate_plane(detected, ids, size, matrix, distortion, layout=None):
    """Return plane-to-camera R,t; lengths are mm throughout."""
    warnings = []
    if layout is not None:
        # Layout entries are measured marker corners in the plane's Z=0 frame.
        object_points = np.concatenate([np.asarray(layout[str(i)], np.float64) for i in ids])
        if object_points.shape != (12, 3) or not np.isfinite(object_points).all():
            raise ValueError("Layout must contain four finite [x,y,z] corners per tag.")
        if not np.allclose(object_points[:, 2], 0, atol=1e-6):
            raise ValueError("Layout corners must lie on the plane Z=0, in mm.")
        canonical = marker_corners(size)
        for quad in object_points.reshape(3, 4, 3):
            distances = np.linalg.norm(quad[:, None] - quad[None, :], axis=2)
            expected = np.linalg.norm(canonical[:, None] - canonical[None, :], axis=2)
            if not np.allclose(distances, expected, rtol=0.01, atol=0.01):
                raise ValueError("Layout corners must form squares matching tag_size (within 1%).")
        check_triangle(object_points.reshape(3, 4, 3).mean(axis=1))
        image_points = np.concatenate([detected[i] for i in ids])
        rvec, tvec, rms, ambiguous = solve_pose(object_points, image_points, matrix, distortion)
        rotation, _ = cv2.Rodrigues(rvec)
        errors = {"joint": rms}
        mode = "measured_layout"
        if ambiguous:
            warnings.append("Two planar poses have similar reprojection errors; orientation is ambiguous.")
    else:
        centers = []
        errors = {}
        for tag_id in ids:
            _, translation, rms, ambiguous = solve_pose(
                marker_corners(size), detected[tag_id], matrix, distortion, square=True
            )
            centers.append(translation.reshape(3))
            errors[str(tag_id)] = rms
            if ambiguous:
                warnings.append(f"Tag {tag_id} has ambiguous planar pose; its center may be less reliable.")
        centers = np.asarray(centers)
        check_triangle(centers)
        tvec = centers.mean(axis=0).reshape(3, 1)
        x_axis = centers[1] - centers[0]
        x_axis /= np.linalg.norm(x_axis)
        normal = np.cross(x_axis, centers[2] - centers[0])
        normal /= np.linalg.norm(normal)
        if np.dot(normal, tvec.reshape(3)) > 0:
            normal = -normal  # In this mode +Z faces the camera.
        y_axis = np.cross(normal, x_axis)
        rotation = np.column_stack((x_axis, y_axis, normal))
        rvec, _ = cv2.Rodrigues(rotation)
        mode = "three_tag_centers"
        warnings.append(
            "Plane fitted to three independently estimated centers. Depth noise can tilt it; "
            "a measured layout and a joint 12-corner fit usually improve accuracy."
        )
    if max(errors.values()) > 2:
        warnings.append("Reprojection RMS exceeds 2 px; check calibration, blur, tag size and layout.")
    normal = rotation[:, 2]
    translation = tvec.reshape(3)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    result = {
        "mode": mode,
        "tag_ids": list(ids),
        "tag_size_mm": size,
        "perpendicular_distance_mm": float(abs(normal @ translation)),
        "distance_to_plane_origin_mm": float(np.linalg.norm(translation)),
        "plane_origin_depth_mm": float(translation[2]),
        "plane_origin_camera_mm": translation.tolist(),
        "plane_normal_camera": normal.tolist(),
        "plane_equation_camera": {"normal": normal.tolist(), "d_mm": float(-normal @ translation)},
        "rotation_vector_rad": rvec.reshape(3).tolist(),
        "rotation_matrix_plane_to_camera": rotation.tolist(),
        "transform_plane_to_camera_mm": transform.tolist(),
        "camera_position_plane_mm": (-rotation.T @ translation).tolist(),
        "reprojection_rms_px": errors,
        "warnings": warnings,
    }
    return result, rvec, tvec


def run_estimate(args):
    image = read_pose_image(args.image, args.calibration)
    matrix, distortion = load_calibration(args.calibration, (image.shape[1], image.shape[0]))
    if len(set(args.ids)) != 3:
        raise ValueError("Supply three distinct tag IDs.")
    detected = detect_tags(image, args.dictionary, args.ids)
    layout = None
    if args.layout:
        layout = json.loads(args.layout.read_text(encoding="utf-8"))
        if any(str(i) not in layout for i in args.ids):
            raise ValueError("Layout must contain every requested tag ID as a string key.")
    result, rvec, tvec = estimate_plane(detected, args.ids, args.tag_size, matrix, distortion, layout)
    cv2.aruco.drawDetectedMarkers(
        image, [detected[i].astype(np.float32).reshape(1, 4, 2) for i in args.ids],
        np.array(args.ids, dtype=np.int32).reshape(-1, 1)
    )
    cv2.drawFrameAxes(image, matrix, distortion, rvec, tvec, args.tag_size)
    label = f"Plane distance: {result['perpendicular_distance_mm']:.1f} mm"
    cv2.putText(image, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)
    if args.annotated:
        if not cv2.imwrite(str(args.annotated), image):
            raise ValueError(f"Cannot save annotated image: {args.annotated}")
    output = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if args.show:
        cv2.imshow("Plane pose (press any key to close)", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_calibrate(args):
    if args.cols < 3 or args.rows < 3:
        raise ValueError("Use at least 3x3 inner chessboard corners.")
    object_grid = np.zeros((args.cols * args.rows, 3), np.float32)
    object_grid[:, :2] = np.mgrid[:args.cols, :args.rows].T.reshape(-1, 2) * args.square_size
    object_points, image_points, paths = [], [], []
    image_size = None
    max_dimension = getattr(args, "max_dimension", CALIBRATION_MAX_DIMENSION)
    if max_dimension < 64:
        raise ValueError("--max-dimension must be at least 64 pixels.")
    filenames = sorted(glob.glob(args.images))
    if not filenames:
        raise ValueError(
            f"No files match {args.images!r}. Current directory: {Path.cwd()}. "
            "Check the photo folder and file extensions (.jpg, .jpeg, .png). "
            "Paths are relative to the current directory."
        )
    for filename in filenames:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping unreadable image: {filename}", file=sys.stderr)
            continue
        original_size = (image.shape[1], image.shape[0])
        try:
            image = prepare_image(image, image_size, max_dimension)
        except ValueError as error:
            print(f"Skipping {filename}: {error}", file=sys.stderr)
            continue
        size = (image.shape[1], image.shape[0])
        if original_size != size:
            print(f"Preparing {filename}: {original_size} -> {size}", file=sys.stderr)
        found, corners = cv2.findChessboardCornersSB(
            image, (args.cols, args.rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            print(f"Chessboard not found: {filename}", file=sys.stderr)
            continue
        if image_size is None:
            image_size = size  # First usable board defines the shared resolution.
            print(f"Calibration processing resolution: {size[0]}x{size[1]}", file=sys.stderr)
        object_points.append(object_grid.copy())
        image_points.append(corners)
        paths.append(filename)
    if len(paths) < 10:
        raise ValueError(f"Only {len(paths)} usable images; need at least 10, preferably 20-30 varied views.")
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    per_view = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rotations, translations):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, distortion)
        per_view.append(float(np.sqrt(np.mean(np.sum((projected - img) ** 2, axis=2)))))
    if not np.isfinite(rms) or not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError("Calibration failed; capture more varied, sharp chessboard images.")
    np.savez(
        args.output, camera_matrix=matrix, dist_coeffs=distortion,
        image_size=np.array(image_size), rms_px=rms,
        image_preprocessing=np.array(IMAGE_PREPROCESSING),
        per_view_rms_px=np.array(per_view), image_paths=np.array(paths),
    )
    print(f"Saved {args.output}: {len(paths)} views, overall RMS {rms:.3f} px")
    for path, error in zip(paths, per_view):
        print(f"  {error:.3f} px  {path}")
    if rms > 1:
        print("High RMS: inspect high-error images and recapture sharp, varied views.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    estimate = commands.add_parser("estimate", help="Estimate plane pose from one image.")
    estimate.add_argument("image", type=Path)
    estimate.add_argument("--calibration", type=Path, required=True, help="Calibration .npz file.")
    estimate.add_argument("--tag-size", type=positive_float, default=tag_size, help="Outer black side in mm.")
    estimate.add_argument("--ids", type=int, nargs=3, default=TAG_IDS)
    estimate.add_argument("--dictionary", default=DICTIONARY)
    estimate.add_argument("--layout", type=Path, help="Optional measured corner coordinates JSON (mm).")
    estimate.add_argument("--output", type=Path, help="Save pose JSON.")
    estimate.add_argument("--annotated", type=Path, help="Save image with tags and pose axes.")
    estimate.add_argument("--show", action="store_true")
    estimate.set_defaults(run=run_estimate)
    calibrate = commands.add_parser("calibrate", help="Calibrate from chessboard photos.")
    calibrate.add_argument("--images", required=True, help='Quoted image glob, e.g. "calibration/*.jpg".')
    calibrate.add_argument("--cols", type=int, default=9, help="Inner corner columns (default 9).")
    calibrate.add_argument("--rows", type=int, default=6, help="Inner corner rows (default 6).")
    calibrate.add_argument("--square-size", type=positive_float, required=True, help="Chessboard square side in mm.")
    calibrate.add_argument("--output", type=Path, default=Path("camera_calibration.npz"))
    calibrate.add_argument("--max-dimension", type=int, default=CALIBRATION_MAX_DIMENSION,
                           help="Longest processing side in pixels (default 1600); originals are unchanged.")
    calibrate.set_defaults(run=run_calibrate)
    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, OSError, KeyError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
