"""Synthetic geometry tests; no camera or test image downloads required."""

import tempfile
import contextlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

# Apply before importing OpenCV so its worker threads also avoid GPU state.
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

import cv2
import numpy as np

from Aruco_Test import (detect_tags, estimate_plane, load_calibration, marker_corners,
                        run_calibrate, solve_pose, prepare_image, read_pose_image,
                        IMAGE_PREPROCESSING)


class PlanePoseTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[1100., 0, 640], [0, 1100., 480], [0, 0, 1]])
        self.distortion = np.array([0.02, -0.01, 0.001, -0.001, 0.])
        self.size = 50.
        self.ids = (0, 1, 2)
        self.centers = np.array([[-100., 50, 0], [100., 50, 0], [0., -100, 0]])
        # Face local +Z toward the camera, with an additional oblique tilt.
        tilt, _ = cv2.Rodrigues(np.array([0.20, -0.30, 0.08]))
        self.rotation = tilt @ np.diag([1., -1., -1.])
        self.rvec, _ = cv2.Rodrigues(self.rotation)
        self.translation = np.array([[35.], [-20.], [850.]])
        self.layout = {
            str(i): (marker_corners(self.size) + c).tolist()
            for i, c in zip(self.ids, self.centers)
        }
        self.detected = {}
        for i in self.ids:
            projected, _ = cv2.projectPoints(
                np.array(self.layout[str(i)]), self.rvec, self.translation,
                self.matrix, self.distortion,
            )
            self.detected[i] = projected.reshape(4, 2)

    def estimate(self, layout=None, size=None, detected=None):
        return estimate_plane(
            self.detected if detected is None else detected, self.ids,
            self.size if size is None else size, self.matrix, self.distortion, layout
        )[0]

    def test_measured_layout_recovers_full_pose_and_distances(self):
        result = self.estimate(self.layout)
        np.testing.assert_allclose(result["plane_origin_camera_mm"], self.translation.ravel(), atol=1e-5)
        np.testing.assert_allclose(result["rotation_matrix_plane_to_camera"], self.rotation, atol=1e-6)
        distance = abs(self.rotation[:, 2] @ self.translation.ravel())
        self.assertAlmostEqual(result["perpendicular_distance_mm"], distance, places=5)
        self.assertAlmostEqual(result["distance_to_plane_origin_mm"], np.linalg.norm(self.translation), places=5)
        np.testing.assert_allclose(result["camera_position_plane_mm"], -self.rotation.T @ self.translation.ravel(), atol=1e-5)

    def test_size_only_recovers_center_and_plane(self):
        result = self.estimate()
        np.testing.assert_allclose(result["plane_origin_camera_mm"], self.translation.ravel(), atol=1e-5)
        np.testing.assert_allclose(result["plane_normal_camera"], self.rotation[:, 2], atol=1e-6)
        axes = np.array(result["rotation_matrix_plane_to_camera"])
        np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(axes), 1)

    def test_tag_size_scales_distance(self):
        normal = self.estimate()
        doubled = self.estimate(size=self.size * 2)
        self.assertAlmostEqual(doubled["perpendicular_distance_mm"], 2 * normal["perpendicular_distance_mm"], places=5)

    def test_centered_front_facing_square(self):
        obj = marker_corners(self.size)
        rvec = np.array([np.pi, 0., 0.])
        tvec = np.array([0., 0., 1000.])
        projected, _ = cv2.projectPoints(obj, rvec, tvec, self.matrix, np.zeros(5))
        _, estimated, rms, _ = solve_pose(obj, projected.reshape(4, 2), self.matrix, np.zeros(5), square=True)
        np.testing.assert_allclose(estimated.ravel(), tvec, atol=1e-5)
        self.assertLess(rms, 1e-6)

    def test_collinear_centers_are_rejected(self):
        detected = {}
        for i in self.ids:
            obj = marker_corners(self.size) + np.array([i * 100., 0, 0])
            points, _ = cv2.projectPoints(obj, self.rvec, self.translation, self.matrix, self.distortion)
            detected[i] = points.reshape(4, 2)
        with self.assertRaisesRegex(ValueError, "collinear"):
            self.estimate(detected=detected)

    def test_calibration_rejects_wrong_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "camera.npz"
            np.savez(path, camera_matrix=self.matrix, dist_coeffs=self.distortion, image_size=[1280, 960])
            matrix, _ = load_calibration(path, (1280, 960))
            np.testing.assert_equal(matrix, self.matrix)
            with self.assertRaisesRegex(ValueError, "resolution"):
                load_calibration(path, (640, 480))

    def test_resize_normalizes_portrait_and_preserves_geometry(self):
        image = np.zeros((1200, 1600, 3), np.uint8)
        image[100:300, 200:400] = 255
        expected = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA)
        portrait = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        np.testing.assert_equal(prepare_image(portrait, (800, 600)), expected)
        self.assertEqual(prepare_image(image, max_dimension=1000).shape, (750, 1000, 3))
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            prepare_image(np.zeros((900, 1600), np.uint8), (800, 600))

    def test_pose_image_uses_saved_preprocessing(self):
        with tempfile.TemporaryDirectory() as folder:
            photo = Path(folder) / "portrait.png"
            calibration = Path(folder) / "calibration.npz"
            image = np.zeros((1600, 1200, 3), np.uint8)
            cv2.imwrite(str(photo), image)
            np.savez(calibration, image_size=[800, 600], image_preprocessing=IMAGE_PREPROCESSING)
            self.assertEqual(read_pose_image(photo, calibration).shape, (600, 800, 3))
            # Existing calibration files without metadata keep their old behavior.
            np.savez(calibration, image_size=[1200, 1600])
            self.assertEqual(read_pose_image(photo, calibration).shape, image.shape)

    def test_noisy_joint_fit_remains_close(self):
        rng = np.random.default_rng(42)
        noisy = {i: p + rng.normal(0, 0.15, p.shape) for i, p in self.detected.items()}
        result = self.estimate(self.layout, detected=noisy)
        np.testing.assert_allclose(result["plane_origin_camera_mm"], self.translation.ravel(), atol=2)
        self.assertLess(result["reprojection_rms_px"]["joint"], 0.5)

    def test_detection_and_pose_on_rendered_image(self):
        # Render perspective-warped marker bitmaps, then detect rather than
        # passing exact projected corners directly to the estimator.
        image = np.full((960, 1280, 3), 255, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        bitmap_corners = np.float32([[0, 0], [399, 0], [399, 399], [0, 399]])
        zero_distortion = np.zeros(5)
        for i in self.ids:
            projected, _ = cv2.projectPoints(
                np.array(self.layout[str(i)]), self.rvec, self.translation,
                self.matrix, zero_distortion,
            )
            marker = cv2.aruco.generateImageMarker(dictionary, i, 400)
            homography = cv2.getPerspectiveTransform(bitmap_corners, projected.astype(np.float32))
            warped = cv2.warpPerspective(marker, homography, (1280, 960), borderValue=255)
            image = np.minimum(image, cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR))
        detected = detect_tags(image, "DICT_4X4_50", self.ids)
        result, _, _ = estimate_plane(detected, self.ids, self.size, self.matrix, zero_distortion, self.layout)
        np.testing.assert_allclose(result["plane_origin_camera_mm"], self.translation.ravel(), atol=5)
        self.assertLess(result["reprojection_rms_px"]["joint"], 1)
        with self.assertRaisesRegex(ValueError, "Missing required tags"):
            detect_tags(np.full_like(image, 255), "DICT_4X4_50", self.ids)

    def test_calibration_from_rendered_chessboards(self):
        # A 10x7 square board contains 9x6 inner corners; project 14 views
        # with known pinhole intrinsics, then run the actual image pipeline.
        square_px = 80
        board = np.full((7 * square_px, 10 * square_px), 255, np.uint8)
        for row in range(7):
            for col in range(10):
                if (row + col) % 2 == 0:
                    board[row * square_px:(row + 1) * square_px,
                          col * square_px:(col + 1) * square_px] = 0
        src = np.float32([[0, 0], [799, 0], [799, 559], [0, 559]])
        obj = np.array([[-125., -87.5, 0], [125., -87.5, 0],
                        [125., 87.5, 0], [-125., 87.5, 0]])
        rng = np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as folder:
            for i in range(14):
                rvec = rng.uniform([-0.45, -0.5, -0.2], [0.45, 0.5, 0.2])
                tvec = rng.uniform([-130., -80., 650.], [130., 80., 1000.])
                projected, _ = cv2.projectPoints(obj, rvec, tvec, self.matrix, np.zeros(5))
                transform = cv2.getPerspectiveTransform(src, projected.astype(np.float32))
                image = cv2.warpPerspective(board, transform, (1280, 960), borderValue=200)
                if i % 2:
                    image = cv2.resize(image, (2560, 1920))
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                self.assertTrue(cv2.imwrite(str(Path(folder) / f"board_{i:02}.png"), image))
            output = Path(folder) / "calibration.npz"
            args = SimpleNamespace(cols=9, rows=6, square_size=25.,
                                   images=str(Path(folder) / "*.png"), output=output)
            with contextlib.redirect_stdout(io.StringIO()):
                run_calibrate(args)
            matrix, _ = load_calibration(output, (1280, 960))
            np.testing.assert_allclose(matrix[:2, :2], self.matrix[:2, :2], atol=25)
            with np.load(output) as data:
                self.assertEqual(data["image_preprocessing"].item(), IMAGE_PREPROCESSING)


if __name__ == "__main__":
    unittest.main()
