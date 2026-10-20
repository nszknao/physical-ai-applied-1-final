"""Tests for duckscout.perception using RGB images without simulator state."""

import unittest

import cv2
import numpy as np

from duckscout.perception import annotate_detection, detect_target


class DetectionTests(unittest.TestCase):
    def test_rgb_red_has_exact_box_centroid_and_area(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[60:80, 100:130] = (255, 0, 0)
        before = image.copy()
        result = detect_target(image)
        self.assertEqual(result.bbox, (100, 60, 30, 20))
        self.assertEqual(result.center, (114.5, 69.5))
        self.assertEqual(result.area, 600)
        np.testing.assert_array_equal(image, before)

    def test_both_ends_of_red_hue_range_are_detected(self):
        for hue in (0, 5, 175, 179):
            with self.subTest(hue=hue):
                hsv = np.full((12, 12, 3), (hue, 200, 180), dtype=np.uint8)
                image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                self.assertEqual(detect_target(image).area, 144)

    def test_background_blue_green_gray_pale_and_dark_are_not_targets(self):
        for color in (
            (0, 0, 0),
            (0, 0, 255),
            (0, 255, 0),
            (180, 180, 180),
            (255, 210, 210),
            (30, 0, 0),
        ):
            with self.subTest(color=color):
                self.assertIsNone(
                    detect_target(np.full((40, 40, 3), color, dtype=np.uint8))
                )

    def test_disconnected_noise_is_not_combined_into_one_target(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[::4, ::4] = (255, 0, 0)
        self.assertIsNone(detect_target(image))
        image[30:35, 30:35] = (255, 0, 0)
        self.assertEqual(detect_target(image).bbox, (30, 30, 5, 5))

    def test_largest_component_is_selected_by_red_area_not_box_area(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        image[10:50, 10] = (255, 0, 0)
        image[49, 10:50] = (255, 0, 0)  # Large bounding box, only 79 red pixels.
        image[70:80, 100:110] = (255, 0, 0)  # 100 red pixels.
        result = detect_target(image)
        self.assertEqual(result.bbox, (100, 70, 10, 10))
        self.assertEqual(result.area, 100)

    def test_partly_visible_targets_at_every_edge_are_detected(self):
        for x, y in ((0, 40), (90, 40), (40, 0), (40, 90)):
            with self.subTest(x=x, y=y):
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                image[y : y + 10, x : x + 10] = (255, 0, 0)
                self.assertEqual(detect_target(image).bbox, (x, y, 10, 10))

    def test_loss_and_reacquisition_do_not_reuse_old_detection(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[80:100, 20:40] = (255, 0, 0)
        self.assertIsNotNone(detect_target(image))
        image[:] = 0
        self.assertIsNone(detect_target(image))
        image[90:110, 200:220] = (255, 0, 0)
        self.assertEqual(detect_target(image).bbox, (200, 90, 20, 20))

    def test_invalid_images_are_rejected_before_native_processing(self):
        for image in (
            np.zeros((0, 10, 3), dtype=np.uint8),
            np.zeros((10, 0, 3), dtype=np.uint8),
            np.zeros((10, 10), dtype=np.uint8),
            np.zeros((10, 10, 4), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.float32),
        ):
            with self.subTest(shape=image.shape, dtype=image.dtype):
                with self.assertRaisesRegex(ValueError, "RGB"):
                    detect_target(image)


class AnnotationTests(unittest.TestCase):
    def test_marks_are_on_a_copy_and_missing_target_clears_previous_box(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[80:100, 100:120] = (255, 0, 0)
        before = image.copy()
        preview = annotate_detection(image, detect_target(image))
        np.testing.assert_array_equal(image, before)
        np.testing.assert_array_equal(preview[80, 100], [0, 255, 0])
        np.testing.assert_array_equal(preview[90, 110], [255, 255, 0])
        missing = np.zeros_like(image)
        next_preview = annotate_detection(missing, detect_target(missing))
        self.assertTrue((next_preview[30:] == 0).all())
        self.assertTrue((next_preview[:30] != 0).any())  # Status text remains.


if __name__ == "__main__":
    unittest.main()
