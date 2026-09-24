"""Timebase / format validation (feedback issue 1)."""
import unittest
from pathlib import Path

from multicam.probe import MediaInfo, validate_format


def info(name="clip.mp4", fps=(60000, 1001), avg=None, wh=(3840, 2160)):
    n, d = fps
    an, ad = avg if avg else fps
    return MediaInfo(
        path=Path(name), duration=10.0, fps_num=n, fps_den=d,
        width=wh[0], height=wh[1], n_frames=600, timecode="",
        sample_rate=48000, has_audio=True, channels=2,
        avg_fps_num=an, avg_fps_den=ad)


# project format used throughout: 3840x2160 @ 60000/1001 (59.94)
PROJ = dict(fps_num=60000, fps_den=1001, width=3840, height=2160)


class TestValidateFormat(unittest.TestCase):
    def test_matching_media_passes(self):
        validate_format([info(), info()], **PROJ)  # no raise

    def test_vfr_is_rejected(self):
        # nominal 30 fps but average 27 fps -> variable frame rate
        vfr = info(fps=(30, 1), avg=(27, 1))
        self.assertTrue(vfr.is_vfr)
        with self.assertRaises(ValueError) as cm:
            validate_format([vfr], **PROJ)
        self.assertIn("variable frame rate", str(cm.exception))
        self.assertIn("clip.mp4", str(cm.exception))

    def test_cfr_not_flagged_vfr(self):
        self.assertFalse(info().is_vfr)
        self.assertFalse(info(fps=(30, 1), avg=(30, 1)).is_vfr)

    def test_fps_mismatch_is_rejected(self):
        # 30 fps source against a 59.94 project
        with self.assertRaises(ValueError) as cm:
            validate_format([info(fps=(30, 1), avg=(30, 1))], **PROJ)
        self.assertIn("does not match", str(cm.exception))

    def test_2997_vs_30_is_a_mismatch(self):
        # 29.97 vs 30 is a real drift source, not a rounding tolerance
        with self.assertRaises(ValueError):
            validate_format([info(fps=(30000, 1001), avg=(30000, 1001))],
                            fps_num=30, fps_den=1, width=3840, height=2160)

    def test_exact_rational_match_passes(self):
        validate_format([info(fps=(30000, 1001), avg=(30000, 1001))],
                        fps_num=30000, fps_den=1001, width=3840, height=2160)

    def test_raster_mismatch_is_rejected(self):
        with self.assertRaises(ValueError) as cm:
            validate_format([info(wh=(1920, 1080))], **PROJ)
        self.assertIn("raster", str(cm.exception))

    def test_all_problems_reported_together(self):
        bad = info(fps=(30, 1), avg=(30, 1), wh=(1920, 1080))
        with self.assertRaises(ValueError) as cm:
            validate_format([bad], **PROJ)
        msg = str(cm.exception)
        self.assertIn("does not match", msg)
        self.assertIn("raster", msg)


if __name__ == "__main__":
    unittest.main()
