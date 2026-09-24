"""FCPXML timecode parsing, file URLs, and probe-driven audio metadata
(feedback issues 8, 9)."""
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from multicam import fcpxml


class TestParseTimecode(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(fcpxml.parse_timecode("", 60000, 1001), (0, False))

    def test_ndf(self):
        fn, drop = fcpxml.parse_timecode("01:00:00:00", 30000, 1001)
        self.assertFalse(drop)
        self.assertEqual(fn, 3600 * 30)  # 30 rounded fps * 3600 s

    def test_drop_frame_marker(self):
        fn, drop = fcpxml.parse_timecode("00:10:00;00", 30000, 1001)
        self.assertTrue(drop)
        # 10 minutes drop-frame drops 2 frames per minute except every 10th
        self.assertEqual(fn, 10 * 60 * 30 - (2 * (10 - 1)))


class TestFileUrl(unittest.TestCase):
    def test_round_trips_spaces(self):
        p = Path("/tmp/a b/clip 1.mp4")
        url = fcpxml.file_url(p)
        self.assertTrue(url.startswith("file://"))
        self.assertIn("%20", url)


def _emit(assets):
    seq = fcpxml.Sequence(
        fps_num=60000, fps_den=1001, width=3840, height=2160,
        items=[fcpxml.VideoClip(ref=assets[0].id, name="c", tl_off_f=0,
                                dur_f=60, media_in_f=0)],
        assets=assets)
    import tempfile
    out = Path(tempfile.mkstemp(suffix=".fcpxml")[1])
    fcpxml.build(seq, out)
    return ET.parse(out).getroot()


class TestAssetAudioMetadata(unittest.TestCase):
    def test_audio_metadata_reflects_probe(self):
        a = fcpxml.Asset(id="v1", name="x", path=Path("/tmp/x.mp4"),
                         start_f=0, dur_f=60, has_audio=True,
                         audio_channels=1, audio_rate=44100)
        asset = _emit([a]).find(".//asset")
        self.assertEqual(asset.get("hasAudio"), "1")
        self.assertEqual(asset.get("audioChannels"), "1")
        self.assertEqual(asset.get("audioRate"), "44100")

    def test_audio_present_but_specifics_unknown_omits_them(self):
        # has audio but probe didn't report channels/rate -> declare audio,
        # never fabricate stereo/48k
        a = fcpxml.Asset(id="v1", name="x", path=Path("/tmp/x.mp4"),
                         start_f=0, dur_f=60, has_audio=True,
                         audio_channels=0, audio_rate=0)
        asset = _emit([a]).find(".//asset")
        self.assertEqual(asset.get("hasAudio"), "1")
        self.assertIsNone(asset.get("audioChannels"))
        self.assertIsNone(asset.get("audioRate"))

    def test_silent_asset_declares_no_audio(self):
        a = fcpxml.Asset(id="v1", name="x", path=Path("/tmp/x.mp4"),
                         start_f=0, dur_f=60, has_audio=False,
                         audio_channels=0, audio_rate=0)
        asset = _emit([a]).find(".//asset")
        self.assertEqual(asset.get("hasAudio"), "0")
        self.assertIsNone(asset.get("audioChannels"))
        self.assertIsNone(asset.get("audioRate"))


if __name__ == "__main__":
    unittest.main()
