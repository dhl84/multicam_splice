"""Content-classification cache key invalidation (feedback issue 6)."""
import os
import tempfile
import time
import unittest
from pathlib import Path

from multicam import classify


class TestCacheKey(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.f1 = Path(self.tmp) / "a.mp4"
        self.f2 = Path(self.tmp) / "b.mp4"
        self.f1.write_bytes(b"aaaa")
        self.f2.write_bytes(b"bbbb")
        self.base = ([self.f1, self.f2], "olive hoodie",
                     "claude-haiku-4-5-20251001", 0.5, 600.0)

    def key(self, *args):
        return classify._cache_key(*args)

    def test_identical_inputs_same_key(self):
        self.assertEqual(self.key(*self.base), self.key(*self.base))

    def test_coach_description_changes_key(self):
        alt = ([self.f1, self.f2], "red shirt",
               "claude-haiku-4-5-20251001", 0.5, 600.0)
        self.assertNotEqual(self.key(*self.base), self.key(*alt))

    def test_model_changes_key(self):
        alt = ([self.f1, self.f2], "olive hoodie", "other-model", 0.5, 600.0)
        self.assertNotEqual(self.key(*self.base), self.key(*alt))

    def test_fps_changes_key(self):
        alt = ([self.f1, self.f2], "olive hoodie",
               "claude-haiku-4-5-20251001", 1.0, 600.0)
        self.assertNotEqual(self.key(*self.base), self.key(*alt))

    def test_max_seconds_changes_key(self):
        alt = ([self.f1, self.f2], "olive hoodie",
               "claude-haiku-4-5-20251001", 0.5, 1200.0)
        self.assertNotEqual(self.key(*self.base), self.key(*alt))

    def test_file_order_changes_key(self):
        alt = ([self.f2, self.f1], "olive hoodie",
               "claude-haiku-4-5-20251001", 0.5, 600.0)
        self.assertNotEqual(self.key(*self.base), self.key(*alt))

    def test_file_content_change_invalidates(self):
        before = self.key(*self.base)
        time.sleep(0.01)
        self.f1.write_bytes(b"aaaaaaaa")          # different size
        os.utime(self.f1, (time.time() + 5, time.time() + 5))
        self.assertNotEqual(before, self.key(*self.base))

    def test_cache_path_uses_first_stem_and_is_local(self):
        p = classify._cache_path(*self.base)
        self.assertEqual(p.parent, self.f1.parent)
        self.assertTrue(p.name.startswith("a_content_"))
        self.assertTrue(p.name.endswith(".npy"))


if __name__ == "__main__":
    unittest.main()
