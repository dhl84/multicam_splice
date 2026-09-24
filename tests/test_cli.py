"""CLI entry point (feedback issue 7)."""
import unittest

from multicam.__main__ import main


class TestCli(unittest.TestCase):
    def test_no_args_returns_usage_code(self):
        self.assertEqual(main([]), 2)

    def test_too_many_args_returns_usage_code(self):
        self.assertEqual(main(["a.json", "b.json"]), 2)

    def test_main_is_zero_arg_callable_for_console_script(self):
        # the console script (multicam-splice) calls main() with no arguments;
        # it must default argv to sys.argv[1:] instead of raising TypeError
        import inspect
        sig = inspect.signature(main)
        params = list(sig.parameters.values())
        self.assertTrue(all(p.default is not inspect.Parameter.empty
                            for p in params))


if __name__ == "__main__":
    unittest.main()
