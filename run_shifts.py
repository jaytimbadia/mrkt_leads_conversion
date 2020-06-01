import unittest
from model import eval_test_case

class Test(unittest.TestCase):
    # just a single naive Test case to assure version shift, also allows to integrate jenkins.

    def test_version_shift(self):
        assert True == eval_test_case.version_shift()

if __name__ == '__main__':
    unittest.main()