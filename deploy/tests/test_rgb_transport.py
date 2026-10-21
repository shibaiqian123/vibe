import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim'))
from rgb_transport import Frame,pack,unpack,match
class RGBTests(unittest.TestCase):
    def setUp(self):self.f=Frame(2,40,.2,np.full((63,112,3),127,np.uint8))
    def test_roundtrip(self):
        restored=unpack(pack(self.f));np.testing.assert_array_equal(match(restored,2,40,.2),self.f.rgb)
    def test_reset(self):
        with self.assertRaises(ValueError):match(self.f,3,40,.2)
    def test_stale(self):
        with self.assertRaises(ValueError):match(self.f,2,44,.22)
    def test_color_size(self):
        parts=pack(self.f);parts[1]=parts[1][:-1]
        with self.assertRaises(ValueError):unpack(parts)
    def test_source_immutable(self):
        restored=unpack(pack(self.f));restored.rgb[:]=0
        self.assertEqual(self.f.rgb[0,0,0],127)
if __name__=='__main__':unittest.main()
