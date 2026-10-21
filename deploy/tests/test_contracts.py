import sys,unittest,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'input'))
from wire import decode,describe,packer
class ProtocolTests(unittest.TestCase):
    def test_command_uses_upstream(self):
        topic,fields=decode(packer.build_command_message(True,False,False))
        self.assertEqual(topic,'command');self.assertEqual(fields['start'][0],1)
    def test_planner_not_smpl(self):
        topic,fields=decode(packer.build_planner_message(0,[0,0,0],[1,0,0]))
        self.assertEqual(topic,'planner');self.assertNotIn('smpl_joints',fields)
        self.assertFalse(describe(topic,fields)['deployment_ready'])
    def test_pose_roundtrip(self):
        source=np.arange(72,dtype=np.float32).reshape(1,24,3)
        _,fields=decode(packer.pack_pose_message({'smpl_joints':source}))
        np.testing.assert_array_equal(fields['smpl_joints'],source)
    def test_reject_truncated(self):
        with self.assertRaises(ValueError):decode(packer.build_command_message(True,False,False)[:-1])
    def test_reject_trailing(self):
        with self.assertRaises(ValueError):decode(packer.build_command_message(True,False,False)+b'0')
    def test_reject_nan(self):
        with self.assertRaises(ValueError):decode(packer.pack_pose_message({'smpl_joints':np.array([np.nan],np.float32)}))
    def test_unknown_topic(self):
        with self.assertRaises(ValueError):decode(b'nonsense')
if __name__=='__main__':unittest.main()
