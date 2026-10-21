"""Dependency-light contract tests; no training task registration required."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

path = Path(__file__).resolve().parents[1]/'src/vibe/sim2sim/runtime.py'
if not path.exists(): path=Path(__file__).parent/'src/runtime.py'
spec=importlib.util.spec_from_file_location('native_runtime',path)
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)

class NativeContracts(unittest.TestCase):
    def test_history_backfill_and_chronology(self):
        h=runtime.History([np.array([1.,2.]),np.array([3.])],length=3)
        np.testing.assert_array_equal(h.flat(),[[1,2,1,2,1,2,3,3,3]])
        h.append([np.array([4.,5.]),np.array([6.])])
        np.testing.assert_array_equal(h.flat(),[[1,2,1,2,4,5,3,3,6]])
        h.append([np.array([7.,8.]),np.array([9.])])
        np.testing.assert_array_equal(h.flat(),[[1,2,4,5,7,8,3,6,9]])
    def test_bgr_boundary_no_mutation(self):
        rgb=np.array([[[1,2,3],[4,5,6]]],np.uint8); saved=rgb.copy()
        bgr=runtime.bgr_input(rgb)
        np.testing.assert_array_equal(bgr,[[[[3,2,1],[6,5,4]]]])
        bgr[:]=0
        np.testing.assert_array_equal(rgb,saved)
        self.assertTrue(bgr.flags.c_contiguous)
        with self.assertRaises(ValueError): runtime.bgr_input(rgb.astype(np.float32))
    def test_deadline_compensates_overrun_without_skipping(self):
        with patch.object(runtime.time,'perf_counter',side_effect=[10.,10.005,10.05,10.055]), patch.object(runtime.time,'sleep') as sleep:
            d=runtime.Deadline(.02)
            d.wait(); d.wait(); d.wait()
            self.assertEqual(sleep.call_count,2)
            self.assertAlmostEqual(sleep.call_args_list[0].args[0],.015)
            self.assertAlmostEqual(sleep.call_args_list[1].args[0],.005)
    def test_link_angular_velocity_not_inertial_local_velocity(self):
        # Root LINK axes rotated 90 deg, independently of inertial principal axes.
        rot=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        case=object.__new__(runtime.NativeCase)
        from types import SimpleNamespace
        case.root=0; case.qids=np.array([7]); case.vids=np.array([6])
        case.default=np.array([.1]);case.last=np.array([.2])
        case.data=SimpleNamespace(xmat=rot.reshape(1,9),cvel=np.array([[1,0,0,0,0,0.]]),qpos=np.array([0]*7+[.3]),qvel=np.array([0]*6+[.4]))
        av,jp,jv,action,g=case.terms()
        np.testing.assert_array_equal(av,[0,-1,0])
        np.testing.assert_allclose(jp,[.2]);np.testing.assert_array_equal(g,[0,0,-1])

if __name__=='__main__': unittest.main()
