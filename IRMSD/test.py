from math import sqrt
import unittest

import numpy as np

from IRMSD import Conformations
from IRMSD import align_array
import IRMSD

np.set_printoptions(linewidth=120)
# From the pyqcprot example code
struct_1 = [[[ -2.803,   0.893,   1.368,  -1.651,  -0.44 ,   2.551,   0.105],
             [-15.373, -16.062, -12.371, -12.153, -15.218, -13.273, -11.33 ],
             [ 24.556,  25.147,  25.885,  28.177,  30.068,  31.372, 33.567]]]

struct_2 = [[[-14.739, -12.473, -14.802, -17.782, -16.124, -15.029, -18.577],
            [-18.673, -15.81 , -13.307, -14.852, -14.617, -11.037, -10.001],
            [ 15.04 ,  16.074,  14.408,  16.171,  19.584,  18.902,  17.996]]]


class TestIRMSD(unittest.TestCase):
    def test_data_type(self):
        f32_structure = align_array(np.array(struct_1), 'axis')
        f64_structure = np.empty(f32_structure.shape, dtype=np.float64)
        f64_structure[:, :, :] = f32_structure

        self.assertRaises(IRMSD.FloatPrecisionError,
                          Conformations, f64_structure, 'axis', 7)

    def test_dimensions(self):
        axis2d = np.empty((4, 4), dtype=np.float32)
        self.assertRaises(IRMSD.DimensionError,
                          Conformations, axis2d, 'axis', 4)

        non_r3 = np.empty((4, 4, 4), dtype=np.float32)
        self.assertRaises(IRMSD.NumberOfAxesError,
                          Conformations, non_r3, 'axis', 4)

        unpadded_axis = np.empty((1, 3, 5), dtype=np.float32)
        self.assertRaises(IRMSD.NumberOfAtomsError,
                          Conformations, unpadded_axis, 'axis', 5)

        unpadded_atom = np.empty((1, 5, 3), dtype=np.float32)
        self.assertRaises(IRMSD.NumberOfAtomsError,
                          Conformations, unpadded_atom, 'atom', 5)

    def test_zeros(self):
        return
        confs = align_array(np.zeros((2, 3, 8)), 'axis')
        conf_obj = Conformations(confs, 'axis', 8)
        rmsds = conf_obj.rmsds_to_reference(conf_obj, 1)
        self.assertEqual(rmsds[0], 0)
        self.assertEqual(rmsds[1], 0)

    def test_centering(self):
        return
        confs = align_array(np.zeros((2, 3, 8)), 'axis')
        confs[1,:,:] += 1
        conf_obj = Conformations(confs, 'axis', 8)
        rmsds = conf_obj.rmsds_to_reference(conf_obj, 1)
        self.assertEqual(rmsds[0], 0)
        self.assertEqual(rmsds[1], 0)
        pass

    def test_simple_structure(self):
        return
        confs = align_array(np.zeros((2, 3, 8)), 'axis')
        confs[1,:,0] += 1
        conf_obj = Conformations(confs, 'axis', 2)
        rmsds = conf_obj.rmsds_to_reference(conf_obj, 1)
        self.assertAlmostEqual(rmsds[0], sqrt(3.0/4))
        self.assertEqual(rmsds[1], 0)

    def test_accuracy(self):
        confs = align_array(np.vstack(map(np.array, (struct_1, struct_2))),
                            'axis')
        conf_obj = Conformations(confs, 'axis', 7)
        rmsds = conf_obj.rmsds_to_reference(conf_obj, 0)
        self.assertAlmostEqual(rmsds[0], 0)
        self.assertAlmostEqual(rmsds[1], 0.719106, places=6)

if __name__ == "__main__":
    unittest.main()
