"""Geometric safeguards against the observed background-only crop failure."""
import unittest
import numpy as np
from gmic_fullfield import fullfield_array


class FullfieldTests(unittest.TestCase):
    def test_large_image_retains_all_four_corner_regions(self):
        raw=np.zeros((400,300),dtype=np.uint16)
        raw[:, :60]=200
        for y in [0,380]:
            for x in [0,280]:raw[y:y+20,x:x+20]=1000
        _,canvas,record=fullfield_array(raw,'L-MLO',200,150)
        self.assertEqual(record['source_window'],[0,400,0,300])
        self.assertFalse(record['geometric_cropping'])
        for y in [0,190]:
            for x in [0,140]:self.assertGreater(canvas[y:y+10,x:x+10].max(),500)

    def test_mirrored_views_have_identical_normalized_inputs(self):
        raw=np.zeros((80,50),dtype=np.uint16);raw[:, :20]=400;raw[10:60,20:35]=600
        for suffix in ['CC','MLO']:
            left,_,_=fullfield_array(raw,'L-'+suffix,100,100)
            right,_,_=fullfield_array(np.fliplr(raw),'R-'+suffix,100,100)
            np.testing.assert_array_equal(left,right)

    def test_small_images_are_not_enlarged_and_padding_matches_view(self):
        raw=np.zeros((80,50),dtype=np.uint16);raw[:, :20]=400
        _,cc,a=fullfield_array(raw,'L-CC',100,100)
        _,mlo,b=fullfield_array(raw,'L-MLO',100,100)
        self.assertEqual(a['scale'],1);self.assertEqual(b['scale'],1)
        np.testing.assert_array_equal(cc[10:90,:50],raw)
        np.testing.assert_array_equal(mlo[:80,:50],raw)


if __name__=='__main__':unittest.main()
