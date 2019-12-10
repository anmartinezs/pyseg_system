"""

Tests module pickled and multi_data.
 
# Author: Vladan Lucic
# $Id: test_pickled.py 1461 2017-10-12 10:10:49Z vladan $
"""

__version__ = "$Revision: 1461 $"

from copy import copy, deepcopy
import pickle
import os.path
import unittest

import numpy
import numpy.testing as np_test 
import scipy

#import common
import pyto.scene.test.common as scene_cmn
import pyto.analysis.test.common as analysis_cmn
from pyto.io.multi_data import MultiData
from pyto.io.pickled import Pickled


class TestPickled(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        Makes pickles (analysis.CleftLayers objects)
        """

        # define files and convert paths to absolute
        self.files = {'first' : {'exp_1' : 'results_1.pkl',
                                 'exp_2' : 'results_2.pkl'},
                      'second' : {'exp_5': 'results_5.pkl'}
                      }
        curr_dir, base = os.path.split(os.path.abspath(__file__))
        for categ, observs in self.files.items():
            for exp, file_ in observs.items():
                self.files[categ][exp] = os.path.join(curr_dir, file_)

        # keep paths of all files generated here
        self.tmp_paths = []

        # make pickles
        path = os.path.join(curr_dir, self.files['first']['exp_1'])
        analysis_cmn.make_and_pickle(file_=path)
        self.tmp_paths.append(path)
        path = os.path.join(curr_dir, self.files['first']['exp_2'])
        analysis_cmn.make_and_pickle(file_=path, data=1)
        self.tmp_paths.append(path)
        path = os.path.join(curr_dir, self.files['second']['exp_5'])
        analysis_cmn.make_and_pickle(file_=path)
        self.tmp_paths.append(path)

    def testGetSingle(self):
        """
        Tests getSingle()
        """
        
        # instantiate
        pickled = Pickled(files=self.files)
        exp_2_direct = pickle.load(open(self.files['first']['exp_2']))

        # test exp 2 pickle 
        exp_2 = pickled.getSingle(category='first', identifier='exp_2')
        np_test.assert_almost_equal(exp_2.width, exp_2_direct.width)
        np_test.assert_almost_equal(exp_2.widthVector.theta, 
                                    exp_2_direct.widthVector.theta)
        np_test.assert_almost_equal(exp_2.regionDensity.mean, 
                                    exp_2_direct.regionDensity.mean)

        # simple comparison with exp 1 pickle
        exp_1 = pickled.getSingle(category='first', identifier='exp_1')
        np_test.assert_almost_equal(exp_1.regionDensity.mean+1,
                                    exp_2.regionDensity.mean)
        np_test.assert_almost_equal(exp_1.regionDensity.std,
                                    exp_2.regionDensity.std)


    def testIdentifiers(self):
        """
        Tests identifiers(), which is currently (r920) implemented in the super
        (MultiData).
        """

        multi = MultiData(files=self.files)
        np_test.assert_equal(
            set([foo for foo in multi.identifiers(category=None)]),
            set(['exp_1', 'exp_2', 'exp_5']))
        np_test.assert_equal(
            set([foo for foo in multi.identifiers(category='first')]),
            set(['exp_1', 'exp_2']))
        np_test.assert_equal(
            [foo for foo in multi.identifiers(category=['second', 'nonexist'])],
            ['exp_5'])
        np_test.assert_equal(
            [foo for foo in multi.identifiers(category='nonexisting')], [])

    def testData(self):
        """
        Tests self.data, which is currently (r680) implemented in the super
        (MultiData).
        """

        # test one experiment (exp 2)
        pickled = Pickled(files=self.files)
        exp_2_direct = pickle.load(open(self.files['first']['exp_2']))
        for obj, categ, ident in pickled.data(category='first', 
                                              identifier='exp_2'):
            np_test.assert_equal(categ, 'first')
            np_test.assert_equal(ident, 'exp_2')
            np_test.assert_equal(obj.regions.ids, range(1,10))
            np_test.assert_almost_equal(obj.width, scene_cmn.cleft_layers_width)

        # test one group
        pickled = Pickled(files=self.files)
        exp_2_direct = pickle.load(open(self.files['first']['exp_2']))
        identifiers = set() 
        for obj, categ, ident in pickled.data(category='first'):
            identifiers.add(ident)
            np_test.assert_equal(categ, 'first')
            np_test.assert_equal(obj.regions.ids, range(1,10))
            np_test.assert_almost_equal(obj.width, scene_cmn.cleft_layers_width)
        np_test.assert_equal(identifiers, set(['exp_1', 'exp_2']))

        # test all
        pickled = Pickled(files=self.files)
        exp_2_direct = pickle.load(open(self.files['first']['exp_2']))
        groups = set()
        identifiers = set() 
        for obj, categ, ident in pickled.data():
            groups.add(categ)
            identifiers.add(ident)
            np_test.assert_equal(obj.regions.ids, range(1,10))
            np_test.assert_almost_equal(obj.width, scene_cmn.cleft_layers_width)
        np_test.assert_equal(groups, set(['first', 'second']))
        np_test.assert_equal(identifiers, set(['exp_1', 'exp_2', 'exp_5']))

        # test category order
        pickled = Pickled(files=self.files)
        categs = [categ for obj, categ, ident 
                  in pickled.data(category=['first', 'second'])]
        np_test.assert_equal(categs, ['first', 'first', 'second'])
        categs = [categ for obj, categ, ident 
                  in pickled.data(category=['second', 'first'])]
        np_test.assert_equal(categs, ['second', 'first', 'first'])

        # test experiment (identifier) order
        pickled = Pickled(files=self.files)
        idents = [ident for obj, categ, ident 
                  in pickled.data(category='first', 
                                  identifier=['exp_1', 'exp_2'])]
        np_test.assert_equal(idents, ['exp_1', 'exp_2'])
        idents = [ident for obj, categ, ident 
                  in pickled.data(category='first', 
                                  identifier=['exp_2', 'exp_1'])]
        np_test.assert_equal(idents, ['exp_2', 'exp_1'])

    def testReadPropertiesGen(self):
        """
        Tests self.readPropertiesGen(), which is currently (r680) implemented 
        in the super (MultiData)
        """

        # initialize
        pickled = Pickled(files=self.files)
        properties = ['regions.ids', 'width', 'widthVector.thetaDeg', 
                      'regionDensity.mean', 'regionDensity.volume']
        indexed = ['regionDensity.mean', 'regionDensity.volume']

        # run readPropertiesGen
        for observ, obj, category, identifier in pickled.readPropertiesGen(
            properties=properties, index='regions.ids', indexed=indexed, 
            deep='last'):

            # test at each iteration
            np_test.assert_almost_equal(obj.width, scene_cmn.cleft_layers_width)
            if (identifier == 'exp_1') or (identifier == 'exp_5'): 
                np_test.assert_almost_equal(
                    obj.regionDensity.mean[obj.regions.ids], 
                    scene_cmn.cleft_layers_density_mean)
            elif (identifier == 'exp_2'):
                np_test.assert_almost_equal(
                    obj.regionDensity.mean[obj.regions.ids]-1, 
                    scene_cmn.cleft_layers_density_mean)

        # test final results
        exp_2 = observ.getExperiment(identifier='exp_2')
        np_test.assert_almost_equal(exp_2.width, scene_cmn.cleft_layers_width)
        np_test.assert_almost_equal(exp_2.mean-1, 
                                    scene_cmn.cleft_layers_density_mean)
        exp_5 = observ.getExperiment(identifier='exp_5')
        np_test.assert_almost_equal(exp_5.width, scene_cmn.cleft_layers_width)
        np_test.assert_almost_equal(exp_5.mean, 
                                    scene_cmn.cleft_layers_density_mean)

        # properties
        np_test.assert_equal(
            observ.properties, 
            set(['identifiers', 'ids', 'width', 'thetaDeg', 'mean', 'volume']))

        # test order
        categs = [categ for observ, obj, categ, ident 
                  in pickled.readPropertiesGen(
                properties=properties, index='regions.ids', indexed=indexed, 
                deep='last', category=['first', 'second'])]
        np_test.assert_equal(categs, ['first', 'first', 'second'])

        categs = [categ for observ, obj, categ, ident 
                  in pickled.readPropertiesGen(
                properties=properties, index='regions.ids', indexed=indexed, 
                deep='last', category=['second', 'first'])]
        np_test.assert_equal(categs, ['second', 'first', 'first'])

        idents = [ident for observ, obj, categ, ident 
                  in pickled.readPropertiesGen(
                properties=properties, index='regions.ids', indexed=indexed, 
                deep='last', category='first', identifier=['exp_1', 'exp_2'])]
        np_test.assert_equal(idents, ['exp_1', 'exp_2'])
        np_test.assert_equal(observ.identifiers, ['exp_1', 'exp_2'])
        np_test.assert_equal(observ.mean, 
                             [scene_cmn.cleft_layers_density_mean,
                              scene_cmn.cleft_layers_density_mean+1])

        idents = [ident for observ, obj, categ, ident 
                  in pickled.readPropertiesGen(
                properties=properties, index='regions.ids', indexed=indexed, 
                deep='last', category='first', identifier=['exp_2', 'exp_1'])]
        np_test.assert_equal(idents, ['exp_2', 'exp_1'])
        np_test.assert_equal(observ.identifiers, ['exp_2', 'exp_1'])
        np_test.assert_equal(observ.mean, 
                             [scene_cmn.cleft_layers_density_mean+1,
                              scene_cmn.cleft_layers_density_mean])

    def testReadPropertiesGenDict(self):
        """
        Tests self.readPropertiesGen() where arg properties is a dict. 
        Currently (r680) implemented in the super (MultiData)
        """

        # initialize
        pickled = Pickled(files=self.files)
        properties = {
            'regions.ids' : 'ids', 'width' : 'width_x', 
            'widthVector.thetaDeg' : 'thetaDeg_x', 
            'regionDensity.mean' : 'mean_x', 'regionDensity.volume':'volume_x'}
        indexed = ['regionDensity.mean', 'regionDensity.volume']

        # run readPropertiesGen
        for observ, obj, category, identifier in pickled.readPropertiesGen(
            properties=properties, index='regions.ids', indexed=indexed, 
            deep='last'):

            # test at each iteration
            np_test.assert_almost_equal(obj.width, scene_cmn.cleft_layers_width)
            if (identifier == 'exp_1') or (identifier == 'exp_5'): 
                np_test.assert_almost_equal(
                    obj.regionDensity.mean[obj.regions.ids], 
                    scene_cmn.cleft_layers_density_mean)
            elif (identifier == 'exp_2'):
                np_test.assert_almost_equal(
                    obj.regionDensity.mean[obj.regions.ids]-1, 
                    scene_cmn.cleft_layers_density_mean)

        # test final results
        exp_2 = observ.getExperiment(identifier='exp_2')
        np_test.assert_almost_equal(exp_2.width_x, scene_cmn.cleft_layers_width)
        np_test.assert_almost_equal(exp_2.mean_x - 1, 
                                    scene_cmn.cleft_layers_density_mean)
        exp_5 = observ.getExperiment(identifier='exp_5')
        np_test.assert_almost_equal(exp_5.width_x, scene_cmn.cleft_layers_width)
        np_test.assert_almost_equal(exp_5.mean_x, 
                                    scene_cmn.cleft_layers_density_mean)

        # test order
        categs = [categ for observ, obj, categ, ident 
                  in pickled.readPropertiesGen(
                properties=properties, index='regions.ids', indexed=indexed, 
                deep='last', category=['first', 'second'])]
        np_test.assert_equal(categs, ['first', 'first', 'second'])

        categs = [categ for observ, obj, categ, ident 
                  in pickled.readPropertiesGen(
                properties=properties, index='regions.ids', indexed=indexed, 
                deep='last', category=['second', 'first'])]
        np_test.assert_equal(categs, ['second', 'first', 'first'])

        idents = [ident for observ, obj, categ, ident 
                  in pickled.readPropertiesGen(
                properties=properties, index='regions.ids', indexed=indexed, 
                deep='last', category='first', identifier=['exp_1', 'exp_2'])]
        np_test.assert_equal(idents, ['exp_1', 'exp_2'])
        np_test.assert_equal(observ.identifiers, ['exp_1', 'exp_2'])
        np_test.assert_equal(observ.mean_x, 
                             [scene_cmn.cleft_layers_density_mean,
                              scene_cmn.cleft_layers_density_mean+1])

        idents = [ident for observ, obj, categ, ident 
                  in pickled.readPropertiesGen(
                properties=properties, index='regions.ids', indexed=indexed, 
                deep='last', category='first', identifier=['exp_2', 'exp_1'])]
        np_test.assert_equal(idents, ['exp_2', 'exp_1'])
        np_test.assert_equal(observ.identifiers, ['exp_2', 'exp_1'])
        np_test.assert_equal(observ.mean_x, 
                             [scene_cmn.cleft_layers_density_mean+1,
                              scene_cmn.cleft_layers_density_mean])

    def tearDown(self):
        """
        Remove temporary files
        """

        for path in self.tmp_paths:
            try:
                os.remove(path)
            except OSError:
                print("Tests fine but could not remove " + str(path))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPickled)
    unittest.TextTestRunner(verbosity=2).run(suite)
