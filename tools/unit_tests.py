#!/usr/bin/env python
# Unit tests for the pyORAC suite

import numpy as np
import os.path
import unittest
from argparse import ArgumentParser
from datetime import datetime, timedelta

import pyorac.arguments as ar
import pyorac.local_defaults as ld
from pyorac.definitions import FileName, FileMissing, SETTINGS
from pyorac.lut import OracLut
from pyorac.regression_tests import REGRESSION_TESTS


class TestDriverCreation(unittest.TestCase):
    """Check creation of driver files"""

    def test_preproc(self):
        """Make a preprocessor driver file"""
        from pyorac.drivers import build_preproc_driver

        pars = ArgumentParser()
        pars.add_argument('target')
        ar.args_common(pars)
        ar.args_cc4cl(pars)
        ar.args_preproc(pars)
        args = pars.parse_args([
            ld.REGRESS_IN_DIR + "/" + REGRESSION_TESTS["DAYMYD"][0],
            "--revision", "1"
        ])
        args = ar.check_args_common(args)
        args = ar.check_args_preproc(args)
        self.assertMultiLineEqual(build_preproc_driver(args),
                                  """MODIS
/data/povey/data/testinput/MYD021KM.A2008172.0405.061.2018034142316.hdf
/data/povey/data/testinput/MYD03.A2008172.0405.061.2018034002648.hdf
/data/povey/data/LSM/global_lsm.nc
/data/povey/data/ecmwf/ECMWF_ERA5_20080620_00_0.5.nc
/data/povey/data/rtcoef_rttov13
/data/povey/data/rtcoef_rttov13/emis_data
/data/povey/data/ice_snow/NISE_SSMIF13_20080620.HDFEOS
/data/povey/data/albedo/MCD43C3.A2008169.005.2008192032521.hdf
/data/povey/data/albedo/MCD43C1.A2008169.005.2008192031349.hdf
/data/povey/data/emissivity/global_emis_inf10_monthFilled_MYD11C3.A2008153.041.nc
1.38888889
1.38888889
/data/povey/data/testinput
0
0
0
0
4.8.1
CF-1.4
UoOx
ORAC
adam.povey@physics.ox.ac.uk
http://github.com/ORAC-CC/orac
R1
doi:10.5194/amt-5-1889-2012
n/a
n/a
aerosol; cloud; optimal estimation
n/a
ESACCI
https://github.com/ORAC-CC/orac/blob/master/COPYING
n/a
{:%Y%m%d%H%M%S}
/data/povey/data/AATSR_VIS_DRIFT_V03-00.DAT
1


False
3
False
-
True
True
13.0.0
n/a
2.17.1
ECMWF_TIME_INT_METHOD=2
ECMWF_PATH_2=/data/povey/data/ecmwf/ECMWF_ERA5_20080620_06_0.5.nc
ECMWF_PATH2_2=
ECMWF_PATH3_2=
USE_ECMWF_SNOW_AND_ICE=False
USE_MODIS_EMIS_IN_RTTOV=False
ECMWF_NLEVELS=137
USE_L1_LAND_MASK=False
USE_OCCCI=False
OCCCI_PATH=
DISABLE_SNOW_ICE_CORR=False
DO_CLOUD_EMIS=False
DO_IRONLY=False
DO_CLDTYPE=True
USE_CAMEL_EMIS=False
USE_SWANSEA_CLIMATOLOGY=False
N_CHANNELS=6
CHANNEL_IDS=1,2,6,20,31,32
PRODUCT_NAME=L2-CLOUD-CLD""".format(datetime.now()))

    def test_main(self):
        """Make a main processor driver file"""
        from pyorac.drivers import build_main_driver

        pars = ArgumentParser()
        pars.add_argument('target')
        ar.args_common(pars)
        ar.args_cc4cl(pars)
        ar.args_main(pars)
        args = pars.parse_args([
            ld.REGRESS_IN_DIR + "/ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_"
            "200806200405_R1.config.nc",
            "--phase", "WAT"
        ])
        args = ar.check_args_common(args)
        args = ar.check_args_main(args)
        self.assertMultiLineEqual(build_main_driver(args),
                                  """# ORAC New Driver File
Ctrl%FID%Data_Dir           = "/data/povey/data/testinput"
Ctrl%FID%Filename           = "ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_200806200405_R1"
Ctrl%FID%Out_Dir            = "/data/povey/data/testinput"
Ctrl%FID%SAD_Dir            = "/network/aopp/apres/ORAC_LUTS/modis/AQUA/WAT"
Ctrl%InstName               = "MODIS-AQUA"
Ctrl%Ind%NAvail             = 6
Ctrl%Ind%Channel_Proc_Flag  = 1,1,1,1,1,1
Ctrl%LUTClass               = "WAT"
Ctrl%Process_Cloudy_Only    = False
Ctrl%Process_Aerosol_Only   = False
Ctrl%Verbose                = False
Ctrl%RS%Use_Full_BRDF       = True
Ctrl%NTypes_To_Process      = 11
Ctrl%Types_To_Process(1:11) = CLEAR_TYPE,SWITCHED_TO_WATER_TYPE,FOG_TYPE,WATER_TYPE,SUPERCOOLED_TYPE,SWITCHED_TO_ICE_TYPE,OPAQUE_ICE_TYPE,CIRRUS_TYPE,OVERLAP_TYPE,PROB_OPAQUE_ICE_TYPE,PROB_CLEAR_TYPE""")

    def test_postproc_cc4cl(self):
        """Make a preprocessor driver file"""
        from pyorac.drivers import build_postproc_driver

        pars = ArgumentParser()
        pars.add_argument('target')
        ar.args_common(pars)
        ar.args_cc4cl(pars)
        ar.args_postproc(pars)
        args = pars.parse_args([
            ld.REGRESS_IN_DIR + "/ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_"
            "200806200405_R1WAT.primary.nc",
            "--phases", "WAT", "ICE"
        ])
        args = ar.check_args_common(args)
        args = ar.check_args_postproc(args)
        files = [args.File.root_name() + phs + ".primary.nc"
                 for phs in args.phases]
        self.assertMultiLineEqual(build_postproc_driver(args, files),
                                  """False
ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_200806200405_R1WAT.primary.nc
ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_200806200405_R1ICE.primary.nc
ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_200806200405_R1WAT.secondary.nc
ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_200806200405_R1ICE.secondary.nc
ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_200806200405_R1WAT.primary.nc
ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_200806200405_R1WAT.secondary.nc
False
COST_THRESH=0.0
NORM_PROB_THRESH=0.0
OUTPUT_OPTICAL_PROPS_AT_NIGHT=True
VERBOSE=False
USE_CHUNKING=False
USE_NETCDF_COMPRESSION=False
USE_NEW_BAYESIAN_SELECTION=False""")

    def test_postproc_multi(self):
        """Make a preprocessor driver file"""
        from pyorac.drivers import build_postproc_driver

        pars = ArgumentParser()
        pars.add_argument('target')
        ar.args_common(pars)
        ar.args_cc4cl(pars)
        ar.args_postproc(pars)
        args = pars.parse_args([
            ld.REGRESS_IN_DIR + "/ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_"
            "200806200405_R1WAT.primary.nc",
            "--phases", "WAT", "ICE", "WAT_ICE"
        ])
        args = ar.check_args_common(args)
        args = ar.check_args_postproc(args)
        files = [args.File.root_name() + phs + ".primary.nc"
                 for phs in args.phases]
        self.assertMultiLineEqual(build_postproc_driver(args, files),
                                  """True
ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_200806200405_R1WAT.primary.nc
ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_200806200405_R1ICE.primary.nc
ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_200806200405_R1WAT.secondary.nc
ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_200806200405_R1ICE.secondary.nc
ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_200806200405_R1WAT.primary.nc
ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_200806200405_R1WAT.secondary.nc
False
COST_THRESH=0.0
NORM_PROB_THRESH=0.0
OUTPUT_OPTICAL_PROPS_AT_NIGHT=True
VERBOSE=False
USE_CHUNKING=False
USE_NETCDF_COMPRESSION=False
USE_NEW_BAYESIAN_SELECTION=True
ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_200806200405_R1WAT_ICE.primary.nc
ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_200806200405_R1WAT_ICE.secondary.nc""")


class TestFileName(unittest.TestCase):
    """Check parsing of L1 satellite filenames"""

    def test_aatsr(self):
        """Parsing of AATSR L1 filenames"""

        f = FileName(ld.REGRESS_IN_DIR, "ATS_TOA_1PRUPA20080620_002337_"
                     "000065272069_00345_32964_0666.N1")
        self.assertEqual(f.sensor, 'AATSR')
        self.assertEqual(f.platform, 'Envisat')
        self.assertEqual(f.inst, 'AATSR')
        self.assertEqual(f.time, datetime(2008, 6, 20, 0, 23, 37))
        self.assertEqual(f.dur, timedelta(seconds=652))
        self.assertEqual(f.geo, "ATS_TOA_1PRUPA20080620_002337_"
                        "000065272069_00345_32964_0666.N1")
        self.assertEqual(f.root_name(1, 'ORAC', 'ESACCI', 'L2-CLOUD-CLD'),
                        "ESACCI-L2-CLOUD-CLD-AATSR_ORAC_Envisat_200806200023_R1")

    def test_modisterra(self):
        """Parsing of MODIS TERRA filenames"""

        f = FileName(ld.REGRESS_IN_DIR, "MOD021KM.A2008172.0115.061."
                     "2017256012659.hdf")
        self.assertEqual(f.sensor, 'MODIS')
        self.assertEqual(f.platform, 'TERRA')
        self.assertEqual(f.inst, 'MODIS-TERRA')
        self.assertEqual(f.time, datetime(2008, 6, 20, 1, 15, 0))
        self.assertEqual(f.dur, timedelta(seconds=300))
        self.assertEqual(f.geo, "MOD03.A2008172.0115.061.*hdf")
        self.assertEqual(f.root_name(1, 'ORAC', 'ESACCI', 'L2-CLOUD-CLD'),
                        "ESACCI-L2-CLOUD-CLD-MODIS_ORAC_TERRA_200806200115_R1")

    def test_modisaqua(self):
        """Parsing of MODIS AQUA filenames"""

        f = FileName(ld.REGRESS_IN_DIR, "MYD021KM.A2008172.0405.061."
                     "2018034142316.hdf")
        self.assertEqual(f.sensor, 'MODIS')
        self.assertEqual(f.platform, 'AQUA')
        self.assertEqual(f.inst, 'MODIS-AQUA')
        self.assertEqual(f.time, datetime(2008, 6, 20, 4, 5, 0))
        self.assertEqual(f.dur, timedelta(seconds=300))
        self.assertEqual(f.geo, "MYD03.A2008172.0405.061.*hdf")
        self.assertEqual(f.root_name(1, 'ORAC', 'ESACCI', 'L2-CLOUD-CLD'),
                        "ESACCI-L2-CLOUD-CLD-MODIS_ORAC_AQUA_200806200405_R1")

    def test_avhrr(self):
        """Parsing of AVHRR reprocessed filenames"""

        f = FileName(ld.REGRESS_IN_DIR, "noaa18_20080620_0050_99999_satproj_"
                     "00000_13111_avhrr.h5")
        self.assertEqual(f.sensor, 'AVHRR')
        self.assertEqual(f.platform, 'noaa18')
        self.assertEqual(f.inst, 'AVHRR-NOAA18')
        self.assertEqual(f.time, datetime(2008, 6, 20, 0, 50, 0))
        self.assertEqual(f.dur, timedelta(seconds=6555))
        self.assertEqual(f.geo, "noaa18_20080620_0050_99999_satproj_00000_"
                        "13111_sunsatangles.h5")
        self.assertEqual(f.root_name(1, 'ORAC', 'ESACCI', 'L2-CLOUD-CLD'),
                        "ESACCI-L2-CLOUD-CLD-AVHRR_ORAC_noaa18_200806200050_R1")

    def test_seviri(self):
        """Parsing of SEVIRI NAT filenames"""

        f = FileName(ld.REGRESS_IN_DIR, "MSG3-SEVI-MSG15-0100-NA-20170620055740."
                     "316000000Z-20170620055758-1314114-37.nat")
        self.assertEqual(f.sensor, 'SEVIRI')
        self.assertEqual(f.platform, 'MSG3')
        self.assertEqual(f.inst, 'SEVIRI-MSG3')
        self.assertEqual(f.time, datetime(2017, 6, 20, 5, 57, 40))
        self.assertEqual(f.dur, timedelta(seconds=900))
        self.assertEqual(f.geo, "MSG3-SEVI-MSG15-0100-NA-20170620055740."
                                  "316000000Z-20170620055758-1314114-37.nat")
        self.assertEqual(f.root_name(1, 'ORAC', 'ESACCI', 'L2-CLOUD-CLD'),
                        "ESACCI-L2-CLOUD-CLD-SEVIRI_ORAC_MSG3_201706200557_R1")

    def test_slstra(self):
        """Parsing of SLSTR-A filenames"""

        f = FileName(ld.REGRESS_IN_DIR, "S3A_SL_1_RBT____20181221T133848_"
                     "20181221T134148_20181222T201511_0179_039_209_5760_"
                     "LN2_O_NT_003.SEN3")
        self.assertEqual(f.sensor, 'SLSTR')
        self.assertEqual(f.platform, 'Sentinel3a')
        self.assertEqual(f.inst, 'SLSTR-Sentinel-3a')
        self.assertEqual(f.time, datetime(2018, 12, 21, 13, 39, 0))
        self.assertEqual(f.dur, timedelta(seconds=179))
        self.assertEqual(f.geo, "S3A_SL_1_RBT____20181221T133848_"
                        "20181221T134148_20181222T201511_0179_039_209_5760_"
                        "LN2_O_NT_003.SEN3/geodetic_in.nc")
        self.assertEqual(f.root_name(1, 'ORAC', 'ESACCI', 'L2-CLOUD-CLD'),
                        "ESACCI-L2-CLOUD-CLD-SLSTR_ORAC_Sentinel3a_"
                        "201812211339_14812_R1")

    def test_slstrb(self):
        """Parsing of SLSTR-A filenames"""

        f = FileName(ld.REGRESS_IN_DIR, "S3B_SL_1_RBT____20181221T002724_"
                     "20181221T003024_20181222T080855_0179_020_059_3060_"
                     "LN2_O_NT_003.SEN3")
        self.assertEqual(f.sensor, 'SLSTR')
        self.assertEqual(f.platform, 'Sentinel3b')
        self.assertEqual(f.inst, 'SLSTR-Sentinel-3b')
        self.assertEqual(f.time, datetime(2018, 12, 21, 0, 27, 0))
        self.assertEqual(f.dur, timedelta(seconds=179))
        self.assertEqual(f.geo, "S3B_SL_1_RBT____20181221T002724_"
                    "20181221T003024_20181222T080855_0179_020_059_3060_"
                        "LN2_O_NT_003.SEN3/geodetic_in.nc")
        self.assertEqual(f.root_name(1, 'ORAC', 'ESACCI', 'L2-CLOUD-CLD'),
                        "ESACCI-L2-CLOUD-CLD-SLSTR_ORAC_Sentinel3b_"
                        "201812210027_03411_R1")

    def test_preproc(self):
        """Parsing of ORAC preprocessor filenames"""

        f = FileName("TESTING-REGRESS-AVHRR_ORAC_noaa18_200806200050_"
                     "R1898.config.nc")
        self.assertEqual(f.sensor, 'AVHRR')
        self.assertEqual(f.platform, 'noaa18')
        self.assertEqual(f.inst, 'AVHRR-NOAA18')
        self.assertEqual(f.time, datetime(2008, 6, 20, 0, 50, 0))
        self.assertIsNone(f.dur)
        self.assertIsNone(f.geo)
        self.assertEqual(f.oractype, 'config')
        self.assertEqual(f.root_name(1, 'ORAC', 'ESACCI', 'L2-CLOUD-CLD'),
                        "ESACCI-L2-CLOUD-CLD-AVHRR_ORAC_noaa18_200806200050_R1")

    def test_mainproc(self):
        """Parsing of ORAC main processor filenames"""

        f = FileName("TESTING-REGRESS-AATSR_ORAC_Envisat_200806200023_"
                     "R1898WAT.primary.nc")
        self.assertEqual(f.sensor, 'AATSR')
        self.assertEqual(f.platform, 'Envisat')
        self.assertEqual(f.inst, 'AATSR')
        self.assertEqual(f.time, datetime(2008, 6, 20, 0, 23, 0))
        self.assertIsNone(f.dur)
        self.assertIsNone(f.geo)
        self.assertEqual(f.oractype, 'primary')
        self.assertEqual(f.root_name(1, 'ORAC', 'ESACCI', 'L2-CLOUD-CLD'),
                        "ESACCI-L2-CLOUD-CLD-AATSR_ORAC_Envisat_200806200023_R1")


class TestLocalDefaults(unittest.TestCase):
    """Check your local defaults point to actual things"""

    def test_that_required_variables_defined(self):
        """Ensure all the required variables are defined in local_defaults"""

        self.assertIsInstance(getattr(ld, "ORAC_DIR"), str)
        self.assertIsInstance(getattr(ld, "REGRESS_IN_DIR"), str)
        self.assertIsInstance(getattr(ld, "REGRESS_OUT_DIR"), str)
        self.assertIsInstance(getattr(ld, "ORAC_LIB"), str)
        self.assertIsInstance(getattr(ld, "SAD_DIRS"), tuple)
        self.assertIsInstance(getattr(ld, "NWP_FLAG"), int)
        self.assertIsInstance(getattr(ld, "AUXILIARIES"), dict)
        self.assertIsInstance(getattr(ld, "GLOBAL_ATTRIBUTES"), dict)
        self.assertIsInstance(getattr(ld, "LOG_DIR"), str)
        self.assertIsInstance(getattr(ld, "PRE_DIR"), str)
        self.assertIsInstance(getattr(ld, "EXTRA_LINES"), dict)
        self.assertIsInstance(getattr(ld, "RETRIEVAL_SETTINGS"), dict)
        self.assertIsInstance(getattr(ld, "CHANNELS"), dict)
        self.assertIsInstance(getattr(ld, "ATTS_TO_IGNORE"), tuple)
        self.assertIsInstance(getattr(ld, "VARS_TO_ACCEPT"), tuple)
        self.assertIsInstance(getattr(ld, "RTOL"), float)
        self.assertIsInstance(getattr(ld, "ATOL"), float)
        self.assertIsInstance(getattr(ld, "WARN_FILT"), dict)
        self.assertIsInstance(getattr(ld, "BATCH_VALUES"), dict)
        self.assertIsInstance(getattr(ld, "BATCH_SCRIPT"), str)
        self.assertIsInstance(getattr(ld, "DIR_PERMISSIONS"), int)

    def test_that_directories_exist(self):
        """Ensure all local_defaults directories actually exist"""

        self.assertTrue(os.path.isdir(ld.ORAC_DIR))
        self.assertTrue(os.path.isdir(ld.REGRESS_IN_DIR))
        self.assertTrue(os.path.isdir(ld.REGRESS_OUT_DIR))
        self.assertTrue(os.path.isfile(ld.ORAC_LIB))
        for fdr in ld.SAD_DIRS:
            self.assertTrue(os.path.isdir(fdr))
        for val in ld.AUXILIARIES.values():
            if val.endswith("file"):
                self.assertTrue(os.path.isfile(val))
            else:
                self.assertTrue(os.path.isdir(datetime(2008, 1, 1).strftime(fdr)))

    def test_retrieval_settings_all_parse(self):
        """Ensure all default retrievals can be parsed"""

        pars = ArgumentParser()
        pars.add_argument("--sub_dir", default="")
        ar.args_main(pars)
        for key, value in ld.RETRIEVAL_SETTINGS.items():
            for setting in value:
                args = pars.parse_args(setting.split(" "))
                self.assertIsInstance(args.approach, str)
                self.assertIsInstance(args.phase, str)
                self.assertIsInstance(args.use_channels, list)
                self.assertIsInstance(args.use_channels[0], int)

    def test_define_channels_for_every_test_file(self):
        """Ensure default channels are defined for all regression tests"""

        for filename, _, _ in REGRESSION_TESTS.values():
            inst = FileName(ld.REGRESS_IN_DIR, filename)
            self.assertIn(inst.sensor, ld.CHANNELS)

    def test_getting_repository_version(self):
        """Ensure we can fetch the repository version number"""
        from pyorac.util import get_repository_revision

        self.assertIsInstance(get_repository_revision(), int)


class TestLut(unittest.TestCase):
    """Check the look-up table reading and interpolation"""
    QUICK = True

    def test_planck_fct_derivative(self):
        """Ensure derivate of OracLut.temp2rad is accurate"""
        from glob import iglob

        delta = 0.1
        temperatures = np.arange(150, 350, delta)
        files = []
        for fdr in ld.SAD_DIRS:
            files.extend(iglob(fdr + "/*nc"))
            files.extend(iglob(fdr + "/*_RD_Ch*sad"))

        for filename in files:
            lut = OracLut(filename, method='linear')
            if not np.any(lut.thermal):
                continue
            values = np.array([lut.temp2rad(temp) for temp in temperatures])
            print(filename)
            self.assertTrue(np.allclose(
                values[:-1,1], np.diff(values[:,0], axis=0) / delta, atol=3e-3
            ))
            if self.QUICK:
                break


class TestMappable(unittest.TestCase):
    """Check the map plotting methods"""

    def test_simple_corners_are_the_same(self):
        """Ensure a simple rectangle extrapolates correctly"""
        from numpy import array
        from pyorac.mappable import grid_cell_corners, linear_cell_corners

        lat = array([[1., 1.], [2., 2.]])
        lon = array([[10., 12.], [10., 12.]])

        gridlon, gridlat = grid_cell_corners(lat[:,0], lon[0])
        linlat, linlon = linear_cell_corners(lat, lon)
        self.assertListEqual(gridlon.tolist(), linlon.tolist())
        self.assertListEqual(gridlat.tolist(), linlat.tolist())
        self.assertTrue(gridlat[0,0] == 0.5)
        self.assertTrue(gridlat[2,2] == 2.5)
        self.assertTrue(gridlon[0,0] == 9.)
        self.assertTrue(gridlon[2,2] == 13.)


class TestSettings(unittest.TestCase):
    """Check your local defaults works"""

    def test_settings_contains_basic_types(self):
        """Ensure the fundamental particle types are available"""

        self.assertIn("liquid-water", SETTINGS)
        self.assertIn("water-ice", SETTINGS)
        self.assertIn("WAT", SETTINGS)
        self.assertIn("ICE", SETTINGS)

    def test_text_tables_exist(self):
        """Ensure text liquid water tables exist for every sensor"""

        for fname, _, _ in REGRESSION_TESTS.values():
            try:
                inst = FileName(ld.REGRESS_IN_DIR, fname)
                self.assertIsNotNone(SETTINGS["WAT"].sad_dir(ld.SAD_DIRS, inst))
            except FileMissing:
                self.assertIsNone(fname)

    def test_nc_tables_exist(self):
        """Ensure netCDF liquid water tables exist for every sensor"""

        for fname, _, _ in REGRESSION_TESTS.values():
            try:
                inst = FileName(ld.REGRESS_IN_DIR, fname)
                self.assertIsNotNone(SETTINGS["WAT11"].sad_dir(ld.SAD_DIRS, inst))
            except FileMissing:
                self.assertIsNone(fname)


if __name__ == "__main__":
    unittest.TestCase.maxDiff = None
    unittest.main()
