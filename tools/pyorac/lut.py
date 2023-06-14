"""Routines for opening and manipulating ORAC look-up tables

Classes:
RearrangedRegularGridInterpolator
    Hack of scipy.interpolate.RegularGridInterpolator that allows
    us to change the order of dimensions in __call__. This let's
    us use a single call for both text and NCDF tables, which have
    different orders for the dimensions.
OracLut
    Container to open and interpolate the ORAC look-up tables from
    either text or netCDF format.

Functions:
read_orac_chan_file
    Fetches the contents of an ORAC SAD Channel file
stack_orac_chan_file
    Fetches SAD Channel data for a set of channels
read_orac_text_lut
    Fetches the contents of an ORAC SAD LUT file
stack_orac_text_lut
    Fetches SAD LUT data for a set of channels for files with 1 table
stack_orac_text_lut_pair
    Fetches SAD LUT data for a set of channels for files with 2 tables

Notes:
- There is no check that all text LUTs have the same axes, because
  they don't strictly need to with RegularGridInterpolator.
- As done in the main code, we assume the identity of the axes in the
  LUT files. Calls to text tables are manupulated to look like NCDF.

To Do:
- Copy over the Fortran LUT interpolation routines.
"""
import numpy as np
import os.path

from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator


class RearrangedRegularGridInterpolator(RegularGridInterpolator):
    """Hack of scipy.interpolate.RegularGridInterpolator

    Adds the ability to change the order in which grid dimensions
    are referenced by __call__
    """
    def __init__(self, *args, **kwargs):
        self.order = args[-1]
        if len(self.order) != len(args[0]):
            raise ValueError("order must be of same length as grid")
        super().__init__(*args[:-1], **kwargs)

    def __call__(self, *args, **kwargs):
        rearranged_args = tuple(args[0][i] for i in self.order)
        return super().__call__(rearranged_args, *args[1:], **kwargs)


class OracLut(object):
    """
    Container for reading and interpolating ORAC look-up tables

    Both text and netCDF versions are supported. Each table is opened
    an instance of scipy.interpolate.RegularGridInterpolator but the
    routines src/int_lut* and set_crp_* from ORAC are also implemented
    for exact replication.
    """

    def __init__(self, filename, method='cubic',
                 t_dv_from_t_0d=True, check_consistency=False):
        """
        Args:
        filename: name of look-up table to open

        Kwargs:
        method: interpolation method to use. Default cubic.
        t_dv_from_t_0d: ignore the diffuse-to-view transmission table
        check_consistency: ensure that the number of channels is consistent
            across read operations (mostly important for text tables)
        """

        self.filename = filename
        self.check = check_consistency
        self._particle = None
        self._inst = None

        if self.filename.endswith("nc"):
            self._open_ncdf_tables(method)
            self.chan_labels = None
        else:
            self._open_text_chan_files()
            self._open_text_tables(method)

        if t_dv_from_t_0d:
            self.t_dv = self.t_0d

    @staticmethod
    def from_description(sad_dirs, particle, inst, **kwargs):
        """Initialise from pyorac settings objects

        Args:
        sad_dirs: list of directories to search for these LUTs
        particle: instance of definitions.ParticleType for the desired tables
        inst: instance of definitions.FileName for the desired instrument
        """

        fdr = particle.sad_dir(sad_dirs, inst)
        filename = os.path.join(fdr, particle.sad_filename(inst))
        self = OracLut(filename, **kwargs)
        self._particle = particle
        self._inst = inst
        return self

    def _open_text_chan_files(self):
        """Open ORAC channel descriptions for a set of channels"""
        import re
        from glob import iglob

        regex = re.compile("(Ch[0-9ab]+)\.sad")
        self.chan_labels = [regex.search(f).group(1) for f in iglob(self.filename)]
        self.nch = len(self.chan_labels)
        self.channels = np.empty(self.nch, dtype=int)
        self.solar = np.zeros(self.nch, dtype=bool)
        self.thermal = np.zeros(self.nch, dtype=bool)
        f0 = []
        f1 = []
        b1 = []
        b2 = []
        t1 = []
        t2 = []

        fdr, root = os.path.split(self.filename)
        parts = os.path.basename(root).split("_")
        ch_filename = os.path.join(fdr, parts[0]+"_"+parts[-1])
        for i, label in enumerate(self.chan_labels):
            lut_file = ch_filename.replace("Ch*", label)

            chan_properties = read_orac_chan_file(lut_file)
            if self.check and chan_properties["description"] != label:
                raise ValueError("Inconsistent LUT: "+lut_file)

            self.channels[i] = _sensor_ch_num_to_orac_ch_num(
                chan_properties["name"], label
            )
            if chan_properties["solar"]:
                self.solar[i] = True
                f0.append(chan_properties["vis"]["f01"][0])
                f1.append(chan_properties["vis"]["f01"][1])

            if chan_properties["thermal"]:
                self.thermal[i] = True
                b1.append(chan_properties["ir"]["b1"])
                b2.append(chan_properties["ir"]["b2"])
                t1.append(chan_properties["ir"]["t1"])
                t2.append(chan_properties["ir"]["t2"])

        if self.check:
            if len(f0) != self.solar.sum():
                raise ValueError("Inconstent solar channels: "+self.filename)
            if len(b1) != self.thermal.sum():
                raise ValueError("Inconsistent thermal channels: "+self.filename)

        if np.any(self.solar):
            ind = np.argsort(self.channels[self.solar])
            self.f0 = np.asarray(f0)[ind]
            self.f1 = np.asarray(f1)[ind]
        else:
            self.f0 = None
            self.f1 = None

        if np.any(self.thermal):
            ind = np.argsort(self.channels[self.thermal])
            self.b1 = np.asarray(b1)[ind]
            self.b2 = np.asarray(b2)[ind]
            self.t1 = np.asarray(t1)[ind]
            self.t2 = np.asarray(t2)[ind]
        else:
            self.b1 = None
            self.b2 = None
            self.t1 = None
            self.t2 = None

        ind = np.argsort(self.channels)
        self.chan_labels = [self.chan_labels[i] for i in ind]
        self.channels = self.channels[ind]
        self.solar = self.solar[ind]
        self.thermal = self.thermal[ind]

    def _stack_orac_text_lut(self, code):
        """Open a set of channels from ORAC text look-up tables"""

        shape = None
        for i, label in enumerate(self.chan_labels):
            lut_file = self.filename.replace("RD", code).replace("Ch*", label)

            try:
                tables, axes, naxes, _ = read_orac_text_lut(lut_file)
            except FileNotFoundError:
                continue
            if shape is None:
                shape = np.insert(naxes, 0, self.nch)
                result = np.empty(shape)
                base_axes = axes
            elif self.check:
                for ax0, ax1 in zip(base_axes, axes):
                    if not np.allclose(ax0, ax1):
                        raise ValueError("Inconsistent LUT: "+lut_file)

            result[i] = tables[0]

        # Inspect axes spacing
        spacing = []
        for ax in axes:
            spaces = np.unique(np.diff(ax))
            # Check if grid evenly spaced
            label = "uneven_" if spaces.size > 1 else ""
            # Guess axes is log spaced if any values are negative
            label += "logarithmic" if np.any(ax < 0.) else "linear"
            spacing.append(label)

        return result, axes, spacing

    def _stack_orac_text_lut_pair(self, code):
        """Open a pair of ORAC tables for a set of channels"""

        shape0 = None
        for i, label in enumerate(self.chan_labels):
            lut_file = self.filename.replace("RD", code).replace("Ch*", label)

            try:
                tables, axes, naxes, _ = read_orac_text_lut(lut_file)
            except FileNotFoundError:
                continue
            if shape0 is None:
                shape0 = np.insert(naxes, 0, self.nch)
                result0 = np.empty(shape0)
                shape1 = np.insert(naxes[::2], 0, self.nch)
                result1 = np.empty(shape1)
                base_axes = axes
            elif self.check:
                for ax0, ax1 in zip(base_axes, axes):
                    if not np.allclose(ax0, ax1):
                        raise ValueError("Inconsistent LUT: "+lut_file)

            result0[i] = tables[0]
            result1[i] = tables[1]

        # Inspect axes spacing
        spacing = []
        for ax in axes:
            spaces = np.unique(np.diff(ax))
            label = "uneven_" if spaces.size > 1 else ""
            label += "logarithmic" if np.any(ax < 0.) else "linear"
            spacing.append(label)

        return result0, result1, axes, spacing

    def _open_text_tables(self, method):
        """Open set of ORAC text look-up tables"""

        rd, rfd, axes, spacing = self._stack_orac_text_lut_pair("RD")
        self.r_dv = RearrangedRegularGridInterpolator(
            [self.channels] + axes, rd, method, (3, 2, 0, 1)
        )
        self.r_dd = RearrangedRegularGridInterpolator(
            [self.channels, axes[0], axes[2]], rfd, method, (2, 1, 0)
        )
        self.effective_radius_spacing = spacing[0]
        self.satellite_zenith_spacing = spacing[1]
        self.optical_depth_spacing = spacing[2]

        td, tfd, axes, _ = self._stack_orac_text_lut_pair("TD")
        self.t_dv = RearrangedRegularGridInterpolator(
            [self.channels] + axes, td, method, (3, 2, 0, 1)
        )
        self.t_dd = RearrangedRegularGridInterpolator(
            [self.channels, axes[0], axes[2]], tfd, method, (2, 1, 0)
        )

        # Omit the redundant tau dimension
        bext, axes, _ = self._stack_orac_text_lut("Bext")
        self.ext = RearrangedRegularGridInterpolator(
            [self.channels, axes[0]], bext[:,:,0], method, (1, 0)
        )
        bextrat, axes, _ = self._stack_orac_text_lut("BextRat")
        self.ext_ratio = RearrangedRegularGridInterpolator(
            [self.channels, axes[0]], bextrat[:,:,0], method, (1, 0)
        )

        if np.any(self.solar):
            # Have to invert the relative azimuth axis
            rbd, rfbd, axes, spacing = self._stack_orac_text_lut_pair("RBD")
            axes[1] = 180. - axes[1]
            self.r_0v = RearrangedRegularGridInterpolator(
                [self.channels] + axes, rbd, method, (5, 4, 0, 2, 1, 3)
            )
            self.r_0d = RearrangedRegularGridInterpolator(
                [self.channels, axes[0], axes[2], axes[4]], rfbd, method, (3, 2, 0, 1)
            )
            self.relative_azimuth_spacing = spacing[1]
            self.solar_zenith_spacing = spacing[2]

            tb, axes, _ = self._stack_orac_text_lut("TB")
            self.t_00 = RearrangedRegularGridInterpolator(
                [self.channels] + axes, tb, method, (3, 2, 0, 1)
            )

            tbd, tfbd, axes, _ = self._stack_orac_text_lut_pair("TBD")
            axes[1] = 180. - axes[1]
            self.t_0d = RearrangedRegularGridInterpolator(
                [self.channels, axes[0], axes[2], axes[4]], tfbd, method, (3, 2, 0, 1)
            )
        else:
            self.r_0v = None
            self.r_0d = None
            self.t_0d = None
            self.t_00 = None
            self.relative_azimuth_spacing = "unknown"
            self.solar_zenith_spacing = "unknown"

        if np.any(self.thermal):
            em, axes, _ = self._stack_orac_text_lut("EM")
            self.e_md = RearrangedRegularGridInterpolator(
                [self.channels,] + axes, em, method, (3, 2, 0, 1)
            )
        else:
            self.e_md = None

    def _open_ncdf_tables(self, method):
        """Open contents of ORAC netCDF look-up table"""

        with Dataset(self.filename) as lut_file:
            # Identify requested channel subset
            self.channels = lut_file["channel_id"][...]
            self.nch = len(self.channels)

            # Channel information
            self.solar = lut_file["solar_channel_flag"][:] == 1
            self.thermal = lut_file["thermal_channel_flag"][:] == 1

            # Read other axes, noting if they are logarithmic
            def fetch_axes(name):
                var = lut_file[name]
                data = var[...]
                self.__dict__[name + "_spacing"] = var.spacing
                if "logarithmic" in var.spacing:
                    return np.log10(data)
                return data
            tau = fetch_axes("optical_depth")
            re = fetch_axes("effective_radius")
            satzen = fetch_axes("satellite_zenith")
            solzen = fetch_axes("solar_zenith")
            relazi = fetch_axes("relative_azimuth")

            # Read tables
            self.t_dv = RegularGridInterpolator(
                (satzen, tau, re, self.channels), lut_file["T_dv"][...], method
            )
            self.t_dd = RegularGridInterpolator(
                (tau, re, self.channels), lut_file["T_dd"][...], method
            )
            self.r_dv = RegularGridInterpolator(
                (satzen, tau, re, self.channels), lut_file["R_dv"][...], method
            )
            self.r_dd = RegularGridInterpolator(
                (tau, re, self.channels), lut_file["R_dd"][...], method
            )
            self.ext = RegularGridInterpolator(
                (re, self.channels),
                lut_file["extinction_coefficient"][...], method
            )
            self.ext_ratio = RegularGridInterpolator(
                (re, self.channels),
                lut_file["extinction_coefficient_ratio"][...], method
            )

            # Solar channels
            if np.any(self.solar):
                solar_channels = lut_file["solar_channel_id"][:]

                self.r_0v = RegularGridInterpolator(
                    (relazi, satzen, solzen, tau, re, solar_channels),
                    lut_file["R_0v"][...], method
                )
                self.r_0d = RegularGridInterpolator(
                    (solzen, tau, re, solar_channels),
                    lut_file["R_0d"][...], method
                )
                self.t_0d = RegularGridInterpolator(
                    (solzen, tau, re, solar_channels),
                    lut_file["T_0d"][...], method
                )
                self.t_00 = RegularGridInterpolator(
                    (solzen, tau, re, solar_channels),
                    lut_file["T_00"][...], method
                )

                # Solar constant
                self.f0 = lut_file["F0"][:]
                self.f1 = None
            else:
                self.r_0v = None
                self.r_0d = None
                self.t_0d = None
                self.t_00 = None
                self.f0 = None
                self.f1 = None

            # Thermal channels
            if np.any(self.thermal):
                thermal_channels = lut_file["thermal_channel_id"][:]

                self.e_md = RegularGridInterpolator(
                    (satzen, tau, re, thermal_channels),
                    lut_file["E_md"][...], method
                )

                # Planck coefficients
                self.b1 = lut_file["B1"][:]
                self.b2 = lut_file["B2"][:]
                self.t1 = lut_file["T1"][:]
                self.t2 = lut_file["T2"][:]
            else:
                self.e_md = None
                self.b1 = None
                self.b2 = None
                self.t1 = None
                self.t2 = None

    def solar_constant(self, doy):
        if self.f1 is not None:
            # Old approximation for F0 variation throughout the year
            return self.f0 + self.f1 * np.cos(2. * np.pi * doy / 365.)

        # New (better) approximation
        return self.f0 / (1. - 0.0167086 * np.cos((2.* np.pi * (doy - 4.)) /
                                                  365.256363))**2

    def rad2temp(self, radiance):
        """Convert radiance into brightness temperature"""

        radiance[radiance < 1e-6] = 1e-6

        c = np.log(self.b1 / radiance + 1.)
        t_eff = self.b2 / c
        t = (t_eff - self.t1) / self.t2

        dt_dr = self.b1 * self.b2 / (self.t2 * c * c * radiance * (radiance + self.b1))

        return t, dt_dr

    def temp2rad(self, temperature):
        """Convert brightness temperature into radiance"""

        t_eff = temperature * self.t2 + self.t1
        bb = self.b2 / t_eff
        c = np.exp(bb)
        denom = c - 1.
        r = self.b1 / denom

        dr_dt = self.b1 * bb * c * self.t2 / (t_eff * denom * denom)

        return r, dr_dt

    def __call__(self, channels, satzen, solzen, relazi, tau, re):
        """Deal with different dimension order/manner in types of LUT"""
        if self.satellite_zenith_spacing.endswith("logarithmic"):
            satzen = np.log10(satzen)
        if self.solar_zenith_spacing.endswith("logarithmic"):
            solzen = np.log10(solzen)
        if self.relative_azimuth_spacing.endswith("logarithmic"):
            relazi = np.log10(relazi)
        if self.optical_depth_spacing.endswith("logarithmic"):
            tau = np.log10(tau)
        if self.effective_radius_spacing.endswith("logarithmic"):
            re = np.log10(re)

        # Only ouuput sets that apply to every requsted channel
        solar = True
        thermal = True
        for requested_ch in channels:
            for available_ch, is_solar, is_thermal in zip(self.channels, self.solar, self.thermal):
                if requested_ch == available_ch:
                    solar &= is_solar
                    thermal &= is_thermal
                    break
            else:
                raise ValueError(f"Requested unavailable channel: {requested_ch}")

        out = dict(
            t_dv=self.t_dv((satzen, tau, re, channels)),
            t_dd=self.t_dd((tau, re, channels)),
            r_dv=self.r_dv((satzen, tau, re, channels)),
            r_dd=self.r_dd((tau, re, channels)),
            ext=self.ext((re, channels)),
            ext_ratio=self.ext_ratio((re, channels)),
        )
        if solar:
            out["r_0v"] = self.r_0v((relazi, satzen, solzen, tau, re, channels))
            out["r_0d"] = self.r_0d((solzen, tau, re, channels))
            out["t_0d"] = self.t_0d((solzen, tau, re, channels))
            out["t_00"] = self.t_00((solzen, tau, re, channels))
            # Use satzen on solzen axis (which assumes its in bounds)
            out["t_vd"] = self.t_0d((satzen, tau, re, channels))
            out["t_vv"] = self.t_00((satzen, tau, re, channels))
        if thermal:
            out["e_md"] = self.e_md((satzen, tau, re, channels))

        return out


def _orac_ch_num_to_sensor_ch_num(sensor, chan_num):
    """Convert ORAC channel number into a sensor channel number"""

    if sensor.startswith("AVHRR") and chan_num > 2:
        # AVHRR has some bespoke channel numbers
        if chan_num == 3:
            return "Ch3a"
        if chan_num == 4:
            return "Ch3b"
        chan_num = chan_num-1

    return "Ch{:0d}".format(chan_num)


def _sensor_ch_num_to_orac_ch_num(sensor, chan_num):
    """Convert sensor channel number into an ORAC channel number"""

    # AVHRR has some bespoke channel numbers
    if chan_num == "Ch3a":
        return 3
    if chan_num == "Ch3b":
        return 4

    number = int(chan_num[2:])
    if sensor.startswith("AVHRR") and number > 2:
        return number+1
    return number


def read_orac_chan_file(filename):
    """Parse an ORAC channel description text file"""
    def clean_line(unit):
        line = next(unit)
        parts = line.split("%")
        return parts[0].strip()

    with open(filename) as lut_file:
        result = dict(
            name=clean_line(lut_file),
            description=clean_line(lut_file),
            file_id=clean_line(lut_file),
            wavenumber=float(clean_line(lut_file)),
            thermal=bool(int(clean_line(lut_file)))
        )
        if result["thermal"]:
            result["ir"] = dict(
                b1=float(clean_line(lut_file)),
                b2=float(clean_line(lut_file)),
                t1=float(clean_line(lut_file)),
                t2=float(clean_line(lut_file)),
                ir_nehomog=np.fromstring(clean_line(lut_file), sep=","),
                ir_necoreg=np.fromstring(clean_line(lut_file), sep=","),
                nebt=float(clean_line(lut_file))
            )

        result["solar"] = bool(int(clean_line(lut_file)))
        if result["solar"]:
            result["vis"] = dict(
                f01=np.fromstring(clean_line(lut_file), sep=","),
                vis_nehomog=np.fromstring(clean_line(lut_file), sep=",") * 1e-2,
                vis_necoreg=np.fromstring(clean_line(lut_file), sep=",") * 1e-2,
                nedr=float(clean_line(lut_file)),
                rs=np.fromstring(clean_line(lut_file), sep=",") * 1e-2
            )
            result["vis"]["nedr"] /= result["vis"]["f01"][0]

    return result


def read_orac_text_lut(filename):
    """Open an ORAC text look-up table as a numpy array

    Reads contents into string, then parses with np.fromstring
    as we don't know if the file contains 1 or 2 tables before
    reading and they don't always end with an even number of
    'columns'"""

    naxes = []
    daxes = []
    axes = []
    contents = None
    # Parse header manually
    with open(filename) as lut_file:
        for i, line in enumerate(lut_file):
             if contents is None:
                parts = line.split()
                if i == 0 and len(parts) == 1:
                    # First line (which BextRat files don't have)
                    wvl = float(line)
                elif len(parts) == 2:
                    # Axis definition
                    naxes.append(int(parts[0]))
                    daxes.append(float(parts[1]))
                    axes.append([])
                elif len(axes[-1]) != naxes[-1]:
                    # Elements of an axis
                    axes[-1].extend(map(float, parts))
                else:
                    # Contents of the file
                    contents = line
             else:
                contents += " "
                contents += line

    naxes = np.fromiter(reversed(naxes), int)
    daxes = np.fromiter(reversed(daxes), float)
    axes = [np.array(s) for s in reversed(axes)]

    # Tables all scalled by factor 100
    lut = np.fromstring(contents, sep=" ") * 0.01
    n = np.prod(naxes).astype(int)
    if lut.size == n:
        tables = (lut.reshape(naxes), )
    else:
        m = np.prod(naxes[::2]).astype(int)
        tables = (lut[:-m].reshape(naxes),
                  lut[-m:].reshape(naxes[::2]))

    return tables, axes, naxes, daxes
