#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains VMx physical constants and parameters as enumerated values.
"""

from enum import Enum
import sys


class PhysicsConstants(str, Enum):
    """
    Enum describing physical constants i.e. permeability of free space mu_0.
    """

    mu_0 = 1.25663706e-6, 'Permeability of free space', 'm*kg*s^(-2)*A^(-2)'

    def __new__(cls, constant, description, siunits):
        """
        Override the enum __new__ method to add more attributes to each enum.
        :param str constant: Value of physical constant
        :param str description: Description
        :param str siunits: SI units
        :return:
        """

        obj = str.__new__(cls)
        obj._value_ = constant
        obj.desc = description
        obj.units = siunits

        return obj

    def __float__(self):
        return float(str(self.value))

    @staticmethod
    def getEnum(enumString, hardFail=True):
        """
        Return enum associated with input. If no match then fail.
        Optional - set hardFail=False to allow execution to continue.
        Will just return None.
        :param str enumString:
        :param hardFail:
        :return:
        """

        try:
            return PhysicsConstants[enumString]
        except:
            print("ERROR! Could not get enum associated with {}".format(enumString))
            if hardFail:
                print("Stopping with Error!")
                sys.exit(1)
            else:
                print("Returning None")
                return None


class InstrumentParams(str, Enum):
    """
    Enum describing instrument parameters, i.e. magnetometer locations.
    """
    sense_rod_offset = '0.0045 -0.00745 0.0095', 'all', 0.00

    # 20221213 and 20221216 array positions
    mag1 = '-0.05905 0.00 -0.04475', 'sn2215', 0.01
    mag2 = '0.05905 0.00 -0.04475', 'sn2216', 0.01
    mag3 = '0.00 -0.059 0.04425', 'sn2217', 0.01
    mag4 = '0.00 0.059 0.04525', 'sn2214', 0.01

    # 20220317 array positions
    # mag1 = '-0.048301 0.002 -0.0446405', 'sn2216', 0.01
    # mag2 = '0.048301 -0.002 -0.0456565', 'sn2219', 0.01
    # mag3 = '-0.002 -0.048301 0.045275499999999996', 'sn2214', 0.01
    # mag4 = '0.002 0.048301 0.04502149999999999', 'sn2215', 0.01

    afg = '0.002 0.048301 0.04502149999999999', 'afg', 0.01

    def __new__(cls, location, sn, gain):
        """
        Override the enum __new__ method to add more attributes to each enum.
        :param str location: Location of magnetometer
        :param str sn: Mag566 Serial Number
        :return:
        """

        obj = str.__new__(cls)
        obj._value_ = location
        obj.sn = sn
        obj.gain = gain

        return obj

    @staticmethod
    def getEnum(enumString, hardFail=True):
        """
        Return enum associated with input. If no match then fail.
        Optional - set hardFail=False to allow execution to continue.
        Will just return None.
        :param str enumString:
        :param hardFail:
        :return:
        """

        try:
            return InstrumentParams[enumString]
        except:
            print("ERROR! Could not get enum associated with {}".format(enumString))
            if hardFail:
                print("Stopping with Error!")
                sys.exit(1)
            else:
                print("Returning None")
                return None


class SourceVector(str, Enum):
    """
    Enum describing source vectors, i.e. direction to source location or source moment direction
    """
    x = '1.0 0.0 0.0', 'Unit vector along x-axis'
    y = '0.0 1.0 0.0', 'Unit vector along y-axis'
    z = '0.0 0.0 1.0', 'Unit vector along z-axis'
    xy = '0.7071067811865476 0.7071067811865476 0.0', 'Unit vector 45 degrees in xy-plane'
    yz = '0.0 0.7071067811865476 0.7071067811865476', 'Unit vector 45 degrees in yz-plane'
    xz = '0.7071067811865476 0.0 0.7071067811865476', 'Unit vector 45 degrees in xz-plane'
    xyz = '0.5773502691896258 0.5773502691896258 0.5773502691896258', 'Unit vector 45 degrees from xy- and yz- and xy-planes'

    def __new__(cls, unitvec, desc):
        """
        Override the enum __new__ method to add more attributes to each enum.
        :param str unitvec: 3D unit vector
        :param str desc: description of unit vector
        :return:
        """

        obj = str.__new__(cls)
        obj._value_ = unitvec
        obj.sn = desc

        return obj

    @staticmethod
    def getEnum(enumString, hardFail=False):
        """
        Return enum associated with input. If no match then fail.
        Optional - set hardFail=False to allow execution to continue.
        Will just return None.
        :param str enumString:
        :param hardFail:
        :return:
        """

        try:
            return SourceVector[enumString]
        except:
            if (hardFail):
                print("Stopping with Error!")
                sys.exit(1)
            else:
                return None
