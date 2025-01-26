# -*- coding: utf-8 -*-
#
# Copyright European Organization for Nuclear Research (CERN)
# All rights reserved
#
# Please look at the supplied documentation for the user's
# license
#
# Author: Vasilis.Vlachoudis@cern.ch
# Date:   24-Oct-2006

__author__ = "Vasilis Vlachoudis"
__email__  = "Vasilis.Vlachoudis@cern.ch"

import io
import re
import math
import shlex
import struct
try:
	import numpy as np
except ImportError:
	np = None
from collections import namedtuple

from rt2.fluka import bmath
from rt2.fluka import fortran
from rt2.fluka.log import say

_detName         = re.compile(r"^([0-9-]+) (.*)$")
_detectorPattern = re.compile(r"^ ?# ?Detector ?n?:\s*\d*\s*(.*)\s*$")
_blockPattern    = re.compile(r"^ ?# ?Block ?n?:\s*\d*\s*(.*)\s*$")
_varPat          = re.compile(r"^ ?# ?(\S+):?\s*(.*)\s*$")

EnergyEvent = namedtuple("EnergyEvent", ["x", "y", "z", "rull"])
SourceEvent = namedtuple("SourceEvent", ["ptype", "etot", "weight", "x", "y", "z", "tx", "ty", "tz"])

#-------------------------------------------------------------------------------
# Unpack an array of floating point numbers
#-------------------------------------------------------------------------------
def unpackArray(data):
	return struct.unpack("=%df"%(len(data)//4),  data)

#===============================================================================
# Empty class to fill with detector data
#===============================================================================
class Detector:
	def __init__(self, num, name, type_=None):
		self.num  = num
		self.name = name
		self.type = type_
		self.data = None
		self.var  = {}

	#-----------------------------------------------------------------------
	def __getitem__(self, key):
		return self.var[key]

	#-----------------------------------------------------------------------
	def __call__(self, key, default=None):
		return self.var.get(key,default)

	# ----------------------------------------------------------------------
	# Convert data to numpy format if possible
	# ----------------------------------------------------------------------
	def toNumpy(self, dtype=float):
		try:
			self.data = np.array(self.data, dtype=dtype)
		except:
			pass

#===============================================================================
# Base class for all detectors
#===============================================================================
class Usrxxx:
	def __init__(self, filename=None):
		"""Initialize a USRxxx structure"""
		self.reset()
		self.f = None
		if filename is None:
			self.filename = ""
		else:
			self.filename = filename
			self.read()

	# ----------------------------------------------------------------------
	def reset(self):
		"""Reset header information"""
		self.title    = ""
		self.time     = ""
		self.weight   =  0
		self.ncase    =  0
		self.nbatch   =  0
		self.detector = []
		self.detname  = {}
		self.seekpos  = -1
		self.statpos  = -1

	# ----------------------------------------------------------------------
	def addDetector(self, num, name, type_=None):
		det = Detector(num, name, type_)
		self.detector.append(det)
		self.detname[name] = det
		return det

	# ----------------------------------------------------------------------
	def toNumpy(self):
		for det in self.detector:
			det.toNumpy()

	# ----------------------------------------------------------------------
	def __getitem__(self, key):
		if isinstance(key, int):
			return self.detector[key]
		else:
			return self.detname[key]

	# ----------------------------------------------------------------------
	def open(self, filename=None):
		"""Read header information, and return the file handle"""
		if filename is not None:
			self.filename = filename
		#if self.f: raise Exception("File is already open")
		self.f = open(self.filename, "rb")

	# ----------------------------------------------------------------------
	def close(self):
		self.f.close()
		self.f = None

	# ----------------------------------------------------------------------
	def read(self):
		self.readHeader()	# default

	# ----------------------------------------------------------------------
	# Read information from USRxxx file
	# @return the handle to the file opened
	# ----------------------------------------------------------------------
	def readHeader(self):
		"""Read header information, and return the file handle"""
		self.reset()

		# Read header
		data = fortran.read(self.f)
		if data is None: raise IOError("Invalid USRxxx file")
		size   = len(data)
		over1b = 0
		if   size == 116:
			(title, time, self.weight) = \
				struct.unpack("=80s32sf", data)
			self.ncase  = 1
			self.nbatch = 1
		elif size == 120:
			(title, time, self.weight, self.ncase) = \
				struct.unpack("=80s32sfi", data)
			self.nbatch = 1
		elif size == 124:
			(title, time, self.weight,
			 self.ncase, self.nbatch) = \
				struct.unpack("=80s32sfii", data)
		elif size == 128:
			(title, time, self.weight,
			 self.ncase, over1b, self.nbatch) = \
				struct.unpack("=80s32sfiii", data)
		else:
			raise IOError("Invalid USRxxx file")

		if over1b>0:
			self.ncase = self.ncase + over1b*1000000000

		self.title = title.strip().decode(errors="replace")
		self.time  = time.strip().decode(errors="replace")

	# ----------------------------------------------------------------------
	# Read detector data
	# ----------------------------------------------------------------------
	def readData(self, n):
		"""Read n(th) detector data structure"""
		self.open()
		fortran.skip(self.f)	# Skip header
		for _ in range(2*n):
			fortran.skip(self.f)	# Detector Header & Data
		fortran.skip(self.f)		# Detector Header
		data = fortran.read(self.f)
		self.close()
		return data

	# ----------------------------------------------------------------------
	# Read detector statistical data
	# ----------------------------------------------------------------------
	def readStat(self, n):
		"""Read n(th) detector statistical data"""
		if self.statpos < 0: return None
		self.open()
		self.f.seek(self.statpos)
		for _ in range(n):
			fortran.skip(self.f)	# Detector Data
		data = fortran.read(self.f)
		self.close()
		return data

	# ----------------------------------------------------------------------
	def sayHeader(self):
		say("File   : ",self.filename)
		say("Title  : ",self.title)
		say("Time   : ",self.time)
		say("Weight : ",self.weight)
		say("NCase  : ",self.ncase)
		say("NBatch : ",self.nbatch)

#===============================================================================
# Residual nuclei detector
#===============================================================================
class Resnuclei(Usrxxx):
	# ----------------------------------------------------------------------
	# Read information from a RESNUCLEi file
	# Fill the self.detector structure
	# ----------------------------------------------------------------------
	def readHeader(self, filename=None):
		"""Read residual nuclei detector information"""
		self.open(filename)
		super().readHeader()
		self.nisomers = 0
		if self.ncase <= 0:
			self.evol = True
			self.ncase = -self.ncase

			data = fortran.read(self.f)
			nir = (len(data)-4)//8
			self.irrdt = struct.unpack("=i%df"%(2*nir), data)
		else:
			self.evol  = False
			self.irrdt = None

		for _ in range(1000):
			# Header
			data = fortran.read(self.f)
			if data is None: break
			size = len(data)
			self.irrdt = None

			# Statistics are present?
			if size == 14:
				if data[:8] == b"ISOMERS:":
					self.nisomers = struct.unpack("=10xi",data)[0]
					data = fortran.read(self.f)
					data = fortran.read(self.f)
					size = len(data)
				if data[:10] == b"STATISTICS":
					self.statpos = self.f.tell()
					break
			elif size != 38:
				raise IOError("Invalid RESNUCLEi file header size=%d"%(size))

			# Parse header
			header = struct.unpack("=i10siif3i", data)

			det = self.addDetector(
					header[0],
					header[1].strip().decode(errors="replace"),
					header[2])
			det.region = header[3]
			det.volume = header[4]
			det.mhigh  = header[5]
			det.zhigh  = header[6]
			det.nmzmin = header[7]

			if self.evol:
				data = fortran.read(self.f)
				self.tdecay = struct.unpack("=f", data)
			else:
				self.tdecay = 0.0

			size  = det.zhigh * det.mhigh * 4
			if size != fortran.skip(self.f):
				raise IOError("Invalid RESNUCLEi file")
		self.close()

	# ----------------------------------------------------------------------
	# Read detector data
	# ----------------------------------------------------------------------
	def readData(self, n):
		"""Read n(th) detector data structure"""
		self.open()
		fortran.skip(self.f)
		if self.evol:
			fortran.skip(self.f)

		for _ in range(n):
			fortran.skip(self.f)		# Detector Header & Data
			if self.evol:
				fortran.skip(self.f)	# TDecay
			fortran.skip(self.f)		# Detector data
			if self.nisomers:
				fortran.skip(self.f)	# Isomers header
				fortran.skip(self.f)	# Isomers data

		fortran.skip(self.f)			# Detector Header & Data
		if self.evol:
			fortran.skip(self.f)		# TDecay
		data = fortran.read(self.f)		# Detector data
		self.close()
		return data

	# ----------------------------------------------------------------------
	# Read detector isomeric data
	#SM START: Added method to read isomeric data  02/08/2016
	# ----------------------------------------------------------------------
	def readIso(self, n):
		"""Read detector det data structure"""
		#print "self.nisomers:", self.nisomers
		if self.nisomers < 0: return None
		self.open()
		fortran.skip(self.f)
		if self.evol:
			fortran.skip(self.f)

		for _ in range(n):
			fortran.skip(self.f)		# Detector Header & Data
			if self.evol:
				fortran.skip(self.f)	# TDecay
			fortran.skip(self.f)		# Detector data
			if self.nisomers:
				fortran.skip(self.f)	# Isomers header
				fortran.skip(self.f)	# Isomers data
		fortran.skip(self.f)
		if self.evol:
			fortran.skip(self.f)		# TDecay
		fortran.skip(self.f)			# Detector data
		isohead = fortran.read(self.f)		# Isomers header
		data = fortran.read(self.f)		# Isomers data
		#print "isohead:",len(isohead)
		#header = struct.unpack("=10xi", isohead)
		#print "isohead:",header[0]
		self.close()
		return (isohead, data)

	# ----------------------------------------------------------------------
	# Read detector statistical data
	# ----------------------------------------------------------------------
	def readStat(self, n):
		"""Read n(th) detector statistical data"""
		if self.statpos < 0: return None
		self.open()
		self.f.seek(self.statpos)

		self.f.seek(self.statpos)
		if self.nisomers:
			nskip = 7*n
		else:
			nskip = 6*n
		for _ in range(nskip):
			fortran.skip(self.f)		# Detector Data

		total = fortran.read(self.f)
		A     = fortran.read(self.f)
		errA  = fortran.read(self.f)
		Z     = fortran.read(self.f)
		errZ  = fortran.read(self.f)
		data  = fortran.read(self.f)
		if self.nisomers:
			iso = fortran.read(self.f)
		else:
			iso = None
		self.close()
		return (total, A, errA, Z, errZ, data, iso)

	# ----------------------------------------------------------------------
	def say(self, det=None):
		"""print header/detector information"""
		if det is None:
			self.sayHeader()
		else:
			binning = self.detector[det]
			say("Bin    : ", binning.num)
			say("Title  : ", binning.name)
			say("Type   : ", binning.type)
			say("Region : ", binning.region)
			say("Volume : ", binning.volume)
			say("Mhigh  : ", binning.mhigh)
			say("Zhigh  : ", binning.zhigh)
			say("NMZmin : ", binning.nmzmin)

#===============================================================================
# Usrbdx Boundary Crossing detector
#===============================================================================
class Usrbdx(Usrxxx):
	# ----------------------------------------------------------------------
	# Read information from a USRBDX file
	# Fill the self.detector structure
	# ----------------------------------------------------------------------
	def readHeader(self, filename=None):
		"""Read boundary crossing detector information"""
		self.open(filename)
		super().readHeader()

		for _ in range(1000):
			# Header
			data = fortran.read(self.f)
			if data is None: break
			size = len(data)

			# Statistics are present?
			if size == 14:
				# In statistics
				#   1: total, error
				#   2: N,NG,Elow (array with Emaxi)
				#   3: Differential integrated over solid angle
				#   4: -//- errors
				#   5: Cumulative integrated over solid angle
				#   6: -//- errors
				#   7: Double differential data
				self.statpos = self.f.tell()
				for det in self.detector:
					data = unpackArray(fortran.read(self.f))
					det.total = data[0]
					det.totalerror = data[1]
					for j in range(6):
						fortran.skip(self.f)
				break
			elif size != 78: raise IOError("Invalid USRBDX file")

			# Parse header
			header = struct.unpack("=i10siiiifiiiffifffif", data)

			det = self.addDetector(
					header[0],
					header[ 1].strip().decode(errors="replace"), # titusx
					header[ 2])		# itusbx
			det.dist    = header[ 3]		# idusbx
			det.reg1    = header[ 4]		# nr1usx
			det.reg2    = header[ 5]		# nr2usx
			det.area    = header[ 6]		# ausbdx
			det.twoway  = header[ 7]		# lwusbx
			det.fluence = header[ 8]		# lfusbx
			det.lowneu  = header[ 9]		# llnusx
			det.elow    = header[10]		# ebxlow
			det.ehigh   = header[11]		# ebxhgh
			det.ne      = header[12]		# nebxbn
			det.de      = header[13]		# debxbn
			det.alow    = header[14]		# abxlow
			det.ahigh   = header[15]		# abxhgh
			det.na      = header[16]		# nabxbn
			det.da      = header[17]		# dabxbn

			if det.lowneu:
				data = fortran.read(self.f)
				det.ngroup = struct.unpack("=i",data[:4])[0]
				det.egroup = struct.unpack("=%df"%(det.ngroup+1), data[4:])
			else:
				det.ngroup = 0
				det.egroup = []

			size  = (det.ngroup+det.ne) * det.na * 4
			if size != fortran.skip(self.f):
				raise IOError("Invalid USRBDX file")
		self.close()

	# ----------------------------------------------------------------------
	# Read detector data
	# ----------------------------------------------------------------------
	def readData(self, n):
		"""Read n(th) detector data structure"""
		self.open()
		fortran.skip(self.f)
		for i in range(n):
			fortran.skip(self.f)			# Detector Header
			if self.detector[i].lowneu:
					fortran.skip(self.f)	# Detector low energy neutron groups
			fortran.skip(self.f)			# Detector data
		fortran.skip(self.f)				# Detector Header
		if self.detector[n].lowneu:
				fortran.skip(self.f)		# Detector low energy neutron groups
		data = fortran.read(self.f)			# Detector data
		self.close()
		return data

	# ----------------------------------------------------------------------
	# Read detector statistical data
	# ----------------------------------------------------------------------
	def readStat(self, n):
		"""Read n(th) detector statistical data"""
		if self.statpos < 0: return None
		self.open()
		self.f.seek(self.statpos)
		for _ in range(n):
			for j in range(7):
				fortran.skip(self.f)	# Detector Data

		for _ in range(6):
			fortran.skip(self.f)		# Detector Data
		data = fortran.read(self.f)
		self.close()
		return data

	# ----------------------------------------------------------------------
	def say(self, det=None):
		"""print header/detector information"""
		if det is None:
			self.sayHeader()
		else:
			det = self.detector[det]
			say("BDX    : ", det.num)
			say("Title  : ", det.name)
			say("Type   : ", det.type)
			say("Dist   : ", det.dist)
			say("Reg1   : ", det.reg1)
			say("Reg2   : ", det.reg2)
			say("Area   : ", det.area)
			say("2way   : ", det.twoway)
			say("Fluence: ", det.fluence)
			say("LowNeu : ", det.lowneu)
			say("Energy : [", det.elow,"..",det.ehigh,"] ne=", det.ne, "de=",det.de)
			if det.lowneu:
				say("LOWNeut : [",det.egroup[-1],"..",det.egroup[0],"] ne=",det.ngroup)
			say("Angle  : [", det.alow,"..",det.ahigh,"] na=", det.na, "da=",det.da)
			say("Total  : ", det.total, "+/-", det.totalerror)

#===============================================================================
# Usrbin detector
#===============================================================================
class Usrbin(Usrxxx):
	# ----------------------------------------------------------------------
	# Read information from USRBIN file
	# Fill the self.detector structure
	# ----------------------------------------------------------------------
	def readHeader(self, filename=None):
		"""Read USRBIN detector information"""
		self.open(filename)
		super().readHeader()

		for _ in range(1000):
			# Header
			data = fortran.read(self.f)
			if data is None: break
			size = len(data)

			# Statistics are present?
			if size == 14 and data[:10] == b"STATISTICS":
				self.statpos = self.f.tell()
				break
			elif size != 86: raise IOError("Invalid USRBIN file")

			# Parse header
			header = struct.unpack("=i10siiffifffifffififff", data)

			usrbin = self.addDetector(
					header[0],
					header[1].strip().decode(errors="replace"),
					header[2])
			usrbin.score = header[3]

			usrbin.xlow  = float(bmath.format(header[ 4],9))
			usrbin.xhigh = float(bmath.format(header[ 5],9))
			usrbin.nx    = header[ 6]
			if usrbin.nx > 0 and usrbin.type not in (2,12,8,18):
				usrbin.dx = (usrbin.xhigh-usrbin.xlow) / float(usrbin.nx)
			else:
				usrbin.dx = float(bmath.format(header[ 7],9))

			usrbin.ylow  = float(bmath.format(header[ 8],9))
			usrbin.yhigh = float(bmath.format(header[ 9],9))
			if usrbin.type in (1,11):
				# Round to pi if needed
				if abs(usrbin.ylow+math.pi) < 1e-6:
					usrbin.ylow = -math.pi
				if abs(usrbin.yhigh-math.pi) < 1e-6:
					usrbin.yhigh = math.pi
				elif abs(usrbin.yhigh-math.pi*2) < 1e-6:
					usrbin.yhigh = 2*math.pi
			usrbin.ny = header[10]
			if usrbin.ny > 0 and usrbin.type not in (2,12,8,18):
				usrbin.dy = (usrbin.yhigh-usrbin.ylow) / float(usrbin.ny)
			else:
				usrbin.dy = float(bmath.format(header[11],9))

			usrbin.zlow  = float(bmath.format(header[12],9))
			usrbin.zhigh = float(bmath.format(header[13],9))
			usrbin.nz    = header[14]
			if usrbin.nz > 0 and usrbin.type not in (2,12):	# 8=special with z=real
				usrbin.dz = (usrbin.zhigh-usrbin.zlow) / float(usrbin.nz)
			else:
				usrbin.dz = float(bmath.format(header[15],9))

			usrbin.lntzer = header[16]
			usrbin.bk     = header[17]
			usrbin.b2     = header[18]
			usrbin.tc     = header[19]

			size  = usrbin.nx * usrbin.ny * usrbin.nz * 4
			if fortran.skip(self.f) != size:
				raise IOError("Invalid USRBIN file")
		self.close()

	# ----------------------------------------------------------------------
	# Read detector data
	# ----------------------------------------------------------------------
	def readData(self, n):
		"""Read n(th) detector data structure"""
		self.open()
		fortran.skip(self.f)
		for _ in range(n):
			fortran.skip(self.f)		# Detector Header
			fortran.skip(self.f)		# Detector data
		fortran.skip(self.f)			# Detector Header
		data = fortran.read(self.f)		# Detector data
		self.close()
		return data

	# ----------------------------------------------------------------------
	# Read data and return a numpy array
	# ----------------------------------------------------------------------
	def readArray(self, n):
		dim  = [self.detector[n].nx, self.detector[n].ny, self.detector[n].nz]
		return np.reshape(np.frombuffer(self.readData(n), np.float32), dim, order="F")

	# ----------------------------------------------------------------------
	# Read detector statistical data
	# ----------------------------------------------------------------------
	def readStat(self, n):
		"""Read n(th) detector statistical data"""
		if self.statpos < 0: return None
		self.open()
		self.f.seek(self.statpos)
		for _ in range(n):
			fortran.skip(self.f)		# Detector Data
		data = fortran.read(self.f)
		self.close()
		return data

	# ----------------------------------------------------------------------
	def say(self, det=None):
		"""print header/detector information"""
		if det is None:
			self.sayHeader()
		else:
			binning = self.detector[det]
			say("Bin    : ", binning.num)
			say("Title  : ", binning.name)
			say("Type   : ", binning.type)
			say("Score  : ", binning.score)
			say("X      : [", binning.xlow,"-",binning.xhigh,"] x", binning.nx, "dx=",binning.dx)
			say("Y      : [", binning.ylow,"-",binning.yhigh,"] x", binning.ny, "dy=",binning.dy)
			say("Z      : [", binning.zlow,"-",binning.zhigh,"] x", binning.nz, "dz=",binning.dz)
			say("L      : ", binning.lntzer)
			say("bk     : ", binning.bk)
			say("b2     : ", binning.b2)
			say("tc     : ", binning.tc)

#===============================================================================
# MGDRAW output
#===============================================================================
class Mgdraw:
	def __init__(self, filename=None):
		"""Initialize a MGDRAW structure"""
		self.reset()
		if filename is None: return
		self.open(filename)

	# ----------------------------------------------------------------------
	def reset(self):
		"""Reset information"""
		self.filename   = ""
		self.hnd    = None
		self.nevent = 0
		self.data   = None

	# ----------------------------------------------------------------------
	# Open file and return handle
	# ----------------------------------------------------------------------
	def open(self, filename):
		"""Read header information, and return the file handle"""
		self.reset()
		self.filename = filename
		try:
			self.hnd = open(self.filename, "rb")
		except IOError:
			self.hnd = None
		return self.hnd

	# ----------------------------------------------------------------------
	def close(self):
		self.hnd.close()

	# ----------------------------------------------------------------------
	# Read or skip next event from mgread structure
	# ----------------------------------------------------------------------
	def readEvent(self, typeid=None):
		# Read header
		data = fortran.read(self.hnd)
		if data is None: return None
		if len(data) == 20:
			ndum, mdum, jdum, edum, wdum \
				= struct.unpack("=iiiff", data)
		else:
			raise IOError("Invalid MGREAD file")

		self.nevent += 1

		if ndum > 0:
			if typeid is None or typeid == 0:
				self.readTracking(ndum, mdum, jdum, edum, wdum)
			else:
				fortran.skip(self.hnd)
			return 0
		elif ndum == 0:
			if typeid is None or typeid == 1:
				self.readEnergy(mdum, jdum, edum, wdum)
			else:
				fortran.skip(self.hnd)
			return 1
		else:
			if typeid is None or typeid == 2:
				self.readSource(-ndum, mdum, jdum, edum, wdum)
			else:
				fortran.skip(self.hnd)
			return 2

	# ----------------------------------------------------------------------
	def readTracking(self, ntrack, mtrack, jtrack, etrack, wtrack):
		self.ntrack = ntrack
		self.mtrack = mtrack
		self.jtrack = jtrack
		self.etrack = etrack
		self.wtrack = wtrack
		data = fortran.read(self.hnd)
		if data is None: raise IOError("Invalid track event")
		fmt = "=%df" % (3*(ntrack+1) + mtrack + 1)
		self.data = struct.unpack(fmt, data)
		return ntrack

	# ----------------------------------------------------------------------
	def readEnergy(self, icode, jtrack, etrack, wtrack):
		self.icode  = icode
		self.jtrack = jtrack
		self.etrack = etrack
		self.wtrack = wtrack
		data = fortran.read(self.hnd)
		if data is None: raise IOError("Invalid energy deposition event")
		self.data = EnergyEvent(*struct.unpack("=4f", data))
		return icode

	# ----------------------------------------------------------------------
	def readSource(self, ncase, npflka, nstmax, tkesum, weipri):
		self.ncase  = ncase
		self.npflka = npflka
		self.nstmax = nstmax
		self.tkesum = tkesum
		self.weipri = weipri

		data = fortran.read(self.hnd)
		if data is None: raise IOError("Invalid source event")
		fmt = "=" + ("i8f" * npflka)
		self.data = SourceEvent(*struct.unpack(fmt, data))
		return ncase

#===============================================================================
# Tablis format
#===============================================================================
class Tablis(Usrxxx):
	TAGPAT = re.compile(r"^# (.*)?:( *)(.*)$")

	#-----------------------------------------------------------------------
	def __init__(self, filename=None):
		"""Initialize a USRxxx structure"""
		super().__init__(filename)

	# ----------------------------------------------------------------------
	def addDetector(self, num, name, type_=None):
		det = super().addDetector(num, name, type_)
		det.data  = []
		det.block = None
		return det

	# ----------------------------------------------------------------------
	def open(self, filename=None):
		"""Read header information, and return the file handle"""
		if filename is not None:
			self.filename = filename
		self.f = open(self.filename, "r", encoding="utf-8", errors="replace")

	# ----------------------------------------------------------------------
	def read(self, filename=None, readdata=True):
		"""Read header information, and return the file handle"""
		self.reset()
		self.open(filename)

		name  = "Detector"
		det   = None
		idx   = 1		# gnuplot index
		empty = 0		# continuous empty line count 2*empty lines = 1-line :)
		blockid = 0

		# Load detector file
		for line in self.f:
			line = line.strip()
			if line=="":
				empty += 1
				if empty == 2:
					empty = 0
					idx  += 1
					det   = None
				continue
			empty = 0
			if line[0]=="#":
				if line.startswith("# Detector"):
					m = _detectorPattern.match(line)
					name = m.group(1) if m else "unknown"
					det  = self.addDetector(idx, name)
					blockid = 0
				elif line.startswith("# Block"):
					m = _blockPattern.match(line)
					blkname = m.group(1) if m else ""
					det       = self.addDetector(idx,f"{name} {blkname}")
					det.block = blockid
					blockid  += 1
				elif det is None:
					#name = line[1:].strip()
					#det  = self.addDetector(idx, name)
					blockid = 0
				else:
					# Additional variables
					m = _varPat.match(line)
					if m:
						value = m.group(2)
						try:
							value = float(value)
							value = int(value)
						except ValueError:
							pass
						det.var[m.group(1)] = value
			else:
				if det is None:
					det = self.addDetector(idx,"?")
				if readdata:
					det.data.append(shlex.split(line))
		self.close()

	# ----------------------------------------------------------------------
	def readHeader(self, filename=None):
		self.read(filename, False)

	# ----------------------------------------------------------------------
	def readData(self, filename=None):
		self.read(filename, True)

#-------------------------------------------------------------------------------
# FIXME should be removed is used for MPPlot for USR1D
#-------------------------------------------------------------------------------
def tabLis(filename, detector, block=-1):
	f = open(filename,'r')
	raw_data = f.read()				# read whole file as a single line
	f.close()

	#raw_data = raw_data.split(' # Detector n:  ')	# split in to chunks (detector data)
	#if raw_data[0] == '':
	#	del raw_data[0]				# delete blank header

	dataset = raw_data.split('\n\n\n')

	if block != -1:
		datablock = dataset[detector].split('\n\n')
		part = io.StringIO(datablock[block])
	else:
		part = io.StringIO(dataset[detector])

	name = part.readline().split()[1]

	# use the fact that it's a file object to make reading the data easy
	x_bin_min, x_bin_max, x_vals, x_err = np.loadtxt(part, unpack=True)
	return name, x_bin_min, x_bin_max, x_vals, x_err  # return the columns and detector name
