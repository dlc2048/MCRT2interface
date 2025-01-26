#
# Copyright European Organization for Nuclear Research (CERN)
# All rights reserved
#
# Author: Vasilis.Vlachoudis@cern.ch
# Date:   10-Oct-2014

__author__ = "Vasilis Vlachoudis"
__email__  = "Vasilis.Vlachoudis@cern.ch"

import sys
import traceback

#-------------------------------------------------------------------------------
# FIXME convert to class with static members
#-------------------------------------------------------------------------------
_outunit = None		# log unit
_errunit = None		# error unit
_repeat  = set()	# avoid repeated errors
_buffer  = None		# buffered output

WARNING_PREFIX = ">w>"
ERROR_PREFIX   = ">e>"
WARNING        = f"{WARNING_PREFIX} Warning: "
WARNINGCONT    = " "*len(WARNING)
ERROR          = f"{ERROR_PREFIX} ERROR: "
ERRORCONT      = " "*len(ERROR)

#-------------------------------------------------------------------------------
def set(l,e=None):
	"""set logging function"""
	global _outunit, _errunit
	_outunit = l
	_errunit = e if e else _outunit

#-------------------------------------------------------------------------------
def _output(txt, repeat=True):
	global _outunit
	if not repeat and txt in _repeat: return
	if _outunit is not None:
		_outunit(txt)
	else:
		if _buffer is not None:
			_buffer.append(txt)
		print(txt)

#-------------------------------------------------------------------------------
def newline():
	_output("")

#-------------------------------------------------------------------------------
def _display(txt, prefix, cont, suppress_dup=False):
	global _errunit
	if suppress_dup and txt in _repeat: return
	for line in txt.splitlines():
		if _errunit is not None:
			_errunit(f"{prefix}{line}")
		else:
			if _buffer is not None:
				_buffer.append(f"{prefix}{line}")
			print(f"{prefix}{line}", file=sys.stderr)
		prefix = cont

#-------------------------------------------------------------------------------
# Start buffering output
#-------------------------------------------------------------------------------
def buffering():
	global _buffer
	if _buffer is not None: return
	_buffer = []

#-------------------------------------------------------------------------------
def stopBuffering():
	global _buffer
	_buffer = None

#-------------------------------------------------------------------------------
def buffer():
	return _buffer

#-------------------------------------------------------------------------------
# Print out something in the log unit
#-------------------------------------------------------------------------------
def say(*args):
	"""say/print a message"""
	_output(" ".join(map(str,args)))

#-------------------------------------------------------------------------------
def warning(*args):
	"""display a warning message"""
	_display(" ".join(map(str,args)), WARNING, WARNINGCONT)
warn = warning

#-------------------------------------------------------------------------------
def warningCont(*args):
	"""display a continuation of a warning message"""
	_display(" ".join(map(str,args)), WARNINGCONT, WARNINGCONT)
warnCont = warningCont

#-------------------------------------------------------------------------------
def rwarning(*args):
	"""display a warning message only once"""
	_display(" ".join(map(str,args)), WARNING, WARNINGCONT, True)
rwarn = rwarning

#-------------------------------------------------------------------------------
def error(*args):
	"""display an error message"""
	_display(" ".join(map(str,args)), ERROR, ERRORCONT)

#-------------------------------------------------------------------------------
def errorCont(*args):
	"""display a continuation of an error message"""
	_display(" ".join(map(str,args)), ERRORCONT, ERRORCONT)

#-------------------------------------------------------------------------------
def rerror(*args):
	"""display an error message only once"""
	_display(" ".join(map(str,args)), ERROR, ERRORCONT, True)

#-------------------------------------------------------------------------------
def repeat_clear():
	"""clear repeated messages"""
	_repeat.clear()

#-------------------------------------------------------------------------------
def null(*args):
	"""empty log function"""
	pass

#-------------------------------------------------------------------------------
# Log last exception
#-------------------------------------------------------------------------------
def exception():
	error(traceback.format_exc())
