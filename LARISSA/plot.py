#!/usr/bin/python
import sys
from pylab import show, plot
from numpy import loadtxt, transpose
data=loadtxt( sys.argv[1] )
xdata=transpose(data)[0] 
ydata=transpose(data)[1]
plot( xdata, ydata )
show()
