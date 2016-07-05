import matplotlib.pyplot as plt
import os, sys

def getPath():
	# Returns the current working directory
    return os.path.dirname(os.path.realpath(sys.argv[0]))

def checkdir(directory):
	# Creates a directory of a given path if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        return False
    else:
    	return True

def plotdir():
	# Creates a plots directory in the current working directory if it does not exist
	plotdirname = getPath() + '/plots'
	if not checkdir(plotdirname):
		print("\nCreated plots directory in CWD\n")

def savedir():
	# Creates a new plot folder to save images in
	curdir = getPath()
	curdir += '/plots'
	dummybool = checkdir(curdir)
	existingdir = True
	i = 1
	while existingdir:
		plotdir = curdir + '/' + str(i)
		existingdir = checkdir(plotdir)
		i += 1
	return plotdir

def plotset(title = None, xlim = None, ylim = None, xlabel = None, ylabel = None, minticks = None, eqaspect = None, sciaxis = None):
	# Allows many aspects of the plot to manipulated in one function
	if title:
		try:
			plt.title(title)
		except:
			print('Invalid title')
	if xlim:
		try:
			plt.xlim(xlim)
		except:
			print('Invalid x limits')
	if ylim:
		try:
			plt.ylim(ylim)
		except:
			print('Invalid y limits')
	if xlabel:
		try:
			plt.xlabel(xlabel)
		except:
			print('Invalid x label')
	if ylabel:
		try:
			plt.ylabel(ylabel)
		except:
			print('Invalid y label')
	if minticks:
		plt.minorticks_on()
	if eqaspect:
		plt.axes().set_aspect('equal')
	if sciaxis:
		try:
			plt.ticklabel_format(style='sci',axis=sciaxis,scilimits=(0,0))
		except:
			print('Invalid axis choice for scientific units')