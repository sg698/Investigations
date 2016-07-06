import numpy as np


def normal(mu,sigma,x):
	p = np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
	p = p/np.max(p)
	return p

x = np.linspace(564.9,565.1,1000)
y = normal(565,0.01,x)

ydata = y + 0.2*np.random.normal(size=len(x))

writeout = ''
for i in range(len(x)):
	writeout = writeout+str(x[i])+' '+str(ydata[i])+'\n'

with open('data_exercise.dat','w') as f:
	f.write(writeout)
	f.close()
