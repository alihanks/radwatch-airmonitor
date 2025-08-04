import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

im=plt.imread('compass.jpg')
plt.imshow(im,extent=(-828,828,-828,828),origin='lower')

r=np.arange(0,500.0,0.01)
theta=1./2.*np.pi-np.pi*np.ones(r.shape)
x=r*np.cos(theta)
y=r*np.sin(theta)

plt.plot(x,y,color='r',linewidth=2)
plt.show()
