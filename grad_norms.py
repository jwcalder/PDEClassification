import numpy as np
import matplotlib.pyplot as plt
from pde import wave_eq_solver
from pde import heat_eq_solver
from pde import random_source
from scipy import ndimage

n = 64
wave = wave_eq_solver()
heat = heat_eq_solver(n)
b = 5000
sigma = 3
f = random_source(b,n,sigma)
u_wave = wave.solve(f)
u_heat = heat.solve(f)

grad_wave = np.sqrt(np.sum((u_wave[:,:,1:] - u_wave[:,:,:-1])**2,axis=(1,2)))
grad_heat = np.sqrt(np.sum((u_heat[:,:,1:] - u_heat[:,:,:-1])**2,axis=(1,2)))

plt.hist(grad_wave,label='Wave Equation')
plt.hist(grad_heat,label='Heat Equation')
plt.legend(loc='upper right')
plt.title('L2 norm of gradient')
plt.show()

