import numpy as np
import matplotlib.pyplot as plt
from pde import wave_eq_solver
from pde import heat_eq_solver

n = 64
wave = wave_eq_solver()
heat = heat_eq_solver(n)
b = 16
f = 2*np.random.rand(b,n,n)-1
u_wave = wave.solve(f)
u_heat = heat.solve(f)

plt.ion()


dx = 1/(n+1)
x = np.arange(dx,1,dx)

fig = plt.figure()
ax = fig.add_subplot(111)
line_heat, = ax.plot(x, np.zeros(n), 'r-',label='Heat Eq.')
line_wave, = ax.plot(x, np.zeros(n), 'b-',label='Wave Eq.') 
ax.legend()

for i in range(b):
    ax.set_title('Solution %d'%i)
    for k in range(n):
        print(k)
        line_heat.set_ydata(u_heat[i,k,:])
        line_wave.set_ydata(u_wave[i,k,:])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
    plt.pause(0.1)


