import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation



# First set up the figure, the axis, and the plot element we want to

fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(0, 50))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(0, 2, 1000)
    y = 32/np.cosh(4*(x -0.01* i))**2
    line.set_data(x, y)
    return line,

# call the animator.  blit=True means only re-draw the parts that have

anim = animation.FuncAnimation(fig, animate, init_func=init,
                           frames=200, interval=20, blit=True)

plt.show()
