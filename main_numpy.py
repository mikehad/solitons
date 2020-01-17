from verlet_numpy import verlet
import time


def main():
    system= verlet()
    system.velocity_verlet()
    #system.position_graph()
    system.modes_graph()
    #system.energy_graph()



main()
