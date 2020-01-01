import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.linalg import eigh




class verlet(object):

    def __init__(self):
        file_handle = open("file.txt", "r")          #I get the initial conditions from the text file
        self.solitons_array= np.loadtxt(file_handle)
        file_handle.close

        self.dt = 0.001
        self.Tmax=100.
        self.numstep = int(self.Tmax/self.dt)                    #PARAMETERS
        self.time_list = np.arange(0, self.Tmax, self.dt)
        self.alpha=0.1
        self.k_spring=0.5
        self.N=self.solitons_array.shape[0]

        self.total_velocity =0. #self.center_momentum() #I Fix the drift of the center of mass(sometime I choose initial velocities where the cm velocity is zero for convinience)

        self.acceleration_array= self.acceleration()
        self.position_array= np.zeros((self.N, self.numstep))
        self.modes_array = np.zeros((self.N, self.numstep))
        self.position_array[:,0] += self.solitons_array[:,0]
        self.energy_list=[] #list for the total energy_list

        # here I create the rotation matrix for the harmonic case I assume that the modes do not change a lot
        self.tria=diags([1, -2, 1], [-1, 0, 1], shape=(self.N, self.N)).toarray()
        self.tria=self.tria*self.k_spring
        #self.tria[0,self.N-1]=self.k_spring
        #self.tria[self.N-1,0]=self.k_spring
        self.rotation_matrix= self.eigen()[1]
        self.total_energy=self.energy(0)


    def center_momentum(self):
        total_velocity= np.mean(self.solitons_array, axis=0)[1]
        return total_velocity

    #function to calculate the acceleration
    def acceleration(self):
        acceleration_list = np.zeros((self.N))
        acceleration_list[0]= self.k_spring*(self.solitons_array[1,0]-2*self.solitons_array[0,0])*(1+self.alpha*(self.solitons_array[1,0]))
        acceleration_list[self.N-1]=self.k_spring*(self.solitons_array[self.N-2,0]-2*self.solitons_array[self.N-1,0])*(1+self.alpha*(-self.solitons_array[self.N-2,0]))
        for i in range(1,self.N-1):
            acceleration_list[i]=self.k_spring*(self.solitons_array[(i+1) % self.N,0]+self.solitons_array[i-1,0]-2*self.solitons_array[i,0])*(1+self.alpha*(self.solitons_array[(i+1) % self.N,0]-self.solitons_array[i-1,0]))
        return acceleration_list

    def update_position(self):
        self.solitons_array[:,0] += self.dt*self.solitons_array[:,1]+ 0.5*self.acceleration_array*(self.dt**2)

    def update_velocity(self,acceleration):
        self.solitons_array[:,1] +=  self.dt * acceleration


    #the commented code in this function gives the total energy of the system without rotation. If you want to check it comment the 4-7th line of this function and uncomment the others.
    def energy(self,k):
        potential=np.zeros(self.N)
        kinetic=np.zeros(self.N)
        modes=np.dot(np.transpose(self.rotation_matrix),self.solitons_array)

        potential[0]=self.k_spring*modes[0,0]*(self.alpha*(modes[1,0])+1)*(-modes[0,0]+modes[1,0])
        potential[self.N-1]=self.k_spring*modes[self.N-1,0]*((-modes[self.N-2,0])+1)*(modes[self.N-2,0]-modes[self.N-1,0])
        for i in range(1,self.N-1):
            potential[i] = self.k_spring*modes[i,0]*(self.alpha*(modes[(i+1) % self.N,0]-modes[i-1,0])+1)*(modes[i-1,0]-modes[i,0]+modes[(i+1) % self.N,0])
            kinetic[i] = 0.5*(modes[i,1]**2)
        total=kinetic-potential
        self.modes_array[:,k] += total
        #kinetic = 0.5*np.sum((modes[:,1])**2)

        #for i in range(self.N):
            #potential += self.k_spring*self.solitons_array[i,0]*(self.alpha*(self.solitons_array[(i+1) % self.N,0]-self.solitons_array[i-1,0])+1)*(self.solitons_array[i-1,0]-self.solitons_array[i,0]+self.solitons_array[(i+1) % self.N,0])
        #kinetic = 0.5*np.sum((self.solitons_array[:,1])**2)
        return np.sum(total)


    # integrator
    def velocity_verlet(self):
        for k in range(self.numstep):
            self.update_position()
            acceleration_list_new = self.acceleration()
            self.update_velocity(0.5*(acceleration_list_new+self.acceleration_array))
            self.acceleration_array = acceleration_list_new
            self.energy_list.append(self.energy(k))
            self.position_array[:,k] += (self.solitons_array[:,0]-self.total_velocity*self.dt*k)

        #print(self.position_array[0,-1], self.position_array[1,-1],self.position_array[2,-1])
        print(self.energy_list[0])

    # here I am getting the eigenvectors for the rotation matrix to find the normal modes
    def eigen(self):
        w, v = eigh(self.tria)

        return w, v

    # plots
    def position_graph(self):

        plt.title('Position vs Time')
        plt.xlabel('Time ')
        plt.ylabel('Position')
        for i in range(self.N):
            plt.plot(self.time_list, self.position_array[i,:], label="mass"+ str(i))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.show()

    def energy_graph(self):
        plt.title(" Energy vs Time")
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.plot(self.time_list , self.energy_list)
        plt.show()

    def modes_graph(self):

        plt.title(" Modes vs Time")
        plt.xlabel("Time")
        plt.ylabel("Energy")
        for i in range(self.N):
            plt.plot(self.time_list, self.modes_array[i,:], label= "mode"+ str(i))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.show()
        #for i in range(16,self.N):
            #plt.plot(self.time_list, self.modes_array[i,:], label= "mode"+ str(i))
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        #plt.show()
