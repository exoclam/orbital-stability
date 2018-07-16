import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import time
import math
import rebound

start = time.clock()

# conversion constants
year = 1./365.25 # Julian year
solar_radius_to_au = 0.00465 # AU

# But first, some functions

# conversion from mu to mA & mB
def massB(mu,mA):
    return mu*mA/(1-mu)

# calculate a_crit using equation for Holman line
def acrit(ebin,mu_bin):
    ac = 1.60+5.10*ebin+(-2.22)*ebin**2+4.12*mu_bin+(-4.27)*ebin*mu_bin+(-5.09)*mu_bin**2 + 4.61*(ebin**2)*(mu_bin**2)
    return ac

# convert from a to P and P to a with Newtonian version of Kepler's 3rd Law
def a_to_P(a,G,mA,mB):
    return ((a**3)*4*((np.pi)**2)/(G*(mA+mB)))**(1./2)

def P_to_a(P,G,mA,mB):
    return ((P**2)*G*(mA+mB)/(4*(np.pi)**2))**(1./3)

# calculate escape velocity
def escape_velocity(G,mA,mB,r):
    return np.sqrt(2*G*(mA+mB)/r)

# convert from m/s to AU/year
def convert_velocity(v):
    return float(v)*(3.154e7)/(1.496e11)

# use epsilon to more densely sample more interesting eccentricities
def epsilon_to_e(epsilon):
    return np.sqrt(1-epsilon**2)

# vanilla stellar radii, not used for now
#rA = 1.*solar_radius_to_au
#rB = 1.*solar_radius_to_au

# vanilla stellar mass
mA = 1. # in solar masses

# vanilla abin of unity
abin = 1. # in AU

# other REBOUND properties fixed to default, but left here for ease of varying
# source: http://rebound.readthedocs.io/en/latest/python_api.html
#i = 0 # fixed inclination
#omega = 0 # fixed argument of periapsis
#Omega = 0 # fixed longitude of ascending node

draws = 100000
mu = 0.1
elist = np.random.uniform(0.,0.99,draws)
aclist = []
for i in elist: # for every e
    #temp_mu = np.random.uniform(0.1,0.5) # independently and randomly draw a mu
    #mulist.append(temp_mu)
    ac_draw = np.random.uniform((2./3)*acrit(i,mu),(4./3)*acrit(i,mu)) # use that pair of draws to get a
    aclist.append(ac_draw)
features = np.vstack((elist,aclist)).T # ten of each (e,mu), paired with 10 ac's per pt

output = []
### run predictions using these two features
number_of_angles = 10
for point in features:
    fraction_stable = 0
    for angle in range(number_of_angles):
        ebin = point[0]
        ab = point[1]
        omega = np.random.uniform(0,2*np.pi)

        sim = rebound.Simulation()
        sim.units = ('yr','AU','Msun') # default units recommended by Holman & Wise
        sim.add(m=mA)
        mB = massB(mu,mA)
        sim.add(m=mB,a=abin,e=ebin) 
        sim.add(a=ab,omega=omega)
        G = sim.G

        # integrate
        sim.integrator = "ias15"
        sim.dt = 0.001*min(a_to_P(abin,G,mA,mB),a_to_P(ab,G,mA,mB))
        sim.move_to_com()

        torb = a_to_P(abin,G,mA,mB)
        Noutputs = int(1e5)
        times = np.linspace(0,int(1e4)*torb,Noutputs)
        x_0 = np.zeros(Noutputs)
        y_0 = np.zeros(Noutputs)
        x_1 = np.zeros(Noutputs)
        y_1 = np.zeros(Noutputs)
        x_2 = np.zeros(Noutputs)
        y_2 = np.zeros(Noutputs)
        vx_2 = np.zeros(Noutputs)
        vy_2 = np.zeros(Noutputs)

        particles = sim.particles
        unstable_flag = 0
        for i,j in enumerate(times):
            # check for crossing orbits
            if ab < abin*(1+ebin):
                unstable_flag = 1
                break # abort before we get to costly simulation

            sim.integrate(j, exact_finish_time=0)
            x_0[i] = particles[0].x
            y_0[i] = particles[0].y
            x_1[i] = particles[1].x
            y_1[i] = particles[1].y
            x_2[i] = particles[2].x
            y_2[i] = particles[2].y
            vx_2[i] = particles[2].vx
            vy_2[i] = particles[2].vy

            # distance from obj to CoM
            r = np.sqrt(x_2[i]**2+y_2[i]**2)

            # compute escape velocity
            v_esc = escape_velocity(G,mA,mB,r)

            # compute current velocity
            v_curr = np.sqrt(vx_2[i]**2+vy_2[i]**2)

            # take ratio of current and escape velocities
            velocity_ratio = v_curr/float(v_esc)

            if velocity_ratio > 1.: # escape
                unstable_flag = 1
                break # the abort clause

        if unstable_flag == 0: # if we survive every round of if-checks
            fraction_stable = fraction_stable + 1 # then we're stable
        
    output.append(fraction_stable/float(number_of_angles))
        
# convert output from list to array
output = np.asarray(output) 

# attach to big table
table = np.vstack((elist,aclist,output)).T 

# from when I ran this on HPC
#for row in table:
#    print (' '.join(map(str, row)))

# for when running as a pleb on single machine
np.savetxt("big_mu_10.txt",table,delimiter=" ")

# for after training model
#np.savetxt("test_mu_10.txt",table,delimiter=" ")

end = time.clock()
elapsed = end-start
print ("time elapsed: ", elapsed)

quit()



