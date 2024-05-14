import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
np.random.seed(1234)

potential = 1 # 0 : SIS, 1 : MW, 2 : Kepler.
if potential == 0:
	Vc = 220. * 1.023 # pc/Myr 
Halo = True
G = 4.3009125E-3 *(1.023)**2 # pc^3/Myr^2/Msun
R = 8.5 * 1e3 # pc
Mstar = 0.5 # Msun

@njit
def acc_SIS(pos_t):
	rs = np.sqrt(np.sum(pos_t**2,axis=1))
	return ((-Vc**2 / rs) * (pos_t.T/rs)).T

@njit
def acc_MW(pos_t):
	rs = np.sqrt(np.sum(pos_t**2,axis=1))
	xys = np.sqrt(np.sum(pos_t[:,:2]**2,axis=1))
	zs = pos_t[:,2]
	# Bulge # Bovy (2015)
	Mb = 5e9 # Msun
	ab = 0.44e3 # pc
	a_bulge = np.zeros(np.shape(pos_t))
	a_bulge[:,:2] = (-G*Mb*xys/((rs+ab)**2*rs) * pos_t[:,:2].T/xys).T
	a_bulge[:,2] =  -G*Mb*zs/((rs+ab)**2*rs)
	# Disk # Bovy (2015)
	ad = 3e3
	bd = 280
	Md = 6.8e10
	z2b2 = np.sqrt(zs**2+bd**2)
	a_disk = np.zeros(np.shape(pos_t))
	a_disk[:,:2] = (-G*Md*xys/(xys**2+(ad+z2b2)**2)**(3/2) * pos_t[:,:2].T/xys).T
	a_disk[:,2] = -G*Md*zs*(ad+z2b2)/((xys**2+(ad+z2b2)**2)**(3/2) * z2b2)
	if Halo == False:
		return a_bulge+a_disk
	else:
		# DM halo # Based on Barros (2016) but vh and rh altered to better match MWpotential2014 from Bovy (2015)
		vh = 160*1.023 # pc/Myr
		rh = 5.4e3 # pc
		a_halo = (-vh**2*rs/(rs**2+rh**2) * (pos_t.T/rs)).T
		return a_bulge+a_disk+a_halo
        
@njit
def acc_kepler(pos_t):
	rs = np.sqrt(np.sum(pos_t**2,axis=1))
	Mg = 9.565411991060036e10 # Galactic mass to give Vc = 220 km/s at R=8.5kpc
	return ((-G*Mg / rs**2) * (pos_t.T/rs)).T

@njit
def calc_acc(pos_t,Mc,rj):
	N = len(pos_t)
	acc_array = np.zeros((N,3))
	# Uncomment if you want to include star-star interactions
	'''
	F_grid = np.zeros((N,N,3))
	for j in range(N):
		for k in range(N):
			if j == k:
				pass
			elif j > k:
				pass
			else: 
				F_grid[j,k,:] = -(G*Ms[k]*Ms[j]/np.sum((pos_t[j]-pos_t[k])**2+(1e-7)**2)**(3/2) * (pos_t[j]-pos_t[k]))
				F_grid[k,j,:] = -F_grid[j,k,:] 
	acc_array = (np.sum(F_grid,axis=1).T / Ms[:N]).T
	'''
	if potential == 0:
		acc_galaxy = acc_SIS(pos_t)
	elif potential == 1:
		acc_galaxy = acc_MW(pos_t)
	elif potential == 2:
		acc_galaxy = acc_kepler(pos_t)
	pos_rel_to_cluster = pos_t - pos_cluster
	r_rel_to_cluster = np.sqrt(np.sum(pos_rel_to_cluster**2,axis=1))
	a = (2**(2/3)-1)**(1/2) * 0.15 * rj # Plummer sphere scale radius
	acc_cluster = (-G*Mc*r_rel_to_cluster*(r_rel_to_cluster**2+a**2)**(-3/2) * (pos_rel_to_cluster.T/r_rel_to_cluster)).T # acceleration due to cluster
	acc_array += acc_galaxy + acc_cluster
	return acc_array

Rs = np.zeros((1001,3))
Rs[:,0] = np.linspace(0.1,100e3,1001,endpoint=True)

if potential == 0:
	Vcs = np.sqrt(-acc_SIS(np.atleast_2d(Rs))[:,0]*Rs[:,0])
elif potential == 1:
	Vcs = np.sqrt(-acc_MW(np.atleast_2d(Rs))[:,0]*Rs[:,0])
elif potential == 2:
	Vcs = np.sqrt(-acc_kepler(np.atleast_2d(Rs))[:,0]*Rs[:,0])

if potential != 0:
	Vc = np.interp(R,Rs[:,0],Vcs)
AngVel = Vc/R # rad/Myr
print(Vc)
plt.plot(Rs[:,0]/1e3,Vcs)
plt.xlabel(r'$R_{\rm G}$ [kpc]')
plt.ylabel(r'$V_{\rm c}$ [pc/Myr]')
plt.scatter(R/1e3,Vc,c='r')
plt.savefig('./RotCurve.png',dpi=300,bbox_inches='tight')
plt.close

# Cluster Params
M0 = 2e3 # Msun
Mdep = 1/3 # equivalent to 1-y in Gieles and Gnedin 2023
td = 3e3 # Myr
M = 1 * M0
pos_cluster = np.array([R,0,0])
vel_cluster = np.array([0,Vc,0])
r_cluster = np.sqrt(np.sum(pos_cluster**2))

if potential == 0:
	acc_cluster = acc_SIS(np.atleast_2d(pos_cluster))[0]
elif potential == 1:
	acc_cluster = acc_MW(np.atleast_2d(pos_cluster))[0]
elif potential == 2:
	acc_cluster = acc_kepler(np.atleast_2d(pos_cluster))[0]

# Star ICs
Nstars = int(M0/Mstar)
Ms = np.ones(Nstars)*Mstar # Star Masses # single mass to not have to deal with low mass stars preferentially escaping
Mts = np.cumsum(Ms[::-1])[::-1] # Cluster Mass at escape times 
tesc = td * M0**(Mdep-1)*(M0**(1-Mdep) - (Mts)**(1-Mdep)) # escape times
# Add rj for each Potential
if (potential == 0) or ((potential == 1) and (Halo == True)):
	rjs = (G*Mts/(2*(Vc/R)**2))**(1/3) # Escape radii
elif (potential == 2) or ((potential == 1) and (Halo == False)):
	rjs = (G*Mts/(3*(Vc/R)**2))**(1/3) # Calc
rhs = 0.15 * rjs # Half-mass radii # 0.15 from Henon 1961
xes = 1 * rjs # Escape radius 
sigs = (G*Mts/(6*(2**(2/3)-1)**(1/2)*rhs))**(1/2) * (1+(xes/rhs)**2 / (2**(2/3)-1))**(-1/4) # vel dis at xes Heggie and Hut 2003
dvrs = np.random.normal(0,1,size=Nstars) * sigs
dvts = np.random.normal(0,1,size=Nstars) * sigs
dvzs = np.random.normal(0,1,size=Nstars) * sigs

# every other star goes into leading tail
# First star goes into trailing tail
xes[1::2] *= -1
dvrs[1::2] *= -1
dvts[1::2] *= -1
dvzs[1::2] *= -1
dvzs = np.zeros(Nstars)

Mts = np.append(Mts,[0],axis=0) # Mts = 0 at t=td
rjs = np.append(rjs,[0],axis=0) # rjs = 0 at t=td
    
# at t = 0 only the first star (index 0) is escaping
pos = np.array([pos_cluster+np.array([xes[0],0,0])])
vel = np.array([vel_cluster+np.array([dvrs[0],AngVel*xes[0]+dvts[0],dvzs[0]])])
acc = calc_acc(pos,Mts[1],rjs[1])

t = 0 # initial time
T = 1e3 # run time
Nout = 1 # current number of output files
tout = 100 # output interval
dt = 1e-3 # timestep

hdr = '// t [Myr] // Mstar [Msun] // x [pc] // y [pc] // z [pc] // vx [pc/Myr] // vy [pc/Myr] // vz [pc/Myr] // ax [pc/Myr^2] // ay [pc/Myr^2] // az [pc/Myr^2] // [First Line is Cluster Values]'
stars_out = np.array([np.ones(len(pos))*0,np.ones(len(pos))*Mstar,pos[:,0],pos[:,1],pos[:,2],vel[:,0],vel[:,1],vel[:,2],acc[:,0],acc[:,1],acc[:,2]]).T
cluster_out = np.array([np.array([0,M0,pos_cluster[0],pos_cluster[1],pos_cluster[2],vel_cluster[0],vel_cluster[1],vel_cluster[2],acc_cluster[0],acc_cluster[1],acc_cluster[2]])])
out = np.concatenate((cluster_out,stars_out),axis=0)
np.savetxt('./test/data_'+str(0)+'.txt',out,header=hdr)

for i in tqdm(range(int(T/dt)+1)):
	# Number of escaped stars
	Nesc = len(tesc[np.where(tesc<=t)])
	# Updating Cluster Vals
	vel_temp = vel_cluster+acc_cluster*dt/2
	pos_cluster = pos_cluster+vel_temp*dt
	r = np.sqrt(np.sum(pos_cluster**2))
	if potential == 0:
		acc_cluster = acc_SIS(np.atleast_2d(pos_cluster))[0]
	elif potential == 1:
		acc_cluster = acc_MW(np.atleast_2d(pos_cluster))[0]
	elif potential == 2:
		acc_cluster = acc_kepler(np.atleast_2d(pos_cluster))[0]
	vel_cluster = vel_temp + acc_cluster * dt/2
	if Nesc - len(pos) != 0:
		# Adding new escaped star
		index = len(pos)
		r_UnitVec = pos_cluster/r
		N_UnitVec = np.cross(pos_cluster,vel_cluster)/np.sqrt(np.sum(np.cross(pos_cluster,vel_cluster)**2))
		t_UnitVec = np.cross(N_UnitVec,r_UnitVec)
		new_element = np.array([pos_cluster+xes[index]*r_UnitVec])
		pos = np.concatenate((pos,new_element),axis=0)
		new_element = np.array([vel_cluster+(AngVel*xes[index]+dvts[index])*t_UnitVec+dvrs[index]*r_UnitVec+dvzs[index]*N_UnitVec])
		vel = np.concatenate((vel,new_element),axis=0)
		new_element = calc_acc(np.array([pos[index]]),Mts[index],rjs[index])
		acc = np.concatenate((acc,new_element),axis=0)
	# Updating Star Vals
	vel_temp = vel+acc*dt/2
	pos = pos+vel_temp*dt
	r = np.sqrt(np.sum(pos**2,axis=1))
	acc = calc_acc(pos,Mts[Nesc],rjs[Nesc])
	vel = vel_temp + acc * dt/2
	if t >= tout*Nout:
		stars_out = np.array([np.ones(len(pos))*t,np.ones(len(pos))*Mstar,pos[:,0],pos[:,1],pos[:,2],vel[:,0],vel[:,1],vel[:,2],acc[:,0],acc[:,1],acc[:,2]]).T
		cluster_out = np.array([np.array([t,Mts[Nesc],pos_cluster[0],pos_cluster[1],pos_cluster[2],vel_cluster[0],vel_cluster[1],vel_cluster[2],acc_cluster[0],acc_cluster[1],acc_cluster[2]])])#.T
		out = np.concatenate((cluster_out,stars_out),axis=0)
		np.savetxt('./test/data_'+str(Nout)+'.txt',out,header=hdr)
		Nout += 1
	t += dt
