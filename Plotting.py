import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from scipy.stats import norm

potential = 0 # 0: SIS, 1: MW, 2: Kepler, 3: Triaxial, 4: Harmonic

if (potential == 0) or (potential == 1):
	beta = np.sqrt(2)
elif potential == 2:
	beta = 1
	
G = 4.3009125E-3 *(1.023)**2 # pc^3/Myr^2/Msun

def mag(vec):
	return np.sqrt(np.sum(np.atleast_2d(vec)**2,axis=1))

FileNums = np.linspace(1,10,10,endpoint=True)
StreamLengths = np.zeros(np.shape(FileNums))

for i in range(len(FileNums)):
	FileN = int(FileNums[i])
	print('t=',0.4*FileN,' Gyr')
	# Loading Star Data
	t,M,x,y,z,vx,vy,vz,ax,ay,az = np.loadtxt('./test/data_'+str(FileN)+'.txt',skiprows=2,unpack=True)
	R = np.sqrt(x**2+y**2+z**2)
	# Loading Cluster Data (c denotes a cluster quantity)
	t,Mc,xc,yc,zc,vxc,vyc,vzc,axc,ayc,azc = np.loadtxt('./test/data_'+str(FileN)+'.txt',skiprows=1,max_rows=1,unpack=True)
	Rc = np.sqrt(xc**2+yc**2+zc**2)
	
	R_rel = R-Rc
	# Centre on cluster
	pos = np.array([x-xc,y-yc,z-zc]).T
	vel = np.array([vx-vxc,vy-vyc,vz-vzc]).T
	# Cluster properties
	posc = np.array([xc,yc,zc])
	velc = np.array([vxc,vyc,vzc])
	AngMomc = np.cross(posc,velc)
	# Unit vectors
	normvec = AngMomc/mag(AngMomc) # Stream Aligned z
	radvec = posc/mag(posc) # Stream Aligned x
	tanvec = np.cross(normvec,radvec) # Stream Aligned y
	
	#Rotation Matrix
	Rot = np.array([radvec,tanvec,normvec])
	
	posSA = np.zeros(np.shape(pos)) # Stream Aligned Position
	velSA = np.zeros(np.shape(vel)) # Stream Aligned Velocity
	
	print('x',Rot @ radvec)
	print('y',Rot @ tanvec)
	print('z',Rot @ normvec)
	# Rotating stream
	for j in range(len(pos)):
		posSA[j] = Rot @ pos[j]
		velSA[j] = Rot @ vel[j]
	# Straightening stream
	posSA[:,0] += Rc
	azimuth = np.arctan2(posSA[:,1],posSA[:,0])
	xs = mag(posSA)-Rc
	ys = azimuth*mag(posSA)
	zs = posSA[:,2]
	posSA = np.array([xs,ys,zs]).T
	Vc = np.dot(velc,tanvec)
	
	StreamLengths[i] = np.percentile(mag(posSA)[np.where(posSA[:,1]>0)],98) + np.percentile(mag(posSA)[np.where(posSA[:,1]<0)],98)
	SL = StreamLengths[i]
	print('StreamLength',StreamLengths[i])
	# Nbody Data from a M0 = 2e3 Msun Cluster modelled with PETAR
	#xN,yN,MN = np.loadtxt('./NbodyData/StarPosArrays_AllStars/noBHM2e3/'+str(int(FileN*25))+'Pos.txt',skiprows=1,unpack=True)
	#MN *= 2e3/np.sum(MN)
	#plt.scatter(yN,xN,c='0.5',s=2,label='Nbody')
	plt.scatter(posSA[:,1],posSA[:,0],c='tab:orange',s=2,label='model',alpha=0.5)
	plt.xlabel('y [pc]')
	plt.ylabel('x [pc]')
	plt.ylim((-250,250))
	plt.xlim((-1.4*SL/2,1.4*SL/2))
	plt.show()
	
	BinWidth = 250 # pc
	ymax = ceil(max(abs(posSA[:,1]))/BinWidth)*BinWidth
	nBins = 2*int(ymax/BinWidth)
	Edges = np.linspace(-ymax,ymax,nBins+1,endpoint=True)
	plt.figure(figsize=(9,3))
	counts,bins,bars = plt.hist(posSA[:,1]/1e3,bins=Edges/1e3,histtype='step',weights=(M/BinWidth),color='tab:blue')
	#plt.hist(yN/1e3,bins=Edges/1e3,histtype='step',weights=(MN/BinWidth),color='tab:orange')
	plt.xlabel(r'$y$ [kpc]')
	plt.ylabel(r'$\rho$ [${\rm M}_{\odot}$/pc]')
	plt.title('t = '+str(round(t/1e3,3))+' Gyr')
	plt.show()
	
	if SL/2 > 1e3:
		Mids  = (np.linspace(0,int(SL/2e3),int(SL/2e3),endpoint=False)+0.5)*1e3 # Midpoints of each 1kpc section # 500 pc + n*1000 pc 
		Fit = np.zeros((int(SL/2e3),2,2))
		Mids_rj = np.zeros(int(SL/2e3))
		for i in range(int(SL/2e3)):
			Mass = np.sum(M[np.where(abs(posSA[:,1])<Mids[i])]) + Mc
			Mids_rj[i] = (G*Mass/(2*(Vc/Rc)**2))**(1/3) # Change to 3*(Vc/R) if using Kepler potential
			mu,std = norm.fit((posSA[:,0])[np.where((posSA[:,1]<=-i*1e3)&(posSA[:,1]>-(i+1)*1e3))]) # Fit Radial Distribution of Trailing Tail
			Fit[i,0,:] = np.array([mu,std])
			mu,std = norm.fit((posSA[:,0])[np.where((posSA[:,1]>i*1e3)&(posSA[:,1]<=(i+1)*1e3))]) # Fit Radial Distribution of Leading Tail
			Fit[i,1,:] = np.array([mu,std])
		plt.figure(figsize=(8,4))
		plt.scatter(posSA[:,1],posSA[:,0],c='0.5',s=2)
		# Trailing Tail Fit
		plt.plot(-Mids,Fit[:,0,0],c='tab:blue',label='Fit')
		plt.plot(-Mids,Fit[:,0,0]+Fit[:,0,1],c='tab:blue')
		plt.plot(-Mids,Fit[:,0,0]-Fit[:,0,1],c='tab:blue')
		# Leading Tail Fit
		plt.plot(Mids,Fit[:,1,0],c='tab:blue')
		plt.plot(Mids,Fit[:,1,0]+Fit[:,1,1],c='tab:blue')
		plt.plot(Mids,Fit[:,1,0]-Fit[:,1,1],c='tab:blue')
		# Estimate
		plt.plot(-Mids,2*Mids_rj,c='tab:orange',ls='--',label='Estimate')
		plt.plot(-Mids,2*Mids_rj+2/np.pi*Mids_rj,c='tab:orange',ls='--')
		plt.plot(-Mids,2*Mids_rj-2/np.pi*Mids_rj,c='tab:orange',ls='--')
		plt.plot(Mids,-2*Mids_rj,c='tab:orange',ls='--')
		plt.plot(Mids,-2*Mids_rj+2/np.pi*Mids_rj,c='tab:orange',ls='--')
		plt.plot(Mids,-2*Mids_rj-2/np.pi*Mids_rj,c='tab:orange',ls='--')
		plt.xlabel('y [pc]')
		plt.ylabel('x [pc]')
		plt.title('t = '+str(round(t/1e3,3))+' Gyr')
		plt.xlim((-1.4*SL/2,1.4*SL/2))
		plt.ylim((-250,250))	
		plt.legend()
		plt.show()

	ax = plt.figure().add_subplot(projection='3d')
	ax.scatter(posSA[:,0],posSA[:,1],posSA[:,2],s=2,c='0.5')
	ax.scatter(0,0,0,c='r',s=50)
	ax.set_xlim((-2e3,2e3))
	ax.set_ylim((-2e3,2e3))
	ax.set_zlim((-2e3,2e3))
	ax.set_xlabel('x [pc]')
	ax.set_ylabel('y [pc]')
	ax.set_zlabel('z [pc]')
	plt.show()
	
	ax = plt.figure().add_subplot(projection='3d')
	ax.scatter(x/1e3,y/1e3,z/1e3,s=2,c='0.5')
	ax.set_xlim((-10,10))
	ax.set_ylim((-10,10))
	ax.set_zlim((-10,10))
	ax.set_xlabel('x [pc]')
	ax.set_ylabel('y [pc]')
	ax.set_zlabel('z [pc]')
	plt.show()
	
