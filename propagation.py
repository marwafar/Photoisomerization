import math
import numpy as np
import scipy as sp
from numpy import linalg as LA
#---------------------------------
def diag(M):

 # This function returns the eigenvalues and eigenvectors.
 E,V= LA.eigh(M)
 return E,V
#---------------------------------
def x_Hermit():

 # This function retuen the grid points for Hermit polynomial.
 x_ij = np.zeros((tot_xgrid,tot_xgrid))
 for row in range(tot_xgrid):
  for colm in range(tot_xgrid):
   x_ij[row,colm] = math.sqrt(float(row+1)/(2.0*mass*freq))* \
                float(row==colm-1) + x_eq * float(row==colm)+ math.sqrt\
                    (float(row)/(2.0*mass*freq))*float(row==colm+1)

 x_i,vect = diag(x_ij)
 return x_i, vect
#----------------------------------
def weight_Hermit():

 # This function returs the weight for each grid point x_i.
 w_i = np.zeros(tot_xgrid)
 x_i,vect = x_Hermit()
 for i in range(tot_xgrid):
  w_i[i] = ((mass*freq/math.pi)**(-0.25)*math.exp(0.5*mass*\
              freq*(x_i[i]-x_eq)**2)*vect[0,i])**2

 return w_i
#---------------------------------
def second_derivavtive_Hermit():

 # This function returns the second derivavtive matrix.
 dif2mat = np.zeros((tot_xgrid,tot_xgrid))
 x_i,vect=x_Hermit()
 for row in range(tot_xgrid):
  for colm in range(tot_xgrid):
   for n in range(tot_xgrid):
    dif2mat[row,colm] += vect[n,row] * (n+0.5) * vect[n,colm]* -2.0 * mass * freq
   dif2mat[row,colm] +=  mass**2 * freq**2 *(x_i[row]-x_eq)**2 * float(row==colm)

 return dif2mat
#--------------------------------
def second_derivative_FFT():

 # This function return the second derivavtive 
 dif2mat= np.zeros((tot_ygrid,tot_ygrid))

 dtheta = (2.0*math.pi)/(tot_ygrid+1)
 K=math.pi/dtheta

 for row in range(tot_ygrid):
  for colm in range(tot_ygrid):
   dif2mat[row,colm]=float(row==colm)*(0.5*red_mass)*(K**2/3.0)*(1+(2.0/(tot_ygrid)**2))
   if row!=colm:
    dif2mat[row,colm]=float(row!=colm)*(0.5*red_mass)*(2.0*K**2/(tot_ygrid)**2)*((-1)**(colm-row)/\
                        (math.sin(math.pi*(colm-row)/(tot_ygrid))**2))

 return dif2mat
#---------------------------------
def FFT_grid():

 # This function return the FFT grid point.
 theta = np.zeros(tot_ygrid)
 dtheta = (2.0*math.pi)/(tot_ygrid+1)
 iy=0
 for igrid in range(-tot_ygrid/2,tot_ygrid/2):
  theta[iy] = float(igrid)*dtheta
  iy+=1

 return theta,dtheta
#--------------------------------
def diabatic_potential():

 # This function returns the diabatic potentail.
 potgs = np.zeros(total_grid)
 potex = np.zeros(total_grid)
 coupl = np.zeros(tot_xgrid)
 
 x_i,vect = x_Hermit()
 theta,dtheta = FFT_grid()

# diab = open("PES_diabatic.txt","w")
 igrid=0
 for i in range(tot_xgrid):
  coupl[i] = lamda * x_i[i] 
  for j in range(tot_ygrid):
   potgs[igrid] = 0.5*W_0*(1.0-math.cos(theta[j]))+0.5*freq*(x_i[i])**2
   potex[igrid] = E_1 - 0.5*W_1*(1-math.cos(theta[j])) + 0.5*freq*(x_i[i])**2 + kappa*x_i[i]
#   diab.write(str(x_i[i]) + " " + str(theta[j]) + " " + str(potgs[igrid]) + " " + str(potex[igrid]))
#   diab.write("\n")

   igrid+=1

#  diab.write("\n")

# diab.close()
 return potgs,potex,coupl
#------------------------------------------
def adiabatic_potential():

 #This function compute the transformation 
 # matrix from diabatic to adiabatic
 
 potgs,potex,coupl=diabatic_potential()

 E_di = np.zeros((2,2))
 di_to_ad=np.zeros((total_grid,2,2))

 igrid=0
 for i in range(tot_xgrid):
  for j in range(tot_ygrid):
   E_di[0,0]=potgs[igrid]
   E_di[1,1]=potex[igrid]
   E_di[0,1]=coupl[i]
   E_di[1,0]=coupl[i]

   E_ad,U = diag(E_di)
   di_to_ad[igrid,:,:]=np.copy(U)
   igrid+=1   

 return di_to_ad
#---------------------------------------------------
def kinetic_energy_operator(coef_gs,coef_ex):

 # This function return the kinetic energy opertaor in DVR basis.
 dif2mat_HO = second_derivavtive_Hermit()
 dif2mat_FFT = second_derivative_FFT()

 KEO_HO_gs = np.zeros((total_grid),dtype=complex) 
 KEO_HO_ex = np.zeros((total_grid),dtype=complex)
 KEO_FFT_gs = np.zeros((total_grid),dtype=complex)
 KEO_FFT_ex = np.zeros((total_grid),dtype=complex) 

 igrid=0
 for i in range(tot_xgrid):
  for j in range(tot_ygrid):
   k=j
   for n in range(tot_xgrid):
    KEO_HO_gs[igrid] += -0.5*(1.0/mass)*dif2mat_HO[i,n]*coef_gs[k]   
    KEO_HO_ex[igrid] += -0.5*(1.0/mass)*dif2mat_HO[i,n]*coef_ex[k]
    k+=200
   igrid+=1 
 
 for j in range(tot_ygrid):
  igrid=j
  k=0
  for i in range(tot_xgrid):
   for m in range(tot_ygrid):
    KEO_FFT_gs[igrid] += dif2mat_FFT[j,m]*coef_gs[k]
    KEO_FFT_ex[igrid] += dif2mat_FFT[j,m]*coef_ex[k] 
    k+=1
   igrid+=200

 return KEO_HO_gs, KEO_HO_ex, KEO_FFT_gs, KEO_FFT_ex
#------------------------------------------------------
def potential_energy_operator(coef_gs,coef_ex):

 # This function return the potential energy operator in the DVR basis.
 
 potgs,potex,coupl= diabatic_potential() 

 PEO_gs=np.zeros((total_grid),dtype=complex)
 PEO_ex=np.zeros((total_grid),dtype=complex)
 diab_coupl_01=np.zeros((total_grid),dtype=complex)
 diab_coupl_10=np.zeros((total_grid),dtype=complex)

 igrid=0
 for i in range(tot_xgrid):
  for j in range(tot_ygrid):
   PEO_gs[igrid] = potgs[igrid]*coef_gs[igrid]
   PEO_ex[igrid] = potex[igrid]*coef_ex[igrid]
   diab_coupl_01[igrid] = coupl[i]*coef_ex[igrid]
   diab_coupl_10[igrid] = coupl[i]*coef_gs[igrid]
   igrid+=1

 return PEO_gs, PEO_ex, diab_coupl_01, diab_coupl_10
#------------------------------------------------------
def equation_of_motion(coef_gs,coef_ex):

 # This function compute the numerical integration of 
 # the coeficient by using the RK4 method.

 RK1_gs = np.zeros((total_grid),dtype=complex)
 RK1_ex = np.zeros((total_grid),dtype=complex)
 RK2_gs = np.zeros((total_grid),dtype=complex)
 RK2_ex = np.zeros((total_grid),dtype=complex)
 RK3_gs = np.zeros((total_grid),dtype=complex)
 RK3_ex = np.zeros((total_grid),dtype=complex)
 RK4_gs = np.zeros((total_grid),dtype=complex)
 RK4_ex = np.zeros((total_grid),dtype=complex)

 # Save the current coefficient. 
 coef_old_gs = np.copy(coef_gs)
 coef_old_ex = np.copy(coef_ex)

 # 1- RK1
 KEO_HO_gs, KEO_HO_ex, KEO_FFT_gs, KEO_FFT_ex = kinetic_energy_operator(coef_gs,coef_ex)
 PEO_gs, PEO_ex, diab_coupl_01, diab_coupl_10 = potential_energy_operator(coef_gs,coef_ex)
 
 for i in range(total_grid):
  RK1_gs[i]=(KEO_HO_gs[i]+KEO_FFT_gs[i]+PEO_gs[i]+diab_coupl_01[i])*complex(0.0,-1.0)
  RK1_ex[i]=(KEO_HO_ex[i]+KEO_FFT_ex[i]+PEO_ex[i]+diab_coupl_10[i])*complex(0.0,-1.0)
 
 # Compute the new coeficient.
 for i in range(total_grid):
  coef_gs[i] = coef_old_gs[i]+dt/2.0*RK1_gs[i]
  coef_ex[i] = coef_old_ex[i]+dt/2.0*RK1_ex[i]

 # 2- RK2
 KEO_HO_gs, KEO_HO_ex, KEO_FFT_gs, KEO_FFT_ex = kinetic_energy_operator(coef_gs,coef_ex)
 PEO_gs, PEO_ex, diab_coupl_01, diab_coupl_10 = potential_energy_operator(coef_gs,coef_ex) 

 for i in range(total_grid):
  RK2_gs[i]=(KEO_HO_gs[i]+KEO_FFT_gs[i]+PEO_gs[i]+diab_coupl_01[i])*complex(0.0,-1.0)
  RK2_ex[i]=(KEO_HO_ex[i]+KEO_FFT_ex[i]+PEO_ex[i]+diab_coupl_10[i])*complex(0.0,-1.0) 

 # Compute the new coeficient.
 for i in range(total_grid):
  coef_gs[i] = coef_old_gs[i]+dt/2.0*RK2_gs[i]
  coef_ex[i] = coef_old_ex[i]+dt/2.0*RK2_ex[i]

 # 3- RK3
 KEO_HO_gs, KEO_HO_ex, KEO_FFT_gs, KEO_FFT_ex = kinetic_energy_operator(coef_gs,coef_ex)
 PEO_gs, PEO_ex, diab_coupl_01, diab_coupl_10 = potential_energy_operator(coef_gs,coef_ex)

 for i in range(total_grid):
  RK3_gs[i]=(KEO_HO_gs[i]+KEO_FFT_gs[i]+PEO_gs[i]+diab_coupl_01[i])*complex(0.0,-1.0)
  RK3_ex[i]=(KEO_HO_ex[i]+KEO_FFT_ex[i]+PEO_ex[i]+diab_coupl_10[i])*complex(0.0,-1.0) 

 # Compute the new coeficient.
 for i in range(total_grid):
  coef_gs[i] = coef_old_gs[i]+dt*RK3_gs[i]
  coef_ex[i] = coef_old_ex[i]+dt*RK3_ex[i]

 # 4- RK4
 KEO_HO_gs, KEO_HO_ex, KEO_FFT_gs, KEO_FFT_ex = kinetic_energy_operator(coef_gs,coef_ex)
 PEO_gs, PEO_ex, diab_coupl_01, diab_coupl_10 = potential_energy_operator(coef_gs,coef_ex)

 for i in range(total_grid):
  RK4_gs[i]=(KEO_HO_gs[i]+KEO_FFT_gs[i]+PEO_gs[i]+diab_coupl_01[i])*complex(0.0,-1.0)
  RK4_ex[i]=(KEO_HO_ex[i]+KEO_FFT_ex[i]+PEO_ex[i]+diab_coupl_10[i])*complex(0.0,-1.0)
  
 # Compute the new coeficient.
 for i in range(total_grid):
  coef_gs[i] = coef_old_gs[i]+dt/6.0*(RK1_gs[i]+2.0*RK2_gs[i]+2.0*\
                RK3_gs[i]+RK4_gs[i])
  coef_ex[i] = coef_old_ex[i]+dt/6.0*(RK1_ex[i]+2.0*RK2_ex[i]+2.0*\
                RK3_ex[i]+RK4_ex[i])
 
 return coef_gs,coef_ex
#------------------------------
if __name__ == "__main__":

 # Define the input parameters

 # The paramteres for the HO coordinate.
 tot_xgrid=21
 freq=0.19/27.21138386
 mass=1.0/freq
 x_eq=0.0

 # The parameters for the isomerization coordinate.
 tot_ygrid=200
 red_mass=0.002806/27.21138386

 # The potential energy surface parameters
 W_0=3.56/27.21138386
 W_1=1.19/27.21138386
 E_1=2.58/27.21138386
 kappa=0.19/27.21138386
 lamda=0.19/27.21138386

 # total grid points
 total_grid = tot_xgrid*tot_ygrid

 # total time step and the time step
 tot_step = 7000
 dt =0.01*41.34137333656
 out= 50

 # Read the initial wavefunction coefficient.
 coef = np.loadtxt("coef_init_gs.txt") 
 #print(coef)
 coef_current_gs = np.zeros((total_grid),dtype=complex)
 coef_current_ex = coef.astype(complex)

 coef_ad_gs=np.zeros((total_grid),dtype=complex)
 coef_ad_ex=np.zeros((total_grid),dtype=complex)
#-------------------------------------------------
# propagation

 population_di=open("diabatic-pop.txt","w+")
 population_ad=open("adiabatic-pop.txt","w+")
 cis=open("cis-gs-ex.txt","w+")
 trans=open("trans-gs-ex.txt","w+")

 theta,dtheta=FFT_grid()
 di_to_ad= adiabatic_potential()
 for step in range(tot_step):
#  print(step)
  time_au=dt*step

  output=int(step/out)*out
  if (step==output):
   # Compute the diabatic and adiabatic ES population.
   diab_pop_ex=0.0
   diab_pop_gs=0.0
   ad_pop_gs=0.0
   ad_pop_ex=0.0 
 

   for igrid in range(total_grid):

    diab_pop_gs+= abs(coef_current_gs[igrid])**2
    diab_pop_ex+= abs(coef_current_ex[igrid])**2

    # Compute the coef in the adiabatic represntation.
    coef_ad_gs[igrid]=di_to_ad[igrid,0,0]*coef_current_gs[igrid]+\
                      di_to_ad[igrid,1,0]*coef_current_ex[igrid]
    coef_ad_ex[igrid]=di_to_ad[igrid,0,1]*coef_current_gs[igrid]+\
                      di_to_ad[igrid,1,1]*coef_current_ex[igrid]

    ad_pop_gs+=abs(coef_ad_gs[igrid])**2
    ad_pop_ex+=abs(coef_ad_ex[igrid])**2

   time_fs= time_au/41.34137333656
   population_di.write(str(time_fs)+ " " + str(diab_pop_gs) + " " + str(diab_pop_ex)+ "\n")
   population_ad.write(str(time_fs)+ " " + str(ad_pop_gs) + " " + str(ad_pop_ex)+ "\n")

   # Compute the probability of cis/trans isomer.
   i=0
   cis_gs=0.0
   cis_ex=0.0
   trans_gs=0.0
   trans_ex=0.0
   for x in range(tot_xgrid):
    for y in range(tot_ygrid):
     if theta[y]>= -0.5*math.pi and theta[y] <= 0.5*math.pi:
      cis_gs+=abs(coef_ad_gs[i])**2
      cis_ex+=abs(coef_ad_ex[i])**2
     elif theta[y]< -0.5*math.pi or theta[y]>0.5*math.pi:
      trans_gs+=abs(coef_ad_gs[i])**2
      trans_ex+=abs(coef_ad_ex[i])**2
     i+=1
   cis.write(str(time_fs)+ " " +str(cis_gs)+" "+str(cis_ex)+"\n")
   trans.write(str(time_fs)+ " " +str(trans_gs)+" "+str(trans_ex)+"\n")

  # Compute the new coef at time t.  
  coef_new_gs,coef_new_ex = equation_of_motion(coef_current_gs,coef_current_ex)

  coef_current_gs = np.copy(coef_new_gs)
  coef_current_ex = np.copy(coef_new_ex)

 population_di.close()
 population_ad.close()
 cis.close()
 trans.close()

