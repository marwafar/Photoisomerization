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
 potgs = np.zeros(tot_ygrid)
 potex = np.zeros(tot_ygrid)
 coupl = np.zeros(tot_ygrid)

 theta,dtheta = FFT_grid()

 for i in range(tot_ygrid):
  potgs[i] = 0.5*W_0*(1.0-math.cos(theta[i]))
  potex[i] = E_1-0.5*W_1*(1.0-math.cos(theta[i]))
  g0_phi=math.exp(-(theta[i]-0.52*math.pi)**2/(0.0049*(math.pi)**2))
  g1_phi=math.exp(-(theta[i]+0.52*math.pi)**2/(0.0049*(math.pi)**2))
  coupl[i] = A*(g0_phi+g1_phi)

 return potgs,potex,coupl
#-------------------------------------
def adiabatic_potential():
 
 # This function return the transformation
 # matrix from diabatic to adiabatic 

 potgs,potex,coupl=diabatic_potential()
 mu_eg=dipole()

 mu_di=np.zeros(tot_ygrid)
 D_ad = np.zeros((2,2))
 D_di = np.zeros((2,2))
 E_di = np.zeros((2,2))
 di_to_ad=np.zeros((tot_ygrid,2,2))

 for y in range(tot_ygrid):
  E_di[0,0]=potgs[y]
  E_di[1,1]=potex[y]
  E_di[1,0]=coupl[y]
  E_di[0,1]=coupl[y]

  E_ad,U = diag(E_di)
  di_to_ad[y,:,:]=np.copy(U)

 # Convert transition dipole moment to diabatic representation.
  D_ad[0,0]=0.0
  D_ad[1,1]=0.0
  D_ad[1,0]=mu_eg[y]
  D_ad[0,1]=mu_eg[y]

  D_di=np.matmul(np.matmul(U,D_ad) , U.T)
  mu_di[y]= D_di[1,0]

 return di_to_ad,mu_di
#--------------------------------------------
def dipole():

 # This function compute the transition dipole moment in atomic unit.

# dipl=open("transition-dipole.txt","w")
 theta,dtheta=FFT_grid()

 mu_eg=np.zeros(tot_ygrid)
 for i in range(tot_ygrid):
  g0_phi=math.exp(-(theta[i]-0.52*math.pi)**2/(0.0049*(math.pi)**2))
  g1_phi=math.exp(-(theta[i]+0.52*math.pi)**2/(0.0049*(math.pi)**2))
  mu_eg[i]=10*(1-g0_phi-g1_phi)/2.5412

#  dipl.write(str(theta[i]) + " "+str(mu_eg[i])+"\n")

# dipl.close()

 return mu_eg
#----------------------------------------------
def kinetic_energy_operator(coef_gs,coef_ex):
 
 # This function return the kinetic energy opertaor in DVR basis.
 dif2mat_FFT = second_derivative_FFT()

 KEO_FFT_gs = np.zeros((tot_ygrid,n_fock),dtype=complex)
 KEO_FFT_ex = np.zeros((tot_ygrid,n_fock),dtype=complex)

 for n in range(n_fock):
  for j in range(tot_ygrid):
   for m in range(tot_ygrid):
    KEO_FFT_gs[j,n]+=dif2mat_FFT[j,m]*coef_gs[m,n]
    KEO_FFT_ex[j,n]+=dif2mat_FFT[j,m]*coef_ex[m,n]

 return KEO_FFT_gs,KEO_FFT_ex
#-----------------------------------------------
def potential_energy_operator(coef_gs,coef_ex):

 # This function return the potential energy operator in the DVR basis.

 potgs,potex,coupl= diabatic_potential()

 PEO_gs=np.zeros((tot_ygrid,n_fock),dtype=complex)
 PEO_ex=np.zeros((tot_ygrid,n_fock),dtype=complex)
 diab_coupl_01=np.zeros((tot_ygrid,n_fock),dtype=complex)
 diab_coupl_10=np.zeros((tot_ygrid,n_fock),dtype=complex)

 for n in range(n_fock):
  for j in range(tot_ygrid):
   PEO_gs[j,n]=(potgs[j]+(n+0.5)*omega_c)*coef_gs[j,n] 
   PEO_ex[j,n]=(potex[j]+(n+0.5)*omega_c)*coef_ex[j,n]
   diab_coupl_01[j,n] = coupl[j]*coef_ex[j,n]
   diab_coupl_10[j,n] = coupl[j]*coef_gs[j,n]

 return PEO_gs, PEO_ex, diab_coupl_01, diab_coupl_10
#------------------------------------------------------
def H_interaction(coef_gs,coef_ex):

 # This function return the interaction between the 
 # molecule and the cavity.

 H_ad, mu_eg=adiabatic_potential()
 Hgs_int=np.zeros((tot_ygrid,n_fock),dtype=complex)
 Hex_int=np.zeros((tot_ygrid,n_fock),dtype=complex)

 for y in range(tot_ygrid):
  for n in range(n_fock):
   for m in range(n_fock):
    Hgs_int[y,n]+=g_c*mu_eg[y]*(math.sqrt(m)*float(m-1==n)+\
                      math.sqrt(m+1)*float(m+1==n))*coef_ex[y,m]
    Hex_int[y,n]+=g_c*mu_eg[y]*(math.sqrt(m)*float(m-1==n)+\
                      math.sqrt(m+1)*float(m+1==n))*coef_gs[y,m]

 return Hgs_int, Hex_int
#----------------------------------------------------
def equation_of_motion(coef_gs,coef_ex):

 # This function compute the numerical integration of 
 # the coeficient by using the RK4 method.

 RK1_gs = np.zeros((tot_ygrid,n_fock),dtype=complex)
 RK1_ex = np.zeros((tot_ygrid,n_fock),dtype=complex)
 RK2_gs = np.zeros((tot_ygrid,n_fock),dtype=complex)
 RK2_ex = np.zeros((tot_ygrid,n_fock),dtype=complex)
 RK3_gs = np.zeros((tot_ygrid,n_fock),dtype=complex)
 RK3_ex = np.zeros((tot_ygrid,n_fock),dtype=complex)
 RK4_gs = np.zeros((tot_ygrid,n_fock),dtype=complex)
 RK4_ex = np.zeros((tot_ygrid,n_fock),dtype=complex)

 # Save the current coefficient. 
 coef_old_gs = np.copy(coef_gs)
 coef_old_ex = np.copy(coef_ex)

 #1- RK1
 KEO_FFT_gs, KEO_FFT_ex = kinetic_energy_operator(coef_gs,coef_ex)
 PEO_gs, PEO_ex, diab_coupl_01, diab_coupl_10 = potential_energy_operator(coef_gs,coef_ex)
 Hgs_int, Hex_int=H_interaction(coef_gs,coef_ex)

 for n in range(n_fock):
  for i in range(tot_ygrid):
   RK1_gs[i,n]=(KEO_FFT_gs[i,n]+PEO_gs[i,n]+\
                diab_coupl_01[i,n]+Hgs_int[i,n])*complex(0.0,-1.0)
   RK1_ex[i,n]=(KEO_FFT_ex[i,n]+PEO_ex[i,n]+\
               diab_coupl_10[i,n]+Hex_int[i,n])*complex(0.0,-1.0)

 # Compute the new coefficient
 for n in range(n_fock):
  for i in range(tot_ygrid):
   coef_gs[i,n] = coef_old_gs[i,n]+dt/2.0*RK1_gs[i,n]
   coef_ex[i,n] = coef_old_ex[i,n]+dt/2.0*RK1_ex[i,n]

 #2- RK2
 KEO_FFT_gs, KEO_FFT_ex = kinetic_energy_operator(coef_gs,coef_ex)
 PEO_gs, PEO_ex, diab_coupl_01, diab_coupl_10 = potential_energy_operator(coef_gs,coef_ex)
 Hgs_int, Hex_int=H_interaction(coef_gs,coef_ex)

 for n in range(n_fock):
  for i in range(tot_ygrid):
   RK2_gs[i,n]=(KEO_FFT_gs[i,n]+PEO_gs[i,n]+\
                diab_coupl_01[i,n]+Hgs_int[i,n])*complex(0.0,-1.0)
   RK2_ex[i,n]=(KEO_FFT_ex[i,n]+PEO_ex[i,n]+\
               diab_coupl_10[i,n]+Hex_int[i,n])*complex(0.0,-1.0)

 # Compute the new coefficient
 for n in range(n_fock):
  for i in range(tot_ygrid):
   coef_gs[i,n] = coef_old_gs[i,n]+dt/2.0*RK2_gs[i,n]
   coef_ex[i,n] = coef_old_ex[i,n]+dt/2.0*RK2_ex[i,n]

 #3- RK3
 KEO_FFT_gs, KEO_FFT_ex = kinetic_energy_operator(coef_gs,coef_ex)
 PEO_gs, PEO_ex, diab_coupl_01, diab_coupl_10 = potential_energy_operator(coef_gs,coef_ex)
 Hgs_int, Hex_int=H_interaction(coef_gs,coef_ex)

 for n in range(n_fock):
  for i in range(tot_ygrid):
   RK3_gs[i,n]=(KEO_FFT_gs[i,n]+PEO_gs[i,n]+\
                diab_coupl_01[i,n]+Hgs_int[i,n])*complex(0.0,-1.0)
   RK3_ex[i,n]=(KEO_FFT_ex[i,n]+PEO_ex[i,n]+\
               diab_coupl_10[i,n]+Hex_int[i,n])*complex(0.0,-1.0)

 # Compute the new coefficient
 for n in range(n_fock):
  for i in range(tot_ygrid):
   coef_gs[i,n] = coef_old_gs[i,n]+dt*RK3_gs[i,n]
   coef_ex[i,n] = coef_old_ex[i,n]+dt*RK3_ex[i,n]

 #4- RK4
 KEO_FFT_gs, KEO_FFT_ex = kinetic_energy_operator(coef_gs,coef_ex)
 PEO_gs, PEO_ex, diab_coupl_01, diab_coupl_10 = potential_energy_operator(coef_gs,coef_ex)
 Hgs_int, Hex_int=H_interaction(coef_gs,coef_ex)

 for n in range(n_fock):
  for i in range(tot_ygrid):
   RK4_gs[i,n]=(KEO_FFT_gs[i,n]+PEO_gs[i,n]+\
                diab_coupl_01[i,n]+Hgs_int[i,n])*complex(0.0,-1.0)
   RK4_ex[i,n]=(KEO_FFT_ex[i,n]+PEO_ex[i,n]+\
               diab_coupl_10[i,n]+Hex_int[i,n])*complex(0.0,-1.0)

 # Compute the new coefficient.
 for n in range(n_fock):
  for i in range(tot_ygrid):
   coef_gs[i,n] = coef_old_gs[i,n]+dt/6.0*(RK1_gs[i,n]+2.0*RK2_gs[i,n]+2.0*\
                RK3_gs[i,n]+RK4_gs[i,n])
   coef_ex[i,n] = coef_old_ex[i,n]+dt/6.0*(RK1_ex[i,n]+2.0*RK2_ex[i,n]+2.0*\
                RK3_ex[i,n]+RK4_ex[i,n])

 return coef_gs, coef_ex
#---------------------------------------------
if __name__ == "__main__":

 # Define the input parameters

 # The parameters for the isomerization coordinate.
 tot_ygrid=200
 red_mass=0.002806/27.21138386

 # The potential energy surface parameters
 W_0=3.56/27.21138386
 W_1=1.19/27.21138386
 E_1=2.58/27.21138386
 kappa=0.19/27.21138386
 lamda=0.19/27.21138386
 A= 0.124/27.21138386

 # the parameter of the cavity photon mode
 omega_c= E_1*0.5
 g_c = 0.04*omega_c
 n_fock = 4

 # total time step and the time step
 tot_step = 10000
 dt =0.01*41.34137333656
 out= 50

 # Read the initial wavefunction coefficient.
 coef = np.loadtxt("coef_init_gs.txt")
 #print(coef)

 coef_current_gs = np.zeros((tot_ygrid,n_fock),dtype=complex)
 coef_current_ex = np.zeros((tot_ygrid,n_fock),dtype=complex)
 coef_current_ex[:,0] = coef.astype(complex)
#----------------------------------------------
# propagation

 population_di_gs=open("diabatic-pop_gs.txt","w+")
 population_di_ex=open("diabatic-pop_ex.txt","w+")
 population_ad_gs=open("adiabatic-pop_gs.txt","w+")
 population_ad_ex=open("adiabatic-pop_ex.txt","w+")
 cis=open("cis-gs-ex.txt","w+")
 trans=open("trans-gs-ex.txt","w+")

 theta,dtheta=FFT_grid()
 di_to_ad,mu = adiabatic_potential()

 for step in range(tot_step):
#  print(step)
  time_fs= dt*step/41.34137333656

  output=int(step/out)*out
  if (step==output):
   # Analysis.

   # 1- Compute the diabatic ES population.
   diab_pop_ex=np.zeros((n_fock),dtype=complex)
   diab_pop_gs=np.zeros((n_fock),dtype=complex)

   for n in range(n_fock):
    diab_pop_gs[n]=np.dot(coef_current_gs[:,n],np.conj(coef_current_gs[:,n]))
    diab_pop_ex[n]=np.dot(coef_current_ex[:,n],np.conj(coef_current_ex[:,n]))

   diab_pop_gs=np.real(diab_pop_gs)
   diab_pop_ex=np.real(diab_pop_ex)
   population_di_gs.write(str(time_fs)+" " + " ".join(diab_pop_gs.astype(str))+"\n")
   population_di_ex.write(str(time_fs)+" " + " ".join(diab_pop_ex.astype(str))+"\n")

   # 2-Compute the adiabatic ES population.
   coef_gs_ad=np.zeros((tot_ygrid,n_fock),dtype=complex)
   coef_ex_ad=np.zeros((tot_ygrid,n_fock),dtype=complex)
   ad_pop_ex=np.zeros((n_fock),dtype=complex)
   ad_pop_gs=np.zeros((n_fock),dtype=complex)

   for n in range(n_fock):
    for i in range(tot_ygrid):
     coef_gs_ad[i,n]=coef_current_gs[i,n]*di_to_ad[i,0,0]+\
                    coef_current_ex[i,n]*di_to_ad[i,1,0]
     coef_ex_ad[i,n]=coef_current_gs[i,n]*di_to_ad[i,0,1]+\
                    coef_current_ex[i,n]*di_to_ad[i,1,1]

    ad_pop_gs[n]=np.dot(coef_gs_ad[:,n],np.conj(coef_gs_ad[:,n]))
    ad_pop_ex[n]=np.dot(coef_ex_ad[:,n],np.conj(coef_ex_ad[:,n]))

   ad_pop_gs=np.real(ad_pop_gs)
   ad_pop_ex=np.real(ad_pop_ex)
   population_ad_gs.write(str(time_fs)+" " + " ".join(ad_pop_gs.astype(str))+"\n")
   population_ad_ex.write(str(time_fs)+" " + " ".join(ad_pop_ex.astype(str))+"\n")

   # 4- Compute the probability of cis/trans isomer. 
   cis_gs=np.zeros((n_fock),dtype=complex)
   cis_ex=np.zeros((n_fock),dtype=complex)
   trans_gs=np.zeros((n_fock),dtype=complex)
   trans_ex=np.zeros((n_fock),dtype=complex)

   for n in range(n_fock):
    for y in range(tot_ygrid):
     if theta[y]>= -0.5*math.pi and theta[y]<= 0.5*math.pi:
      cis_gs[n]+=coef_gs_ad[y,n]*coef_gs_ad[y,n].conjugate()
      cis_ex[n]+=coef_ex_ad[y,n]*coef_ex_ad[y,n].conjugate()
     elif theta[y]<-0.5*math.pi or theta[y]>0.5*math.pi:
      trans_gs[n]+=coef_gs_ad[y,n]*coef_gs_ad[y,n].conjugate()
      trans_ex[n]+=coef_ex_ad[y,n]*coef_ex_ad[y,n].conjugate()

   trans_gs=np.absolute(trans_gs)
   trans_ex=np.absolute(trans_ex)
   cis_gs=np.absolute(cis_gs)
   cis_ex=np.absolute(cis_ex)
   cis.write(str(time_fs)+" " + " ".join(cis_gs.astype(str))+" ")
   cis.write(" ".join(cis_ex.astype(str))+"\n")
   trans.write(str(time_fs)+" " + " ".join(trans_gs.astype(str))+" ")
   trans.write(" ".join(trans_ex.astype(str))+"\n")

  # Compute the new coef at time t.
  coef_new_gs,coef_new_ex = equation_of_motion(coef_current_gs,coef_current_ex)
  coef_current_gs = np.copy(coef_new_gs)
  coef_current_ex = np.copy(coef_new_ex)

 population_di_gs.close()
 population_di_ex.close()
 population_ad_gs.close()
 population_ad_ex.close()
 cis.close()
 trans.close()

