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
def Hamiltonian_Harmonic():
  
# This function returns the energy and the wavefunction. hbar=1
 H_ij = np.zeros((tot_xgrid,tot_xgrid))
 x_i,vect = x_Hermit()
 dif2mat = second_derivavtive_Hermit()

 # Compute the Hamiltonian.
 for row in range(tot_xgrid):
  for colm in range(tot_xgrid):
   H_ij[row,colm] = -0.5 * (1.0/mass) * dif2mat[row,colm] + \
                0.5* mass * freq**2 * (x_i[row])**2 * float(row==colm)

 energy,coef = diag(H_ij)

 return energy,coef
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
#---------------------------------
def Hamiltonian_FFT():
 # This function compute the energy and the eigenvector.

 H_ij=np.zeros((tot_ygrid,tot_ygrid))
 dif2mat = second_derivative_FFT()
 theta,dtheta= FFT_grid()

 for row in range(tot_ygrid):
  for colm in range(tot_ygrid):
   H_ij[row,colm]= dif2mat[row,colm] + float(row==colm)* 0.5*W_0*(1.0-math.cos(theta[colm]))

 energy,coef = diag(H_ij)

 return energy,coef

#---------------------------------
if __name__ == "__main__":

 # Define the input parameters

 # The paramteres for the x coordinate.
 tot_xgrid=21
 freq=0.19/27.211
 mass=1.0/freq
 x_eq=0.0

 # The parameters for the isomerization coordinate.
 tot_ygrid=400 
 red_mass=0.002806/27.211 

 # The potential energy surface parameters
 W_0=3.56/27.211
 W_1=1.19/27.211
 E_1=2.58/27.211
 kappa=0.19/27.211
 lamda=0.19/27.211

 energy_HO,coef_HO = Hamiltonian_Harmonic()
 x_i,vect = x_Hermit()
 dx = weight_Hermit()

# gs_wf= open("wf-HO-gs.txt", "w")
# erg = open("energy_HO.txt", "w")
# for i in range(tot_xgrid):
#  wf = (coef[i,0])**2/dx[i]
#  gs_wf.write(str(x_i[i]) + " " + str(wf) + "\n")
#  erg.write(str(energy[i]) + "\n")
# gs_wf.close()
# erg.close()

 energy_FFT,coef_FFT = Hamiltonian_FFT()
 theta,dtheta= FFT_grid()

# FFT_wf=open("wf-FFT-g0.txt","w")
# FFT_erg=open("energy-FFT.txt","w")
# for i in range(tot_ygrid):
#  wf = (coef[i,0])**2/dtheta
#  FFT_wf.write(str(theta[i])+ " " + str(wf) + "\n")
#  FFT_erg.write(str(energy[i])+ "\n")
 
# FFT_wf.close()
# FFT_erg.close()

# gs_wf_total = open("gs_wf_init_total.txt","w")
 coef_gs= open("coef_init_gs.txt","w")
 tot_wf=np.zeros((tot_xgrid,tot_ygrid))
 norm=0.0
 for i in range(tot_xgrid):
  for j in range(tot_ygrid):
#   tot_wf[i,j] = (coef_HO[i,0]*coef_FFT[j,0])**2/(dx[i]*dtheta)
    tot_wf[i,j] = coef_HO[i,0]*coef_FFT[j,0]
    norm+=tot_wf[i,j]**2
   
    coef_gs.write(str(tot_wf[i,j])+"\n") 
#   gs_wf_total.write(str(x_i[i]) + " " + str(theta[j]) + " " + str(tot_wf[i,j]) + "\n")
# gs_wf_total.write("\n")

# gs_wf_total.close()
 # Check normalization
# print(norm)
 coef_gs.close()
