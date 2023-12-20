from Laminar_Numerical_Model import *

vz = 0.0387
R = 0.00794
L = 2
Tw = 7
Ti = 22.1
alpha = 9.8 * 10 ** -7
mu = 0.004
Va = 400
n = 1
ni = 200
nj = 200
a = 4.9 * 10 ** (-9)
b = 17.8
c = 6
z = np.linspace(0, L, ni)
r = np.linspace(0, R, nj)
T = Numerical_model_1.temperature(ni, nj, R, L, Tw, Ti, alpha, vz)

C = Numerical_model_1.concentration(ni, nj, R, L, Tw, Ti, T, mu, vz, Va, a, b, c)

#contour plots
CS = plt.contour(z,r,T,15,linewidth=1,colors='k')
CS2 = plt.contourf(z,r,T,15,cmap=plt.cm.jet)
plt.clabel(CS,fontsize=10)
plt.ylabel('r')
plt.xlabel('z')
plt.title('T(r,z)')
plt.colorbar()
plt.show()
