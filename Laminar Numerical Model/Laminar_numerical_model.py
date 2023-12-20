import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from matplotlib import cm
import numpy as np

class Numerical_model_1:

    @classmethod
    def concentration(self,ni, nj, R, L, Tw, Ti, T, mu, vz, Va, a, b, c):
        dr = R / nj
        dz = L / ni

        C = np.zeros((nj, nj))

        Ci = self.solubility(Ti)
        Cw = self.solubility(Tw)
        C[:, 0] = Ci

        D_C = np.ones((nj, 1))
        D_C[0] = 0
        D_C[nj - 1] = Cw

        A_C = np.zeros((nj, nj))
        A_C[0, 0] = 1
        A_C[0, 1] = -1
        A_C[nj - 1, nj - 1] = 1

        for i in range(1, ni):

            for j in range(1, nj - 1):
                Dwo2 = self.Dwo_tot(dr, (j), i,Va,T,mu)  # Dwo i
                Dwo3 = self.Dwo_tot(dr, (j + 1), i,Va,T,mu)  # Dwo i+1

                A_C[j, j + 1] = -(j + 1 + 1) * Dwo3 / ((j + 1) * dr ** 2)  # C_T
                A_C[j, j] = ((j + 1) * Dwo2 + (j + 1 + 1) * Dwo3) / ((j + 1) * dr ** 2) + vz / dz  # A_T
                A_C[j, j - 1] = -Dwo2 / dr ** 2  # B_T

            for j in range(1, nj - 1):
                D_C[j] = C[j, i - 1] * vz / dz  # D_C

            N = np.linalg.solve(A_C, D_C)
            C[:, i] = np.ravel(N)
            C[0, i] = C[1, i]

        return C

    @classmethod
    def Dwo_tot(self,dr, j, i,Va,T,mu):
        gamma = 10.2 / Va - 0.791
        Dwo = 13.3 * 10 ** (-8) * T[j, i] ** 1.47 * mu ** gamma / (Va ** 0.71)
        Dwo_tot = Dwo

        return Dwo_tot

    @classmethod
    def solubility(self,temperature):
        a = 4.9 * 10 ** (-9)
        b = 17.8
        c = 6

        solubility = a * (temperature + b) ** c
        # solubility = 0.0007*temperature**2 + 0.0989*temperature + 1.7706
        return solubility

    @classmethod
    def temperature(self,ni, nj, R, L, Tw, Ti, alpha, vz):
        dr = R / nj
        dz = L / ni

        T = np.zeros((nj, nj))
        T[:, 0] = Ti
        A_T = np.zeros((nj, nj))
        A_T[0, 0] = 1
        A_T[0, 1] = -1
        A_T[nj - 1, nj - 1] = 1
        for j in range(1, nj - 1):
            A_T[j, j - 1] = -alpha / dr ** 2
            A_T[j, j] = vz / dz + (alpha * (2 * (j + 1) + 1)) / ((j + 1) * dr ** 2)
            A_T[j, j + 1] = -alpha * (j + 1 + 1) / ((j + 1) * dr ** 2)

        D_T = np.ones((nj, 1))
        D_T[0] = 0
        D_T[nj - 1] = Tw

        for i in range(1, ni):
            T[nj - 1, i] = T[nj - 1, i - 1]
            T[nj - 2, i] = T[nj - 1, i - 1]

            for j in range(1, nj - 1):
                D_T[j] = T[j, i - 1] * vz / dz

            N = np.linalg.solve(A_T, D_T)
            T[:, i] = np.ravel(N)
            T[0, i] = T[1, i]

        return T

    @classmethod
    def Answers(self,ni,nj,R,L,Tw,Ti,T,mu,vz,Va,a,b,c,alpha):

        T = self.temperature(ni,nj,R,L,Tw,Ti,alpha,vz)
        C = self.concentration(ni,nj,R,L,Tw,Ti,T,mu,vz,Va,a,b,c)

        response = {'T': T.tolist(),
                     'C': C.tolist()}

        return response





"""plt.plot(r, C)
plt.ylim(0,23)
plt.xlabel('r')
plt.ylabel('C')
plt.title('C vs r')
plt.show()"""

""""#create a figure and a 3D Axes
fig = plt.figure(figsize=(8,6))
ax3d = plt.axes(projection='3d')
X,Y = np.meshgrid(r,z)
Z = T
ax3d = plt.axes(projection='3d')
surf = ax3d.plot_surface(X,Y,Z,rstride=7,cstride=7,cmap='jet')
fig.colorbar(surf,ax=ax3d)
ax3d.set_title('T(r,z)')
ax3d.set_xlabel('r')
ax3d.set_ylabel('z')
ax3d.set_zlabel('T(r,z)')
plt.show()"""

#contour plots
"""CS = plt.contour(z,r,C,15,linewidth=1,co
+lors='k')
CS2 = plt.contourf(z,r,C,15,cmap=plt.cm.jet)
plt.clabel(CS,fontsize=10)
plt.ylabel('r')
plt.xlabel('z')
plt.title('C(r,z)')
plt.colorbar()
plt.show()"""

#Cartesian Plots
"""plt.plot(r,T)
plt.ylim(7,75)
plt.xlabel('r')
plt.ylabel('T')
plt.title('T vs r')
plt.show()"""
