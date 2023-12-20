import numpy as np
import matplotlib.pyplot as plt
import math


class Interfaces_temperatures:
    @classmethod
    def IT(self, d, dr_wall, dz, Tsea, Tb, u, U, rho_oil, Cp, mu, Kwax, Koil, Kpipe, z, Fw):
        ri = d / 2
        ro = ri + dr_wall
        Across = np.pi * ri ** 2
        Q = u * Across
        m = Q * rho_oil

        Re = (rho_oil * u * d) / mu
        Pr = (Cp * mu) / Koil
        Nu = 0.023 * Re ** (0.8) * Pr ** (0.3)
        hi = (Nu * Koil) / d
        thick = len(z)
        frac = len(Fw)

        # Initialization of matrices

        Kdeposit = np.zeros(frac)
        Vwax = np.zeros(frac)
        Voil = np.zeros(frac)

        for i in range(frac):
            Vwax[i] = Fw[i]
            Voil[i] = (1 - Fw[i])

        diffTd = np.zeros(frac)
        diffTwi = np.zeros(frac)
        diffTwo = np.zeros(frac)
        diffTch = np.zeros(frac)

        rd = np.zeros(thick)
        Ad = np.zeros(thick)
        Alm_d = np.zeros(thick)

        Rd = np.zeros((frac, thick))
        Rtot = np.zeros((frac, thick))
        q = np.zeros((frac, thick))
        Td = np.zeros((frac, thick))
        Twi = np.zeros((frac, thick))
        Two = np.zeros((frac, thick))
        Tcheck = np.zeros((frac, thick))
        zTd = np.zeros((frac, thick))
        zTp = np.zeros((frac, thick))

        Ai = 2 * np.pi * ri * dz
        Ao = 2 * np.pi * ro * dz
        Alm_p = (Ao - Ai) / np.log(Ao / Ai)

        R = 1 / U
        Ri = 1 / (hi * Ai)
        Rp = (ro - ri) / (Kpipe * Alm_p)
        hp = 1 / Rp
        ho = 1 / (R - 1 / hi - 1 / hp)
        Ro = 1 / (ho * Ao)

        for j in range(0, frac):

            for i in range(0, thick):

                rd[i] = ri - z[i]
                Ad[i] = 2 * np.pi * rd[i] * dz
                Alm_d[i] = (Ai - Ad[i]) / (np.log(Ai / Ad[i]))

                # Effective Medium Theory (EMT)
                a = -2 * (Vwax[j] + Voil[j])
                b = 2 * (Vwax[j] * Kwax + Voil[j] * Koil) - Voil[j] * Kwax - Vwax[j] * Koil
                c = Kwax * Koil * (Vwax[j] + Voil[j])
                k1 = -b / (2 * a) + np.sqrt(b ** 2 - 4 * a * c) / (2 * a)
                k2 = -b / (2 * a) - np.sqrt(b ** 2 - 4 * a * c) / (2 * a)

                if k1 >= 0:
                    Kdeposit[j] = k1
                else:
                    Kdeposit[j] = k2

                Rd[j, i] = (ri - rd[i]) / (Kdeposit[j] * Alm_d[i])
                Rtot[j, i] = Ri + Rd[j, i] + Rp + Ro
                q[j, i] = (Tsea - Tb) / Rtot[j, i]

                Td[j, i] = Tb + (q[j, i] * Ri)
                Twi[j, i] = Td[j, i] + (q[j, i] * Rd[j, i])
                Two[j, i] = Twi[j, i] + (q[j, i] * Rp)
                Tcheck[j, i] = Two[j, i] + (q[j, i] * Ro)

                zTd[j, i] = Td[j, i] - Twi[j, i]
                zTp[j, i] = Twi[j, i] - Two[j, i]

                diffTd[j] = ((1 - Td[j, thick - 1]) / Td[j, 0]) * 100
                diffTwi[j] = ((1 - Twi[j, thick - 1]) / Twi[j, 0]) * 100
                diffTwo[j] = ((1 - Two[j, thick - 1]) / Two[j, 0]) * 100

        return diffTd, diffTwi, diffTwo


# Assigning input values:
d = 0.3048
dr_wall = 0.012
dz = 1
Tsea = 5
Tb = 53
u = 2
U = 20
rho_oil = 750
Cp = 2300
mu = 0.5 * 10 ** (-3)

Kwax = 0.25
Koil = 0.1
Kpipe = 20

z = np.linspace(0.00001, 0.02, 2000)
Fw = [0.05, 0.2, 0.4, 0.5, 0.7]

It = Interfaces_temperatures.IT(d, dr_wall, dz, Tsea, Tb, u, U, rho_oil, Cp, mu, Kwax, Koil, Kpipe, z, Fw)
print(It)
