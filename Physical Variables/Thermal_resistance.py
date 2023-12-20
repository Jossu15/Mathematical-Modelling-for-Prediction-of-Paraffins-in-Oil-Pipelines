import numpy as np
import matplotlib.pyplot as plt


class Thermal_resistance:

    @classmethod
    def TR(self, d, dz, ri, ro, kwax, koil, kpipe, u, mu, Utot_init, rho_oil, Cp, Fw, Main_matrix):

        number = len(Fw)
        step = len(Main_matrix)
        rd = np.zeros((step))

        Re = (rho_oil * u * d) / mu
        Pr = (Cp * mu) / koil

        Nu = 0.023 * Re ** 0.8 * Pr ** 0.3
        hi = (Nu * koil) / d

        kdep = np.zeros((number, 1))

        Vwax = np.zeros((number, 1))
        Voil = np.zeros((number, 1))

        Ad = np.zeros((number, step))
        Alm_d = np.zeros((number, step))

        Ri = np.zeros((number, step))
        Rd = np.zeros((number, step))

        Ri_contr = np.zeros((number, step))
        Rd_contr = np.zeros((number, step))
        Rp_contr = np.zeros((number, step))
        Ro_contr = np.zeros((number, step))

        Rtot = np.zeros((number, step))
        Utot = np.zeros((number, step))

        sum1 = np.zeros((number, step))

        Ai = 2 * np.pi * ri * dz
        Ao = 2 * np.pi * ro * dz

        Alm_p = (Ao - Ai) / np.log(Ao / Ai)

        Rp = (ro - ri) / (kpipe * Alm_p)
        hp = 1 / Rp

        Rtot_init = 1 / Utot_init
        ho = 1 / (Rtot_init - 1 / hi - 1 / hp)
        Ro = 1 / (ho * Ao)

        for i in range(0, number):
            Vwax[i] = Fw[i]
            Voil[i] = (1 - Fw[i])

        for i in range(0, number):
            a = -2 * (Vwax[i] + Voil[i])
            b = 2 * (Vwax[i] * kwax + Voil[i] * koil) - Voil[i] * kwax - Vwax[i] * koil
            c = kwax * koil * (Vwax[i] + Voil[i])
            k1 = -b / (2 * a) + np.sqrt(b ** 2 - 4 * a * c) / (2 * a)
            k2 = -b / (2 * a) - np.sqrt(b ** 2 - 4 * a * c) / (2 * a)

            if k1 >= 0:
                kdep[i] = k1
            else:
                kdep[i] = k2

            for j in range(0, step):
                rd[j] = ri - Main_matrix[j]

                Ad[i, j] = 2 * np.pi * rd[j] * dz
                Alm_d[i, j] = (Ai - Ad[i, j]) / np.log(Ai / Ad[i, j])

                Ri[i, j] = 1 / (hi * Ad[i, j])
                Rd[i, j] = (ri - rd[j]) / (kdep[i] * Alm_d[i, j])
                Rtot[i, j] = Ri[i, j] + Rd[i, j] + Rp + Ro
                Utot[i, j] = 1 / (Rtot[i, j] * Ad[i, j])
                Ri_contr[i, j] = (Ri[i, j] / Rtot[i, j]) * 100
                Rd_contr[i, j] = (Rd[i, j] / Rtot[i, j]) * 100
                Rp_contr[i, j] = (Rp / Rtot[i, j]) * 100
                Ro_contr[i, j] = (Ro / Rtot[i, j]) * 100

                sum1[i, j] = Ri_contr[i, j] + Rd_contr[i, j] + Rp_contr[i, j] + Ro_contr[i, j]

        return Ri_contr, Rd_contr, Rp_contr, Ro_contr


# Assigning input values:

d = 0.3048
dr_wall = 0.012
dz = 1
ri = d / 2
ro = ri + dr_wall
kwax = 0.25
koil = 0.1
kpipe = 20
u = 2
mu = 0.5 * 10 ** -3
Utot_init = 20
rho_oil = 750
Cp = 2300
Fw = [0.05, 0.2, 0.4, 0.5, 0.7]
Main_matrix = np.linspace(0.00005, 0.02, 400)

Ri_contr, Rd_contr, Rp_contr, Ro_contr = Thermal_resistance.TR(d, dz, ri, ro, kwax, koil, kpipe, u, mu, Utot_init,
                                                               rho_oil, Cp, Fw, Main_matrix)

list = ['5 % wax', '20 % wax', '40 % wax', '50 % wax', '70 % wax']

# Ri_contr
for i in range(len(Ri_contr)):
    plt.plot(Main_matrix, Ri_contr[i], label=list[i])

plt.xlabel('Deposit Thickness, [m]')
plt.ylabel('Ri/Rtot [%]')
plt.title('Inner Heat Resistance Contribution')
plt.xlim(min(Main_matrix), max(Main_matrix))
plt.legend()
plt.show()

# Rd_contr
for i in range(len(Rd_contr)):
    plt.plot(Main_matrix, Rd_contr[i], label=list[i])

plt.xlabel('Deposit Thickness, [m]')
plt.ylabel('Rd/Rtot [%]')
plt.title('Deposit Heat Resistance Contribution')
plt.xlim(min(Main_matrix), max(Main_matrix))
plt.legend()

# Rp_contr
for i in range(len(Rp_contr)):
    plt.plot(Main_matrix, Rp_contr[i], label=list[i])

plt.xlabel('Deposit Thickness, [m]')
plt.ylabel('Rp/Rtot [%]')
plt.title('Pipe Heat Resistance Contribution')
plt.xlim(min(Main_matrix), max(Main_matrix))
plt.legend()

# Ro_contr
for i in range(len(Ro_contr)):
    plt.plot(Main_matrix, Ro_contr[i], label=list[i])

plt.xlabel('Deposit Thickness, [m]')
plt.ylabel('Ro/Rtot [%]')
plt.title('outer Heat Resistance Contribution')
plt.xlim(min(Main_matrix), max(Main_matrix))
plt.legend()

plt.show()
