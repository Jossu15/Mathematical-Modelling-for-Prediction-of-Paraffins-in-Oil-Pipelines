import numpy as np
from math import exp, log


class analytical_model_1:

    @classmethod
    def tempL(self, Tsea, Ti, U, d, Cp, dz, nj, time, m):
        print(Tsea, Ti, U, d, Cp, dz, nj, time, m)
        temperatur = np.zeros((time, nj))
        temperatur[:, 0] = Ti

        for i in range(0, time):
            Tsection = Ti
            for j in range(1, nj):
                temperatur[i, j] = Tsea + (Tsection - Tsea) * exp((-U * np.pi * d * dz) / (m * Cp))

                Tsection = temperatur[i, j]

        return temperatur

    @classmethod
    def tempSub(self, rho, Pr, mu, u, y_p, q, Tw, Cp, Re):
        f = 0.305 / Re ** 0.25
        u_star = u * np.sqrt(f / 8)

        Tw_p = Tw * (rho * Cp * u_star) / (-q)

        if y_p <= 5:
            T_p = Tw_p + Pr * y_p
        elif y_p > 5 or y_p < 30:
            T_p = Tw_p - 5 * Pr + 5 * log(0.2 * Pr * y_p + (1 - Pr))
        else:
            T_p = Tw_p - 5 * Pr + 5 * log(0.2 * Pr * y_p + (1 - Pr)) - 2.5 * log(y_p / 30)

        Tsub = T_p * (-q) / (rho * Cp * u_star)
        y = (y_p * mu) / (rho * u_star)

        return Tsub, y

    @classmethod
    def deposit(self, thickness, ri, ro, hi, Kpipe, Kdep, Tsea, tempBulk, dz, nj, time, mu, Va, gamma, WAT, dt, y_p, Cp,
                Re, U, rho, Pr, u):
        Tdep = np.zeros((time, nj))
        Twin = np.zeros((time, nj))
        Two = np.zeros((time, nj))

        Tcheck = np.zeros((time, nj))
        Tsub = np.zeros((time, nj))

        Ain = 2 * np.pi * ri * dz
        Ao = 2 * np.pi * ro * dz
        Alm_p = (Ao - Ain) / log(Ao / Ain)

        Rpipe = (ro - ri) / (Kpipe * Alm_p)
        hpipe = 1 / Rpipe
        R = 1 / U
        ho = 1 / (R - 1 / hi - 1 / hpipe)
        Rout = 1 / (ho * Ao)

        Ad = np.zeros((time, nj))
        Alm_d = np.zeros((time, nj))

        Roil = np.zeros((time, nj))
        Rdep = np.zeros((time, nj))
        Rtot = np.zeros((time, nj))

        q = np.zeros((time, nj))

        rdep = np.zeros((time, nj))
        rdep[0, :] = ri - thickness
        dr = np.zeros((time, nj))

        depThick = np.zeros((time, nj))

        depThick[0, :] = thickness

        conc_dep = np.zeros((time, nj))
        conc_sub = np.zeros((time, nj))

        Roil[0, :] = 1 / (hi * Ain)
        Rtot[0, :] = Roil[0, 0] + Rpipe + Rout

        for i in range(1, time):
            for j in range(0, nj):
                if depThick[i - 1, j] <= thickness:
                    Alm_d[i, j] = 0
                    Roil[i, j] = 1 / (hi * Ain)
                    Rtot[i, j] = Roil[i, j] + Rpipe + Rout
                    q[i, j] = (Tsea - tempBulk[i, j]) / Rtot[i, j]

                    Twin[i, j] = tempBulk[i, j] + (q[i, j] * Roil[i, j])
                    Two[i, j] = Twin[i, j] + (q[i, j] * Rpipe)
                    Tcheck[i, j] = Two[i, j] + (q[i, j] * Rout)
                    Tsub[i, j] = self.tempSub(rho, Pr, mu, u, y_p, q[i, j], Twin[i, j], Cp, Re)[0]
                    dr[i, j] = self.tempSub(rho, Pr, mu, u, y_p, q[i, j], Twin[i, j], Cp, Re)[1]

                    if Tsub[i, j] < WAT:
                        conc_dep[i, j] = 0.0007 * Twin[i, j] ** 2 + 0.00989 * Twin[i, j] + 1.7706
                        conc_sub[i, j] = 0.0007 * Tsub[i, j] ** 2 + 0.00989 * Tsub[i, j] + 1.7706

                        if depThick[i, j] >= ri:
                            print('The pipe is clogged after %d hours/days at %d meter\n' % (i, j))

                    rdep[i, j] = ri - depThick[i, j]
                    Ad[i, j] = 2 * np.pi * rdep[i, j] * dz


                else:
                    Alm_d[i, j] = (Ain - Ad[i - 1, j]) / log(Ain / Ad[i - 1, j])

                    Roil[i, j] = 1 / (hi * Ad[i - 1, j])
                    Rdep[i, j] = (ri - rdep[i - 1, j]) / (Kdep * Alm_d[i, j])
                    Rtot[i, j] = Roil[i, j] + Rdep[i, j] + Rpipe + Rout

                    q[i, j] = (Tsea - tempBulk[i, j]) / Rtot[i, j]

                    Tdep[i, j] = tempBulk[i, j] + (q[i, j] * Roil[i, j])
                    Twin[i, j] = Tdep[i, j] + (q[i, j] * Rdep[i, j])
                    Two[i, j] = Twin[i, j] + (q[i, j] * Rpipe)
                    Tcheck[i, j] = Two[i, j] + (q[i, j] * Rout)
                    Tsub[i, j] = self.tempSub(rho, Pr, mu, u, y_p, q[i, j], Tdep[i, j], Cp, Re)[0]
                    dr[i, j] = self.tempSub(rho, Pr, mu, u, y_p, q[i, j], Tdep[i, j], Cp, Re)[1]

                    conc_dep[i, j] = 0.0007 * Tdep[i, j] ** 2 + 0.00989 * Tdep[i, j] + 1.7706
                    conc_sub[i, j] = 0.0007 * Tsub[i, j] ** 2 + 0.00989 * Tsub[i, j] + 1.7706

                    depThick[i, j] = depThick[i - 1, j] + (
                            13.3 * 10 ** (-12) * (Tdep[i, j] ** (1.47) * mu ** gamma) / Va ** (0.71) * (
                            conc_sub[i, j] - conc_dep[i, j]) / (dr[i, j] * 100)) * dt

                    rdep[i, j] = ri - depThick[i, j]
                    Ad[i, j] = 2 * np.pi * rdep[i, j] * dz

                if depThick[i, j] >= ri:
                    print('The pipe is clogged after %d hours/days at %d meter\n' % (i, j))

        result = {'Two': Two.tolist(),
                     'Twin': Twin.tolist(),
                     'Tdep': Tdep.tolist(),
                     'Tsub': Tsub.tolist()}

        return result
