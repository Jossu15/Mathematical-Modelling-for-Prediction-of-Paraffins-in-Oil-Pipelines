import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def temperature_calc(ni, nj, ri, rd, ro, nu, Tsea, Ti,
                     alpha, dr, dz, Fw, hi, ho, kwax, koil, kpipe, Pr, u, Re):
    T = np.zeros((nj, ni))
    T[:, 0] = Ti
    T[nj - 1, 0] = Tw(Tsea, Ti, dz, Fw, hi, ho, kwax, koil, kpipe, rdnew[0, 0], ri, ro)
    Tb = np.zeros(ni)
    Tb[0] = Ti

    A_T = np.zeros((nj, nj))
    A_T[nj - 1, nj - 1] = 1

    A_T[0, 0] = 1
    A_T[0, 1] = -1

    for j in range(1, nj - 1):
        alpha1 = alpha_tot(dr * (j - 1))
        alpha2 = alpha_tot(dr * (j))
        alpha3 = alpha_tot(dr * (j + 1))
        vz = Velocity(j * dr, ri, nu, u, Re)

        A_T[j, j - 1] = -1 / (2 * j * dr ** 2) * (j * alpha2 + (j - 1) * alpha1)  # C_T
        A_T[j, j] = (vz / dz) + 1 / (2 * j * dr ** 2) * (
                2 * j * alpha2 + (j + 1) * alpha3 + (j - 1) * alpha1)  # A_T
        A_T[j, j + 1] = -1 / (2 * j * dr ** 2) * ((j + 1) * alpha3 + (j) * alpha2)  # B_T

    D_T = np.ones((nj, 1))
    D_T[0] = 0

    for i in range(1, ni):
        T[nj - 1, i] = T[nj - 1, i - 1]
        T[nj - 2, i] = T[nj - 1, i - 1]

        for j in range(1, nj - 1):
            vz = Velocity(j * dr, ri, nu, u, Re)
            D_T[j] = T[j, i - 1] * vz / dz

        D_T[nj - 1] = Tw(Tsea, Tb[i - 1], dz, Fw, hi, ho, kwax, koil, kpipe, rdnew[0, i], ri, ro)

        N = np.linalg.solve(A_T, D_T)
        T[:, i] = np.ravel(N)
        T[0, i] = T[1, i]
        Tb[i] = T[0, i]

    return T


def alpha_tot(r):
    if Re > 4000:
        Pr_T = 0.85 + 0.015 / Pr
        alpha_t = alpha + eddyDiffusivity(r, ri, dr, Re) * (Pr / Pr_T) * alpha
    else:
        alpha_t = alpha

    return alpha_t


def Tw(Tsea, Tb, dz, Fw, hi, ho, kwax, koil, kpipe, rd, ri, ro):
    Vwax = Fw
    Voil = 1 - Fw

    a = -2 * (Vwax + Voil)
    b = 2 * (Vwax * kwax + Voil * koil) - Voil * kwax - Vwax * koil
    c = kwax * koil * (Vwax + Voil)
    kdep1 = -b / (2 * a) + np.sqrt(b ** 2 - 4 * a * c) / (2 * a)
    kdep2 = -b / (2 * a) - np.sqrt(b ** 2 - 4 * a * c) / (2 * a)

    if kdep1 >= 0:
        kdeposit = kdep1
    else:
        kdeposit = kdep2

    Ad = 2 * np.pi * rd * dz
    Ai = 2 * np.pi * ri * dz
    Ao = 2 * np.pi * ro * dz

    Alm_d = (Ai - Ad) / np.log(Ai / Ad)
    Alm_p = (Ao - Ai) / np.log(Ao / Ai)

    Ri = 1 / (hi * Ai)
    Rd = (ri - rd) / (kdeposit * Alm_d)
    Rp = (ro - ri) / (kpipe * Alm_p)
    Ro = 1 / (ho * Ao)
    Rtot = Ri + Rd + Rp + Ro

    q = (Tsea - Tb) / Rtot

    Td = Tb + (q * Ri)
    Twall_in = Td + (q * Rd)
    Two = Twall_in + (q * Rp)
    Tcheck = Two + (q * Ro)

    return Td


def concentration(ni, nj, R, nu, Ti, T, mu, Va, gamma, kr, dr, dz, u, Re):
    def Dwo_tot(dr, j, i):
        Dwo = 13.3 * 10 ** (-12) * (T[j, i] + 273.15) ** 1.47 * mu ** gamma / Va ** 0.71
        if Re > 4000:
            Sc = nu / Dwo
            Sc_T = 0.85 + 0.015 / Sc
            Dwo_t = Dwo + eddyDiffusivity(j * dr, R[i], dr, Re) * Sc / (Sc_T * Dwo)
        else:
            Dwo_t = Dwo

        return Dwo_t

    C = np.zeros((nj, ni))
    C[:, 0] = Solubility(Ti)
    C[nj - 1, 0] = Solubility(T[nj - 1, 0])

    A_C = np.zeros((nj, nj))
    A_C[0, 0] = 1
    A_C[0, 1] = -1
    A_C[nj - 1, nj - 1] = 1

    D_C = np.ones(nj)
    D_C[0] = 0
    D_C[nj - 1] = Solubility(T[nj - 1, 0])

    for i in range(1, ni):
        for j in range(1, nj - 1):
            Dwo1 = Dwo_tot(dr, (j - 1), i)
            Dwo2 = Dwo_tot(dr, j, i)
            Dwo3 = Dwo_tot(dr, (j + 1), i)

            vz = Velocity(j * dr, R[i], nu, u, Re)

            A_C[j, j - 1] = -(1) / (2 * j * dr ** 2) * (j * Dwo2 + (j - 1) * Dwo1)
            A_C[j, j] = (vz / dz) + (1) / (2 * j * dr ** 2) * (
                    2 * j * Dwo2 + (j + 1) * Dwo3 + (j - 1) * Dwo1) + kr[j, i]
            A_C[j, j + 1] = -(1) / (2 * j * dr ** 2) * ((j + 1) * Dwo3 + (j) * Dwo2)

            D_C[j] = C[j, i - 1] * vz / dz + kr[j, i] * Solubility(T[j, i])

        D_C[nj - 1] = Solubility(T[nj - 1, i])

        N = np.linalg.solve(A_C, D_C)
        C[:, i] = np.ravel(N)
        C[0, i] = C[1, i]

    return C


def Solubility(T):
    solubility = 0.0007 * T ** 2 + 0.0989 * T + 1.7706
    return solubility


def wat_solubility(nj, ni, C, Tc):
    wat_sol = np.zeros((nj, ni))
    deposited = 0

    solubility_Tc = 0.0007 * Tc ** 2 + 0.0989 * Tc + 1.7706
    for i in range(0, ni):
        for j in range(0, nj):
            if deposited:
                wat_sol[j, i] = max(0, solubility_Tc - C[j, i])  # doubt
            else:
                wat_sol[j, i] = min(C[j, i], solubility_Tc)  # doubt

    return wat_sol


def kr_generator(ni, nj, T, gamma, Tc, kr_c, E, Re):
    kr = np.zeros(np.shape(T))
    for j in range(0, nj):
        for i in range(0, ni):
            if Re < 2300 or T[j, i] > Tc:
                kr[j, i] = 0
            else:
                kr[j, i] = kr_c * ((T[j, i] / Tc) ** 1.47) * np.exp(((gamma * E) / 8.314) * ((1 / T[j, i]) - (1 / Tc)))
    return kr


def eddyDiffusivity(r, R, dr, Re):
    def y_p(r):

        y_p = (1 - (r / R)) * (Re / 2) * np.sqrt(f / 8)

        return y_p

    def vz_p(y_p):
        if y_p <= 5:
            vz_p = y_p
        elif 5 <= y_p <= 30:
            vz_p = 5 * np.log(y_p) - 3.05
        else:
            vz_p = 2.5 * np.log(y_p) + 5.5

        return vz_p

    C1 = 0.4
    C2 = 26
    f = 0.305 / (Re ** 0.25)
    y2 = y_p(r + dr)
    y1 = y_p(r)
    dv = vz_p(y2) - vz_p(y1)
    dy = y2 - y1
    dvdy = dv / dy

    eddyDiff = (C1 * y_p(r)) ** 2 * (1 - np.exp(-y_p(r) / C2)) ** 2 * dvdy
    return eddyDiff


def Velocity(r, R, nu, u, Re):
    if Re < 2300:
        velocity = 2 * u * (1 - (r / R) ** 2)
        return velocity

    elif 2300 < Re < 4000:
        velocity = 2 * u * (1 - (r / R) ** 2)
        return velocity

    else:
        y = R - r
        f = 0.305 / (Re ** 0.25)
        y_p = (1 - r / R) * (Re / 2) * np.sqrt(f / 8)

        if y_p <= 5:
            vz_p = y_p
        elif 5 <= y_p <= 30:
            vz_p = 5 * np.log(y_p) - 3.05
        else:
            vz_p = 2.5 * np.log(y_p) + 5.5

        velocity = (vz_p * nu / y) * (1 - r / R) * (Re / 2) * np.sqrt(f / 8)

    return velocity


def FwdepThick(asp, ri, rho_gel, dr, dt, rd, Fwax, Dwo, dC, dCb):
    if dC < 0:
        De = Dwo / (1 + ((asp + Fwax ** 2) / (1 - Fwax)))
        Sh = ((-2 * np.pi * rd) * dC / dr) / dCb
        kM = (Dwo * Sh) / (2 * rd)
        drddt = -(1 / (rho_gel * Fwax)) * (kM * dCb + De * dC / dr)
        rdnew = rd + drddt * dt
        depThick = ri - rdnew

        if rdnew <= 0:
            print('The pipe is clogged')

        if abs(ri - rd) < 1.1e-12:
            dFwdt = 0
        else:
            dFwdt = (2 * rd / (rho_gel * (ri ** 2 - rd ** 2))) * (-De * dC / dr)

        Fwax_new = Fwax + dFwdt * dt
    else:
        De = 0
        Sh = 0
        kM = 0
        drddt = 0
        rdnew = rd + drddt * dt
        depThick = ri - rdnew
        dFwdt = 0
        Fwax_new = Fwax + dFwdt * dt

    return Fwax_new, depThick, rdnew, Sh, dFwdt, drddt, De, kM


d = 0.3048
xWall = 0.012
L = 1500
ni = L
nj = 100
Tsea = 5
Ti = 43
WAT = 39
u = 2
U = 20
x1 = 10 ** -12
ri = d / 2
ro = ri + xWall
kr_c = 1.4
E = 37700
Fw = 0.4
Fwi = 0.05
kwax = 0.25
koil = 0.1
kpipe = 20
mu = 0.5 * 10 ** -3
Va = 430
rho_oil = 750
rho_gel = 750
Cp = 2300
asp = 3
gamma = (10.2 / Va) - 0.791
nu = mu / rho_oil
Re = u * d / nu
alpha = koil / (rho_oil * Cp)
Pr = nu / alpha
Nu = 0.023 * Re ** 0.8 * Pr ** 0.3
hi = (Nu * koil) / d
Across = np.pi * ri ** 2
Q = u * Across
dr = ri / nj
dz = L / ni
Ai = 2 * np.pi * ri * dz
Ao = 2 * np.pi * ro * dz
Alm_p = (Ao - Ai) / np.log(Ao / Ai)
Rtotal = 1 / U
Rp = (ro - ri) / (kpipe * Alm_p)
hp = 1 / Rp

ho = 1 / (Rtotal - 1 / hi - 1 / hp)
Ro = 1 / (ho * Ao)

dt = int(3600)
t_final = int(dt * 24)
time = int(t_final / dt)
z = np.linspace(0, L, ni)
r = np.linspace(0, ri, nj)
print('inicio de modelo')
print('cargando modelo')
if True:
    rd = ri - x1
    rdnew = np.zeros((time + 1, ni))
    rdnew[0, :] = rd
    R = rdnew
    dmdt = np.zeros((time, ni))
    x = np.zeros((time, ni))
    x_in = 0

    Fwax = np.zeros((time + 1, ni))
    K = np.zeros((time, ni))
    Dwo = np.zeros((time, ni))
    dC = np.zeros((time, ni))
    dCb = np.zeros((time, ni))

    waxFrac = np.zeros((time, ni))
    depThickness = np.zeros((time, ni))
    Sh = np.zeros((time, ni))
    dFwdt = np.zeros((time, ni))
    drddtout = np.zeros((time, ni))
    De = np.zeros((time, ni))
    kM = np.zeros((time, ni))

    T = np.zeros((nj, ni, time))
    C = np.zeros((nj, ni, time))
    for t in range(0, time):
        T[:, :, t] = temperature_calc(ni, nj, ri, rdnew[t, :], ro, nu, Tsea, Ti, alpha, dr, dz, Fwi, hi, ho, kwax, koil,
                                      kpipe, Pr, u, Re)

        kr = kr_generator(ni, nj, T[:, :, t], gamma, WAT, kr_c, E, Re)
        C[:, :, t] = concentration(ni, nj, rdnew[t, :], nu, Ti, T[:, :, t], mu, Va, gamma, kr, dr, dz, u, Re)
        C[:, :, t] = wat_solubility(nj, ni, C[:, :, t], WAT)

        for i in range(0, ni):
            dC[t, i] = C[len(C) - 1, i, t] - C[len(C) - 2, i, t]
            dCb[t, i] = C[0, i, t] - C[len(C) - 1, i, t]

            if dC[0, i] < 0:
                Fwax[0, i] = Fwi
            if dC[t, i] == 0:
                Dwo[t, i] = 0
            else:
                K[t, i] = T[len(T) - 1, i, t] + 273.15
                Dwo[t, i] = 13.3 * 10 ** (-12) * (K[t, i] ** 1.47 * mu ** gamma) / (Va ** 0.71)

            Fwax[t + 1, i], depThickness[t, i], rdnew[t + 1, i], Sh[t, i], dFwdt[t, i], drddtout[t, i], De[t, i], kM[
                t, i] = FwdepThick(asp, ri, rho_gel, dr, dt, rdnew[t, i], Fwax[t, i], Dwo[t, i], dC[t, i], dCb[t, i])

    print(C)

    # transpose
    # x = z
    # y = r
    # z = T[:, :, 23]
    #
    # fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])
    #
    # fig.update_traces(contours_z=dict(
    #     show=True, usecolormap=True,
    #     highlightcolor="yellowgreen",
    #     project_z=True))
    #
    # fig.update_layout(scene=dict(
    #     xaxis_title='Z [m]',
    #     yaxis_title='R [m]',
    #     zaxis_title='T[Z,R]'),
    #     width=700,
    #     margin=dict(r=20, b=10, l=10, t=10))
    #
    # fig.show()

    # plt.plot(r, (C[:, 0, 23]))
    # plt.plot(r, (C[:, 1, 23]))
    # plt.plot(r, (C[:, 2, 23]))
    # plt.plot(r, (C[:, 3, 23]))
    # plt.show()

    # print(C)
    # CS = plt.contour(z, r, T[:, :, 23], 15, linewidth=1, colors='k')
    # CS2 = plt.contourf(z, r, T[:, :, 23], 15, cmap=plt.cm.jet)
    # plt.clabel(CS, fontsize=10)
    # plt.ylabel('r')
    # plt.xlabel('z')
    # plt.title('T(r,z)')
    # plt.colorbar()
    # plt.show()

    # plt.plot(r, (T[:, 0, 23]))
    # plt.plot(r, (T[:, 1, 23]))
    # plt.plot(r, (T[:, 2, 23]))
    # plt.plot(r, (T[:, 3, 23]))
    # plt.show()
