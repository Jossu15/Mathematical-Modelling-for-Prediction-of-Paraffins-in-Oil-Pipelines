import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def concentration(ni, nj, R, L, Q, nu, Tw, Ti, T, kr, turbulent, mu, Va):
    dr = R / nj
    dz = L / ni

    C = np.zeros((nj, nj))
    Cw = Solubility(Tw)
    Ci = Solubility(Ti)
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
            Dwo1 = Dwo_tot(dr, (j - 1), i)
            Dwo2 = Dwo_tot(dr, j, i)
            Dwo3 = Dwo_tot(dr, (j + 1), i)

            vz = Velocity(j * dr, R, Q, nu, turbulent)

            A_C[j, j - 1] = (-1 / 2 * j * dr ** 3) * (j * dr * Dwo2 + (j - 1) * dr * Dwo1)
            A_C[j, j] = (vz / dz) + (1 / 2 * j * dr ** 3) * (
                    2 * j * dr * Dwo2 + (j + 1) * dr * Dwo3 + (j - 1) * dr * Dwo1) + kr
            A_C[j, j + 1] = (-1 / 2 * j * dr ** 3) * ((j + 1) * dr * Dwo3 + (j) * dr * Dwo2)

        for j in range(1, nj - 1):
            vz = Velocity(j * dr, R, Q, nu, turbulent)
            D_C[j] = (C[j, i - 1] * vz / dz) + kr * Solubility(T[j, i])

        N = np.linalg.solve(A_C, D_C)
        C[:, i] = np.ravel(N)
        C[0, i] = C[1, i]

    return C


def Dwo_tot(dr, j, i):
    gamma = (10.2 / Va) - 0.791
    Dwo = 13.3 * 10 ** (-12) * (T[j, i] ** 1.47) * (mu ** gamma) / (Va ** 0.71)
    Sc = nu / Dwo
    Sc_T = 0.85 + 0.015 / Sc
    Dwo_t = Dwo + eddyDiffusivity(j * dr, R, Q, nu, dr) * Sc / (Sc_T * Dwo)
    return Dwo_t


def temperature(ni, nj, R, L, Q, nu, Tw, Ti, alpha0, turbulent):
    dr = R / nj
    dz = L / ni
    T = np.zeros((nj, nj))
    T[:, 0] = Ti

    A_T = np.zeros((nj, nj))
    A_T[nj - 1, nj - 1] = 1

    A_T[0, 0] = 1
    A_T[0, 1] = -1

    for j in range(1, nj - 1):
        alpha1 = alpha_tot(dr * (j - 1))
        alpha2 = alpha_tot(dr * (j))
        alpha3 = alpha_tot(dr * (j + 1))

        vz = Velocity(j * dr, R, Q, nu, turbulent)

        A_T[j, j - 1] = -1 / (2 * j * dr ** 3) * (j * dr * alpha2 + (j - 1) * dr * alpha1)  # C_T
        A_T[j, j] = (vz / dz) + 1 / (2 * j * dr ** 3) * (
                2 * j * dr * alpha2 + (j + 1) * dr * alpha3 + (j - 1) * dr * alpha1)  # A_T
        A_T[j, j + 1] = -1 / (2 * j * dr ** 3) * ((j + 1) * dr * alpha3 + (j) * dr * alpha2)  # B_T

    D_T = np.ones((nj, 1))
    D_T[0] = 0
    D_T[nj - 1] = Tw

    for i in range(1, ni):
        T[nj - 1, i] = T[nj - 1, i - 1]
        T[nj - 2, i] = T[nj - 1, i - 1]

        for j in range(1, nj - 1):
            vz = Velocity(j * dr, R, Q, nu, turbulent)
            D_T[j] = T[j, i - 1] * vz / dz

        N = np.linalg.solve(A_T, D_T)
        T[:, i] = np.ravel(N)
        T[0, i] = T[1, i]

    return T


def alpha_tot(r):
    Pr = nu / alpha0
    Pr_T = 0.85 + 0.015 / Pr
    alpha_to = alpha0 + eddyDiffusivity(r, R, Q, nu, dr) * Pr / Pr_T * alpha0
    return alpha_to


def Solubility(temperature):
    solubility = 0.0007 * temperature ** 2 + 0.0989 * temperature + 1.7706
    return solubility


def eddyDiffusivity(r, R, Q, nu, dr):
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

    Re = 2 * Q / (np.pi * R * nu)
    f = 0.305 / Re ** 0.25

    k = 0.4
    A = 26
    y2 = y_p(r + dr)
    y1 = y_p(r)
    dv = vz_p(y2) - vz_p(y1)
    dy = y2 - y1
    dvdy = dv / dy

    eddyDiff = (k * y_p(r)) ** 2 * (1 - np.exp(-y_p(r) / A)) ** 2 * dvdy
    return eddyDiff


def Velocity(r, R, Q, nu, turbulent):
    vm = Q / np.pi * (R ** 2)
    Re = 2 * Q / (np.pi * R * nu)

    if Re > 4000 and turbulent == False:
        velocity = 2 * vm * (1 - (r / R) ** 2)
        return velocity

    y = R - r
    f = 0.305 / (Re ** 0.25)
    y_p = (1 - (r / R)) * (Re / 2) * np.sqrt(f / 8)

    if y_p <= 5:
        vz_p = y_p
    elif 5 <= y_p <= 30:
        vz_p = 2.5 * np.log(y_p) - 3.05
    else:
        vz_p = 2.5 * np.log(y_p) + 5.5

    velocity = (vz_p * nu / y) * (1 - (r / R)) * (Re / 2) * np.sqrt(f / 8)

    return velocity


ni = 250
nj = 400
R = 0.00794
L = 2

Tw = 7
Ti = 70

alpha0 = 9.8 * 10 ** -7
Dwo = 4 * 10 ** -9
Q = 0.00063
nu = 4.44 * 10 ** -5
mu = 0.004
kr = 0.15
Va = 400
turbulent = True

n = 5
dr = float(R / nj)
dz = L / ni
z = np.linspace(0, L, ni)
r = np.linspace(0, R, nj)

T = temperature(ni, nj, R, L, Q, nu, Tw, Ti, alpha0, turbulent)
C = concentration(ni, nj, R, L, Q, nu, Tw, Ti, T, kr, turbulent, mu, Va)
S = Solubility(T)

# transpose
# x = z
# y = r
# z = S
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

# fig = go.Figure(data=
# go.Contour(
#     z=T,
#     x=z,  # horizontal axis
#     y=r,  # vertical axis
# ))
#
# fig.update_layout(title='Temperature distribution',
#                   xaxis_title="Distance from inlet [m]",
#                   yaxis_title="Radial distance from pipe [m]",
#                   )
#
# fig.show()

# ******** Plotting concentration profiles in the boundary layer**********
kr2 = [0.02, 0.05, 0.10, 0.15, 0.2]

profile = np.zeros((50, n))
for k in range(0, n):
    C = concentration(ni, nj, R, L, Q, nu, Tw, Ti, T, kr2[k], turbulent, mu, Va)
    for j in range(0, 180):
        profile[:, k] = C[len(C) - 51:len(C) - 1, 15]

# r1 = np.linspace(0.005955, R, 50)
# plt.plot(r1, profile[:, 0])
# plt.plot(r1, profile[:, 1])
# plt.plot(r1, profile[:, 2])
# plt.plot(r1, profile[:, 3])
# plt.plot(r1, profile[:, 4])
# plt.show()

# ****************Plotting Sherwood number against kr*********************
kr1 = [0.02, 0.05, 0.10, 0.15, 0.2]
Sh = np.ones((n, ni))

for k in range(0, n):
    C = concentration(ni, nj, R, L, Q, nu, Tw, Ti, T, kr1[k], turbulent, mu, Va)
    C[len(C) - 1, 0] = 2.49720
    for i in range(0, ni):
        dcdr = (C[nj - 1, i] - C[nj - 2, i]) / dr
        Sh[k, i] = -2 * R * dcdr / (C[0, i] - C[nj - 1, i])

plt.plot(z, Sh[0, :], label='kr=0.02')
plt.plot(z, Sh[1, :], label='kr=0.05')
plt.plot(z, Sh[2, :], label='kr=0.10')
plt.plot(z, Sh[3, :], label='kr=0.15')
plt.plot(z, Sh[4, :], label='kr=0.20')
plt.xlabel('Distance from inlet [m]')
plt.ylabel('Sherwood number')
plt.xlim(min(z), max(z))
plt.legend()
plt.show()
