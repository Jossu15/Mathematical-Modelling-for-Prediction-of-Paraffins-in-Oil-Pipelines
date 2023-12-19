import numpy as np
import plotly.graph_objects as go


def Temperature(ni, nj, R, L, Q, nu, Tw, Ti, alpha0, turbulent):

    """

    :param ni:
    :param nj:
    :param R:
    :param L:
    :param Q:
    :param nu:
    :param Tw:
    :param Ti:
    :param alpha0:
    :param turbulent:
    :return:
    """
    def alpha_tot(r):
        """

        :param r:
        :return:
        """
        Pr_T = 0.85 + 0.015 / Pr
        alpha_to = alpha0 + eddyDiffusivity(r, R, Q, nu, dr) * Pr / Pr_T * alpha0
        return alpha_to

    done = 0
    epsilon = 0.01

    dr = R / nj
    dz = L / ni

    T = np.zeros((nj, ni))
    T[:, 0] = Ti
    Pr = nu / alpha0

    A_T = np.zeros((nj, nj))
    A_T[nj - 1, nj - 1] = 1
    A_T[0, 0] = 1
    A_T[0, 1] = -1

    for j in range(1, nj - 1):
        alpha1 = alpha_tot(dr * (j - 1))
        alpha2 = alpha_tot(dr * j)
        alpha3 = alpha_tot(dr * (j + 1))

        vz = Velocity(j * dr, R, Q, nu, turbulent)

        A_T[j, j - 1] = -1 / (2 * j * dr ** 3) * (j * dr * alpha2 + (j - 1) * dr * alpha1)  # C_T
        A_T[j, j] = (vz / dz) + 1 / (2 * j * dr ** 3) * (
                2 * j * dr * alpha2 + (j + 1) * dr * alpha3 + (j - 1) * dr * alpha1)  # A_T
        A_T[j, j + 1] = -1 / (2 * j * dr ** 3) * ((j + 1) * dr * alpha3 + j * dr * alpha2)  # B_T

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

        if Tw + epsilon > T[0, i] > Tw - epsilon and not done:
            print('The bulk flow has reached the ambient temperature after', str((i - 1) * dz),
                  'meter, and there is no further precipitation')
            done = 1

    return T


def wat_location(nj, ni, T, dr, choice):
    """

    :param nj:
    :param ni:
    :param T:
    :param dr:
    :param choice:
    :return:
    """
    wat = np.zeros(ni)

    if choice == 1:
        Tc = 39
    elif choice == 2:
        Tc = 51
    elif choice == 3:
        Tc = 34.3

    # else:
    #     Tc = user defined

    for i in range(0, ni):
        for j in range(0, nj):
            if T[j, i] <= Tc:
                wat[i] = dr * (j - 1)
                break
            wat[i] = 1
    return wat


def concentration(ni, nj, R, L, Q, nu, Tw, Ti, T, turbulent, mu, Va, gamma, choice, kr, wat):
    """

    :param ni:
    :param nj:
    :param R:
    :param L:
    :param Q:
    :param nu:
    :param Tw:
    :param Ti:
    :param T:
    :param turbulent:
    :param mu:
    :param Va:
    :param gamma:
    :param choice:
    :param kr:
    :param wat:
    :return:
    """
    def Dwo_tot(dr, j, i):
        """

        :param dr:
        :param j:
        :param i:
        :return:
        """
        Dwo = 13.3 * 10 ** (-12) * (T[j, i] + 273.15) ** 1.47 * mu ** gamma / Va ** 0.71
        Sc = nu / Dwo
        Sc_T = 0.85 + 0.015 / Sc
        dwo_tot = Dwo + eddyDiffusivity(j * dr, R, Q, nu, dr) * Sc / Sc_T * Dwo

        return dwo_tot

    dr = R / nj
    dz = L / ni

    C = np.zeros((nj, ni))
    Cw = solubility(choice, Tw)
    Ci = solubility(choice, Ti)
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
            D_C[j] = (C[j, i - 1] * vz / dz) + kr[j, i] * solubility(choice, T[j, i])

        N = np.linalg.solve(A_C, D_C)
        C[:, i] = np.ravel(N)
        C[0, i] = C[1, i]

    return C


def solubility(choice, temperature):
    """

    :param choice:
    :param temperature:
    :return:
    """
    if choice == 1:
        Solubility = 0.0007 * temperature ** 2 + 0.0989 * temperature + 1.7706
    elif choice == 2:
        Fahrenheit = temperature * (9 / 5) + 32
        Solubility = 5 * 10 ** (-6) * Fahrenheit ** 3 - 0.0016 * Fahrenheit ** 2 + 0.1959 * Fahrenheit - 2.377
    elif choice == 3:
        Fahrenheit = temperature * (9 / 5) + 32
        Solubility = 10 ** (-7) * Fahrenheit ** 4 - 3 * 10 ** (
            -5) * Fahrenheit ** 3 + 0.0033 * Fahrenheit ** 2 - 0.1301 * Fahrenheit + 4.1683

    # else:
    #     Fahrenheit = temperature * (9 / 5) + 32
    #     Solubility = Solubility

    return Solubility


def kr_generator(ni, nj, T, gamma, turbulent, choice):
    """

    :param ni:
    :param nj:
    :param T:
    :param gamma:
    :param turbulent:
    :param choice:
    :return:
    """
    if choice == 1:
        Tc = 39
    elif choice == 2:
        Tc = 51
    elif choice == 3:
        Tc = 34.3

    kr_c = 1.4
    E = 1
    kr = np.zeros((np.shape(T)))

    for j in range(0, nj):
        for i in range(0, ni):
            if turbulent == False or T[j, i] > Tc:
                kr[j, i] = 0
            else:
                kr[j, i] = kr_c * (T[j, i] / Tc) ** 1.47 * np.exp(((gamma * E) / 8.314) * ((1 / T[j, i]) - (1 / Tc)))

    return kr


def eddyDiffusivity(r, R, Q, nu, dr):
    """

    :param r:
    :param R:
    :param Q:
    :param nu:
    :param dr:
    :return:
    """
    def y_p(r):
        """

        :param r:
        :return:
        """

        y_p = (1 - (r / R)) * (Re / 2) * np.sqrt(f / 8)

        return y_p

    def vz_p(y_p):
        """

        :param y_p:
        :return:
        """
        if y_p <= 5:
            vz_p = y_p
        elif 5 <= y_p <= 30:
            vz_p = 5 * np.log(y_p) - 3.05
        else:
            vz_p = 2.5 * np.log(y_p) + 5.5

        return vz_p

    Re = 2 * Q / (np.pi * R * nu)
    f = 0.305 / Re ** 0.25

    C1 = 0.4
    C2 = 26
    y2 = y_p(r + dr)
    y1 = y_p(r)
    dv = vz_p(y2) - vz_p(y1)
    dy = y2 - y1
    dvdy = dv / dy

    eddyDiff = (C1 * y_p(r)) ** 2 * (1 - np.exp(-y_p(r) / C2)) ** 2 * dvdy
    return eddyDiff


def Velocity(r, R, Q, nu, turbulent):
    """

    :param r:
    :param R:
    :param Q:
    :param nu:
    :param turbulent:
    :return:
    """
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


def massrate(rho, R, L, ni, nj, dr, dz, mu, gamma, Va, nu, C, T, Q):
    """

    :param rho:
    :param R:
    :param L:
    :param ni:
    :param nj:
    :param dr:
    :param dz:
    :param mu:
    :param gamma:
    :param Va:
    :param nu:
    :param C:
    :param T:
    :param Q:
    :return:
    """
    def Dwo_tot(dr, j, i):
        """

        :param dr:
        :param j:
        :param i:
        :return:
        """
        Dwo = 13.3 * 10 ** (-12) * (T[j, i] + 273.15) ** 1.47 * mu ** gamma / Va ** 0.71
        Sc = nu / Dwo
        Sc_T = 0.85 + 0.015 / Sc
        dwo_tot = Dwo + eddyDiffusivity(j * dr, R, Q, nu, dr) * Sc / Sc_T * Dwo

        return dwo_tot

    Massrate = np.zeros(ni)
    areal = 2 * np.pi * dr * dz

    for i in range(0, ni):
        Massrate[i] = -rho * Dwo_tot(dr, nj, i) * areal * ((C[nj - 1, i] - C[nj - 2, i]) / (100 * dr))

    return Massrate


def wat_solubility(nj, ni, C, dr, choice):
    """

    :param nj:
    :param ni:
    :param C:
    :param dr:
    :param choice:
    :return:
    """
    wat_sol = np.zeros((nj, ni))
    deposited = 0

    for i in range(0, ni):
        for j in range(0, nj):
            if deposited:
                wat_sol[j, i] = max(0, solubility(choice, T[j, i]) - C[j, i])  # doubt
            else:
                wat_sol[j, i] = min(C[j, i], solubility(choice, T[j, i]))  # doubt

    return wat_sol


ni = 10000
nj = 10000
L = 500
R = 0.1525
Tw = 5
Ti = 53
Q = 0.2192

choice = int(input('Choose the oil to be used. Enter 1 for Norde Crude Oil, 2 for South Pelto Oil,'
                   ' 3 for Garden Banks Condensate, or 4 for user defined option: '))

alpha0 = 9.8 * 10 ** (-7)

Dwo_0 = 4 * 10 ** -9
nu = 4.44 * 10 ** -5
mu = 0.004
Va = 430
rho = 900
turbulent = True
gamma = 10.2 / Va - 0.791

dr = R / nj
dz = L / ni
z = np.linspace(0, L, ni)
r = np.linspace(0, R, nj)

if True:
    T = Temperature(ni, nj, R, L, Q, nu, Tw, Ti, alpha0, turbulent)
    Kr = kr_generator(ni, nj, T, gamma, turbulent, choice)
    wat = wat_location(nj, ni, T, dr, choice)
    C = concentration(ni, nj, R, L, Q, nu, Tw, Ti, T, turbulent, mu, Va, gamma, choice, Kr, wat)
    # C = wat_solubility(nj, ni, C, dr, choice)

    print(T)
