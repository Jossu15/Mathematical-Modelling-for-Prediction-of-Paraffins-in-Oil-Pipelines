import numpy as np
from math import exp, log
import matplotlib.pyplot as plt
from analytical_model import *

Tsea = 5
d = 0.3048
Ti = 43
U = 20
D = np.linspace(0, 10000, 10000)
Cp = 2300
dz = 1
nj = 10000

dt = 3600
t_final = dt * 24 * 7
t = np.linspace(0, t_final, 169)
time = int(t_final / dt)

dt2 = 3600
t_final2 = dt2 * 24 * 2
t2 = np.linspace(0, t_final2, 49)
time2 = int(t_final2 / dt2)

dt1 = 3600
t_final1 = dt1 * 24 * 1
t1 = np.linspace(0, t_final1, 25)
time1 = int(t_final1 / dt1)

m = 109.4488155
Fw = 0.4
Kwax = 0.25
Koil = 0.1
Kpipe = 20

Vwax = Fw
Voil = (1 - Fw)

rho_oil = 750
rho = rho_oil
mu = 0.5 * 10 ** (-3)
Pr = (Cp * mu) / Koil
u = 2
y_p = 5
Re = (rho_oil * u * d) / mu

thickness = 10 ** (-12)
ri = d / 2
dr_wall = 0.012
ro = ri + dr_wall
Nu = 0.023 * Re ** 0.8 * Pr ** 0.3
hi = (Nu * Koil) / d
Va = 430
gamma = 10.2 / Va - 0.791
WAT = 39

a = -2 * (Vwax + Voil)
b = 2 * (Vwax * Kwax + Voil * Koil) - Voil * Kwax - Vwax * Koil
c = Kwax * Koil * (Vwax + Voil)

Kdep1 = -b / (2 * a) + np.sqrt(b ** 2 - 4 * a * c) / (2 * a)
Kdep2 = -b / (2 * a) - np.sqrt(b ** 2 - 4 * a * c) / (2 * a)

if Kdep1 >= 0:
    Kdeposit = Kdep1

else:
    Kdeposit = Kdep2

Kdep = Kdeposit


TemperatureL = analytical_model_1.tempL(Tsea, Ti, U, d, Cp, dz, nj, time, m)

Result = analytical_model_1.deposit( thickness, ri, ro, hi, Kpipe, Kdep, Tsea, TemperatureL, dz, nj, time, mu, Va, gamma, WAT, dt, y_p, Cp,
                Re, U, rho, Pr, u)

print(Result)
