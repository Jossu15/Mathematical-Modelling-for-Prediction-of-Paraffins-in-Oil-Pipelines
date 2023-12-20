from flask import Flask, request, jsonify
import json
from analytical_model import *
from Laminar_Numerical_Model import *

app = Flask(__name__)


@app.route('/analytical_model', methods=['POST'])
def behaviour():
    data = request.json

    Tsea = data['Tsea']
    d = float(data['d'])
    Ti = data['Ti']
    U = data['U']
    D = data['D']
    Cp = data['Cp']
    dz = data['dz']
    nj = data['nj']
    dt = data['dt']
    t_final = data['t_final']
    t = data['t']
    time = data['time']
    dt2 = data['dt2']
    t_final2 = data['t_final2']
    t2 = data['t2']
    time2 = data['time2']
    dt1 = data['dt1']
    t_final1 = data['t_final1']
    t1 = data['t1']
    time1 = data['time1']
    m = data['m']
    Fw = data['Fw']
    Kwax = data['Kwax']
    Koil = data['Koil']
    Kpipe = data['Kpipe']
    Vwax = data['Vwax']
    Voil = data['Voil']
    rho_oil = data['rho_oil']
    rho = data['rho']
    mu = data['mu']
    Pr = data['Pr']
    u = data['u']
    y_p = data['y_p']
    Re = data['Re']
    thickness = data['thickness']
    ri = data['ri']
    dr_wall = data['dr_wall']
    ro = data['ro']
    Nu = data['Nu']
    hi = data['hi']
    Va = data['Va']
    gamma = data['gamma']
    WAT = data['WAT']
    Kdep = data['Kdep']

    TemperatureL = analytical_model_1.tempL(Tsea, Ti, U, d, Cp, dz, nj, time, m)

    respuesta = analytical_model_1.deposit(thickness, ri, ro, hi, Kpipe, Kdep, Tsea, TemperatureL, dz, nj, time, mu, Va,
                                           gamma, WAT, dt, y_p, Cp,
                                           Re, U, rho, Pr, u)

    return respuesta


@app.route('/LaminalModel', methods=['POST'])
def behaviour2():
    data = request.json

    vz = float(data['vz'])
    R = float(data['R'])
    L = data['L']
    Tw = data['Tw']
    Ti = float(data['Ti'])
    alpha = data['alpha']
    mu = data['mu']
    Va = data['Va']
    n = data['n']
    ni = data['ni']
    nj = data['nj']
    a = float(data['a'])
    b = float(data['b'])
    c = data['c']

    z = data['z']
    r = data['r']

    T = Numerical_model_1.temperature(ni, nj, R, L, Tw, Ti, alpha, vz)

    respuesta = Numerical_model_1.Answers(ni, nj, R, L, Tw, Ti, T, mu, vz, Va, a, b, c, alpha)

    return respuesta


if __name__ == '__main__':
    app.run(debug=True)
