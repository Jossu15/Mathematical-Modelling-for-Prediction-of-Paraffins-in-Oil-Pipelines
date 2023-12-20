from Laminar_Numerical_Model import *
import requests


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
a = 4.9 * 10 **-9
b = 17.8
c = 6
z = np.linspace(0, L, ni)
r = np.linspace(0, R, nj)


files1 = {'vz':vz,'R':R,
         'L':L,'Tw':Tw,
         'Ti':Ti,'alpha':alpha,'mu':mu,'Va':Va,
         'n':n,'ni':ni,'nj':nj,'a':a,'b':b,'c':c,'z':z.tolist(),'r':r.tolist()}



r1 = requests.post('http://127.0.0.1:5000//LaminalModel', json=files1)


print(r1)
print(r1.url)
print(r1.status_code)
print(r1.text)

