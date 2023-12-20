import numpy as np
import matplotlib.pyplot as plt


class Effective_thermal_conductivity:

    @classmethod
    def ETC(self, Fw, kwax, koil):

        number = len(Fw)
        Vwax = np.zeros((number, 1))
        Voil = np.zeros((number, 1))

        kdep1 = np.zeros((number, 1))
        kdep2 = np.zeros((number, 1))
        kdep3 = np.zeros((number, 1))
        kdep4 = np.zeros((number, 1))
        kdep5 = np.zeros((number, 1))

        for i in range(0, number):
            Vwax[i] = Fw[i]
            Voil[i] = (1 - Fw[i])

        for i in range(0, number):
            kdep1[i] = kwax * Fw[i] + koil * (1 - Fw[i])

        for i in range(0, number):
            kdep2[i] = 1 / ((Fw[i] / kwax) + ((1 - Fw[i]) / koil))

            for i in range(0, number):
                kdep3[i] = (koil * Voil[i] + kwax * Vwax[i] * (3 * koil / (2 * koil + kwax))) / (
                        Voil[i] + Vwax[i] * ((3 * koil) / (2 * koil + kwax)))

            for i in range(0, number):
                kdep4[i] = (kwax * Vwax[i] + koil * Voil[i] * (3 * kwax / (2 * kwax + koil))) / (
                        Vwax[i] + Voil[i] * ((3 * kwax) / (2 * kwax + koil)))

        for i in range(0, number):

            a = -2 * (Vwax[i] + Voil[i])
            b = 2 * (Vwax[i] * kwax + Voil[i] * koil) - Voil[i] * kwax - Vwax[i] * koil
            c = kwax * koil * (Vwax[i] + Voil[i])
            k1 = -b / (2 * a) + np.sqrt(b ** 2 - 4 * a * c) / (2 * a)
            k2 = -b / (2 * a) - np.sqrt(b ** 2 - 4 * a * c) / (2 * a)

            if k1 >= 0:
                kdep5[i] = k1
            else:
                kdep5[i] = k2

        return kdep1, kdep2, kdep3, kdep4, kdep5


# Assigning input values:

Fw = np.linspace(0.05, 0.7, 6501)
kwax = 0.25
koil = 0.10

kdep1, kdep2, kdep3, kdep4, kdep5 = Effective_thermal_conductivity.ETC(Fw, kwax, koil)

plt.plot(Fw, kdep1, label='parallel model')
plt.plot(Fw, kdep2, label='ME2')
plt.plot(Fw, kdep3, label='EMT')
plt.plot(Fw, kdep4, label='ME1')
plt.plot(Fw, kdep5, label='Series model')
plt.title('Thermal Conductivity of Paraffin Wax Deposit')
plt.xlabel('Weight Fraction of Wax in Deposit')
plt.ylabel('Kdeposit , w/mk')
plt.xlim(min(Fw), max(Fw))
plt.legend()
plt.show()
