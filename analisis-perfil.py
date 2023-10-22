import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from codigo.F_ajustes import chi2_reducido

def z_vortex(r, Gamma, zinfty, r0):
    g = 9.8
    return - Gamma**2 / (8*np.pi**2*g*(r - r0)**2) + zinfty

df = pd.read_csv("resultados-perfil.csv", delimiter = ",", header = 0)  # DATOS TOMADOS DE IMAGEJ
df.Y = df.Y.min() - df.Y

r = df.X.to_numpy()
z = df.Y.to_numpy()
z_err = 0.025    # por el ancho de un px en cm

z = z[np.argsort(r)]
r = r[np.argsort(r)]

centro_aprox = 12.7
cond = np.abs(r - centro_aprox) > 0.5
popt, pcov = curve_fit(z_vortex, r[cond], z[cond], p0 = [200, 0, centro_aprox])

rf = np.linspace(min(r), max(r), 500)
rf = np.where(np.abs(rf - popt[2]) > 0.5, rf, rf*np.nan)

plt.errorbar(r - popt[2], z, yerr = z_err, fmt = 's', ms = 1, color = "lightgray", capsize = 1, zorder = 0,  label = "Datos")
plt.plot(rf - popt[2], z_vortex(rf, *popt), zorder = 2, ls = "--", lw = 2, label = "Ajuste")


plt.xlabel("$r$ [cm]")
plt.ylabel("$z$ [cm]")

plt.xlim(-2.5, 2.5)
plt.ylim(min(df.Y) - 0.5, max(df.Y) + 0.5)

plt.legend()
plt.savefig("perfil-vortice.pdf")
plt.show()

print(popt, np.sqrt(np.diag(pcov)))