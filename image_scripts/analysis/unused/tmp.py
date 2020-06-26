import numpy as pylab;

def geo_efficiency(energy_keV):
    params = [-5.02, -1.272, -0.8367, -0.7216, -0.241, -0.02549]
    ln_eff = 0.
    ln_en = pylab.log(energy_keV/1460.)
    for j in range(len(params)):
        ln_eff += params[j]*pow(ln_en, j)
    return pylab.exp(ln_eff)

for x in range(0,5000,10):
    print x, geo_efficiency(x), 1,1;
