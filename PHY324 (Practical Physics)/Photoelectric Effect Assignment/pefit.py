from random import gauss
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import scipy.optimize as optimize
import scipy.stats as stats
import pickle

with open('exp1.csv', 'r') as file:
    data1 = np.loadtxt(file, delimiter=',', usecols=range(4))
with open('exp2.csv', 'r') as file:
    data2 = np.loadtxt(file, delimiter=',', usecols=range(5))
    
data1 = np.transpose(data1)
data2 = np.transpose(data2)

def linfit(f, m, b):
    return m*f+b

def linfitb0(I,m):
    return m*I

c = 2.99792458*10**8
e = 1.602176634*10**(-19)

wlens = data1[0]*10**(-9)
wunc = data1[1]*10**(-9)

freq = c/wlens
func = freq*(wunc/wlens)

vstopw = data1[2]
vwunc = data1[3]

opt, unc = optimize.curve_fit(linfit, freq, vstopw, absolute_sigma=True, sigma=vwunc)

mopt = opt[0]
munc = np.sqrt(unc[0][0])

bopt = opt[1]
bunc = np.sqrt(unc[1][1])

vwfitted = linfit(freq, mopt, bopt)
vwfitunc = vwfitted*np.sqrt((func/freq)**2+(munc/mopt)**2+(bunc/bopt)**2)

chisquared1 = np.sum(((vwfitted-vstopw)/vwunc)**2)
DOF1 = len(vstopw)-2

h=mopt*e
hunc = munc*e

e0 = -bopt*e
e0unc = bunc*e

f0 = e0/h
f0unc = f0*np.sqrt((e0unc/e0)**2+(hunc/h)**2)

residuals = vstopw - vwfitted
resunc = vwunc

print('chisquared1',chisquared1/DOF1)
print(mopt, munc)
print(bopt, bunc)
print('h1',h,hunc)
print('E1',e0,e0unc)
print('f1',f0,f0unc)
print('')
optmod, uncmod = optimize.curve_fit(linfit, freq[:-1],vstopw[:-1],absolute_sigma=True, sigma=vwunc[:-1])

moptn = optmod[0]
muncn = np.sqrt(uncmod[0][0])

boptn = optmod[1]
buncn = np.sqrt(uncmod[1][1])

vwfittedn = linfit(freq[:-1], moptn, boptn)
vwfituncn = vwfittedn*np.sqrt((func[:-1]/freq[:-1])**2+(muncn/moptn)**2+(buncn/boptn)**2)

chisquared2 = np.sum(((vwfittedn-vstopw[:-1])/vwunc[:-1])**2)
DOF2 = DOF1-1

hn = moptn*e
huncn = muncn*e
percenterr = (hn-6.63*10**(-34))/(6.63*10**(-34))

e0n=-boptn*e
e0uncn=buncn*e

f0n = e0/h
f0uncn=f0n*np.sqrt((e0uncn/e0n)**2+(huncn/hn)**2)

residualsn = vstopw[:-1]-vwfittedn
resuncn=vwunc[:-1]

print('chisquared2',chisquared2/DOF2)
print(moptn, muncn)
print(boptn, buncn)
print('h2',hn,huncn)
print('E2',e0n,e0uncn)
print('f2',f0n, f0uncn)
print('')
ints = data2[0]
intunc = 0.1*ints

vstopc = data2[1]
vcunc = data2[2]

copt, csigma = optimize.curve_fit(linfit, ints, vstopc, absolute_sigma=True, sigma=vcunc)

c = copt[0]
k = copt[1]
cunc = np.sqrt(csigma[0][0])
kunc = np.sqrt(csigma[1][1])

print(c,cunc)
print(k,kunc)
print('')
vcopt = linfit(ints, c, k)

vi = data2[3]
viunc = data2[4]

aopt, asigma = optimize.curve_fit(linfit,ints, vi, absolute_sigma=True, sigma=viunc)

a = aopt[0]
b = aopt[1]
aunc = np.sqrt(asigma[0][0])
bunc = np.sqrt(asigma[1][1])

print(a,aunc)
print(b,bunc)
viopt = linfit(ints, a, b)

tdelay = 85 * 10**(-6)
tunc = 3*10**(-6)

p = 60*10**(-5)
punc = 1*10**(-5)

A = 3.23*10**(-4)
uncA = 0.01*10**(-4)

d = 3*10**(-10)
dunc = 1*10**(-10)

tact = (A*e0n)/(p*d**2)
ttunc = tact*np.sqrt((dunc/d)**2+(uncA/A)**2+(punc/p)**2+(e0uncn/e0n)**2)

print('')
print(tact,ttunc)

plt.plot(freq, vstopw, linestyle='None', marker='d', markersize=4, label='Data')
plt.errorbar(freq, vstopw, xerr=func, yerr=vwunc, linestyle='None', ecolor='black', capsize=1, label='Errors')
plt.plot(freq, vwfitted, label='Fit', color='red')
plt.xlabel('Frequency ($Hz$)', fontsize=14)
plt.ylabel('Stopping Voltage ($V$)', fontsize=14)
plt.legend(loc=1)
plt.savefig('voltfreq1',dpi=1000)
plt.show()
plt.close()

plt.plot(freq, residuals, linestyle='None', marker='d', markersize=4, label='Residuals')
plt.errorbar(freq, residuals, yerr=resunc, linestyle='None', ecolor='black', capsize=1, label='Errors')
plt.xlabel('Frequency ($Hz$)', fontsize=14)
plt.ylabel('Voltage Residuals ($V$)', fontsize=14)
plt.legend(loc=1)
plt.savefig('voltres1',dpi=1000)
plt.show()
plt.close()

plt.plot(freq[:-1], vstopw[:-1], linestyle='None', marker='d', markersize=4, label='Data')
plt.errorbar(freq[:-1], vstopw[:-1], xerr=func[:-1], yerr=vwunc[:-1], linestyle='None', ecolor='black', capsize=1, label='Errors')
plt.plot(freq[:-1], vwfittedn, label='Fit', color='red')
plt.xlabel('Frequency ($Hz$)', fontsize=14)
plt.ylabel('Stopping Voltage ($V$)', fontsize=14)
plt.legend(loc=1)
plt.savefig('voltfreq2',dpi=1000)
plt.show()
plt.close()

plt.plot(freq[:-1], residualsn, linestyle='None', marker='d', markersize=4, label='Residuals')
plt.errorbar(freq[:-1], residualsn, yerr=resuncn, linestyle='None', ecolor='black', capsize=1, label='Errors')
plt.xlabel('Frequency ($Hz$)', fontsize=14)
plt.ylabel('Voltage Residuals ($V$)', fontsize=14)
plt.legend(loc=1)
plt.savefig('voltres2',dpi=1000)
plt.show()
plt.close()

plt.plot(ints, vstopc, linestyle='None', marker='d', markersize=4, label='Data')
plt.errorbar(ints, vstopc, xerr=intunc, yerr=vcunc, linestyle='None', ecolor='black', capsize=1, label='Errors')
plt.plot(ints, vcopt, color='red', label='Fit')
plt.xlabel('Intensity Setting (arb. units)', fontsize=14)
plt.ylabel('Stopping Voltage ($V$)', fontsize=14)
plt.legend(loc=1)
plt.savefig('voltint',dpi=1000)
plt.show()
plt.close()

plt.plot(ints, vi, linestyle='None', marker='d', markersize=4, label='Data')
plt.errorbar(ints, vi, xerr=intunc, yerr=viunc, linestyle='None', ecolor='black', capsize=1, label='Errors')
plt.plot(ints, viopt, color='red', label='Fit')
plt.xlabel('Intensity Setting (arb. units)', fontsize=14)
plt.ylabel('Photocurrent Voltage ($V$)', fontsize=14)
plt.legend(loc=1)
plt.savefig('currint',dpi=1000)
plt.show()
plt.close()