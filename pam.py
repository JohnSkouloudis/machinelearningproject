import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfcinv

#A
def inverse_Qfunction(x):
    return erfcinv(2*x)*np.sqrt(2)

def find_SNRb_DB(Pb,M):
    SNRb = 0
    y = (M*np.log2(M)*Pb)/(2*( M - 1))
    Q = inverse_Qfunction(y) 
    z = ( M**2 - 1)/6 * np.log2(M)
    SNRb = (Q**2) * z
    return 10*np.log10(SNRb)


print('SNRb in db is:',find_SNRb_DB(1e-3, 4))
#B

#it2021091 last digit is 1. Pb= 10^-x-2= 10^-3

Pb =1e-3
m =np.arange(1 , 11)
M=2**m

SNRb = find_SNRb_DB(Pb, M)

print('M values:', M)

plt.close('all')
plt.figure(1)
plt.plot(M, SNRb)
plt.xlabel('M')
plt.ylabel('SNRb(in dB)')
plt.title('SNRb=f(M)(in dB)')

for i, txt in enumerate(M):
    plt.annotate(txt, (M[i], SNRb[i]), textcoords="offset points", xytext=(0, 5), ha='center')

plt.show()

