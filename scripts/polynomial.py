#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from matplotlib import animation
from tqdm.notebook import tqdm, trange

#plot the function a*(F-X)*(1-2*X +(1+b**2)*X**2)-G^2 with F=6 and G=1 b=4, for different values of a
#and find the fixed points

#parameters
F=6
G=1
b=4

#define polynomial function that solves for the fixed points
def f(X, a):
    return a*(F-X)*(1-2*X +(1+b**2)*X**2)-G**2

#plot
X=np.linspace(-50, 50, 10000)

if F==6:
    plt.plot(X, f(X, 0.179179179), label=r'a_c = 0.1788',color='red')
if F==8:
    plt.plot(X, f(X, 0.134134134), label=r'a* $\simeq$ 0.13',color='red')
# plt.plot(X, f(X, 0.2), label='a = 0.20', color='navy')
a_range=np.arange(0.05, 0.3, 0.01)
for a in a_range:
    plt.plot(X, f(X, a), color='navy', alpha=0.2)

plt.plot([-2, 9], [0, 0], 'k')

#y lim to -10, 10
plt.ylim(-1, 1)
plt.xlim(-2, 9)
plt.legend(fontsize=15, loc='upper right')

plt.xlabel('X', fontsize=15)
plt.ylabel('f(X)', rotation=0, fontsize=15)
#plt.title('Steady solutions, b=4, F='+str(F)+', G=1')
plt.xticks(fontsize=15)
plt.yticks(np.linspace(-1, 1, 5), fontsize=15)
plt.tight_layout()
# %%
