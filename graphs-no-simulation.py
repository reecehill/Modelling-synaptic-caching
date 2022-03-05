from numpy import linspace
import matplotlib.pyplot as plt


def makeFigure3b():
    P = 500
    N = 500
    LEARNING_RATE = 0.1
    #T = epochIndexForConvergence
    T = (P**(3/2)) / ((2-(P/N))**2)
    c = [0, 10**-4, 10**-3, 10**-2]
    K = (2*P)/((2-(P/N))**2) #Numerically found
    
    decayRates = linspace(0,0.004,100)
    fig = plt.figure(figsize=(10, 6), dpi=300)
    for cValue in c:
      optThreshold = (LEARNING_RATE**2) * ((3*K) / (1+cValue*T))
      plt.plot(decayRates, optThreshold, label=cValue, color="red",  linewidth=1, linestyle="-")
    plt.show()

makeFigure3b()