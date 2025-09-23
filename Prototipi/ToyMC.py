'''
Toy Montecarlo
Inizio del progetto.
Utilizzando un codice precedente scritto in C++ provo 
a ricreare uno script Python con le stesse funzionalità
'''
import random
import numpy as np
import matplotlib.pyplot as plt

def montecarlo(time_of_experiment: float,
               rate: float,
               number_of_experiments: int,
               eff1: float,
               eff2: float,
               eff3: float) -> None:
    '''
    ## MonteCarlo
        
    '''
    eff_histo = []
    for _ in range(0,number_of_experiments):
        time = 0.
        number_of_events = 0
        A = False
        B = False
        C = False
        AB = 0
        AC = 0
        BC = 0
        ABC = 0
        while time < time_of_experiment:
            time += np.log(random.random()/rate)
            number_of_events += 1
            if random.random() < eff1:
                A = True
            if random.random() < eff2:
                B = True
            if random.random() < eff3:
                C = True
            if A and B:
                AB+=1
            if A and C:
                AC += 1
            if B and C:
                BC += 1
            if (A and B) and C:
                ABC += 1
            A = False
            B = False
            C = False
        eff_histo.append(ABC/AC)
    plt.hist(eff_histo,bins = 31)
    plt.savefig("histo.png")
    plt.show()
if __name__ == "__main__":
    montecarlo(1000,0.1,10000,0.80,0.30,0.50)