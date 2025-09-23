# %%
import numpy as np

# %%
def singole_multi(T,rate_1,rate_2,rate_3,rate_cosm,vec1,vec2,vec3):
    '''
    ## Single
    
    '''
    # Creo una lista di tempi in cui arrivano i cosmici
    t_cosm = 0
    cosmic = []
    while(t_cosm < T):
        t_cosm += -np.log(np.random.random())/rate_cosm
        cosmic.append(t_cosm)
    cosmic = np.round(cosmic,6)
    # Do un valore temporale iniziale ai 3 scintillatori 
    t_1 = np.round(-np.log(np.random.random())/rate_1, 6)
    t_2 = np.round(-np.log(np.random.random())/rate_2, 6)
    t_3 = np.round(-np.log(np.random.random())/rate_3, 6)
    # Questi sono 3 indici per scorrere separatamente "cosmic"
    i_1 = 0
    i_2 = 0
    i_3 = 0
    while(True):
        # In questo modo si crea il vettore dei tempi per lo scintilltore 1
        # la prima parte è un controllo sulla lunghezza di cosmic
        # la seconda parte tiene in considerazione se il valore temporle t1
        # avviene prima o dopo il valore registrato all'indice i_1-esimo
        # se avviene prima del passaggio del cosmico, viene inserito dentro 
        # vec1 e poi viene generato il prossimo valore di t1
        # se invece avviene dopo il passaggio del cosmico, si appende il
        # valore del cosmico e si prosegue con l'indice. 
        # in questo modo si ottiene un vettore con tutti gli scatti degli 
        # scintillatori, sia accidentali sia dovuti ai cosmici. 

        if(i_1 < len(cosmic) and t_1 > cosmic[i_1]):
            vec1.append(cosmic[i_1])
            i_1 = i_1 + 1
        if(i_1 < len(cosmic) and t_1 < cosmic[i_1]):
            vec1.append(t_1)
            t_1 += np.round(-np.log(np.random.random())/rate_1, 6)

        # Equivalenti al blocco per lo scintillatore 1
        
        if(i_2 < len(cosmic) and t_2 < cosmic[i_2]):
            vec2.append(t_2)
            t_2 += np.round(-np.log(np.random.random())/rate_2,6)
        if(i_2 < len(cosmic) and t_2 > cosmic[i_2]):
            vec2.append(cosmic[i_2])
            i_2 = i_2 + 1
        
        if(i_3 < len(cosmic) and t_3 < cosmic[i_3]):
            vec3.append(t_3)
            t_3 += np.round(-np.log(np.random.random())/rate_3, 6)
        if(i_3 < len(cosmic) and t_3 > cosmic[i_3]):
            vec3.append(cosmic[i_3])
            i_3 = i_3 + 1
        # Si controlla che tutti gli indici siano esauriti per uscire dal while
        if(t_1 > T and t_2 > T and t_3 > T and i_1 == len(cosmic) and i_2 == len(cosmic) and i_3 == len(cosmic)):
            break
    return cosmic

# %%
vec_1 = []
vec_2 = []
vec_3 = []
cosmic_v = np.array(singole_multi(10000,300,60,234,10,vec_1,vec_2,vec_3))
vec_1 = np.array(vec_1)
vec_2 = np.array(vec_2)
vec_3 = np.array(vec_3)

# %%
coincidenze_ab = np.intersect1d(vec_3,vec_2)

# %%
j = 0
for n,i in enumerate(coincidenze_ab):
    if cosmic_v[n+j]-i: 
        print(n, cosmic_v[n+j],i)
        j -= 1

# %%


# %%



