---
layout: post
title: Brainstorm
permalink: /ann/brainstorm
---

Modello di Machine Learning:
Servono dei dei dati su cui allenarlo, perchè usare il catalogo non è il migliore dei modi:
Due idee
1. Montecarlo
2. Encoder-Decoder/GAN allenato sui dati che sono nel catalogo (con etichetta), a cui si fornisce random noise (Affidabile?)

Considerazioni su 2:
    Se si utilizza una GAN, potrebbe essere anche utilizzato per la classificazione anche delle sorgenti senza etichetta presenti nel catalogo.

Dopo aver generato i dati:
1. Deep Network Semplice
2. Tripla Deep Network:
    1. 3 Deep Network Semplici, uno per ogni spettro, che poi fanno un qualche tipo di pool alla fine per la classificazione.
    2. Convolutional Neural Network 1D con 3 filtri, uno per ogni spettro, che poi alla fine fanno pool (non so se è effettivamente possibile)
3. Tripla Deep/CNN + Deep Network Aggiuntiva.

