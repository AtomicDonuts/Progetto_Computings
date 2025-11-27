---
layout: page
title: Brainstorm
permalink: /ann/
---

Modello di Machine Learning:
Servono dei dei dati su cui allenarlo, perchè usare il catalogo non è il migliore dei modi:
Due idee
1) Montecarlo
2) Encoder-Decoder/GAN allenato sui dati che sono nel catalogo (con etichetta), a cui si fornisce random noise (Affidabile?)

Considerazioni su 2:
    Se si utilizza una GAN, potrebbe essere anche utilizzato per la classificazione anche delle sorgenti senza etichetta presenti nel catalogo.

Dopo aver generato i dati:
1) Deep Network Semplice
2) Tripla Deep Network, una per ogni tipo di spettro, che 