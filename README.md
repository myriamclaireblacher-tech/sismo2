compilation : dans build -> (mingw32-make clean) mingw32-make

exécution : sismo2.exe


modifier "F:/stage/sundials_install" dans cmakefile
     path of INSTALL_DIR in installation guide of SUNDIALS


Ex CV en série https://github.com/llnl/sundials/blob/main/examples/cvode/serial/cvRoberts_dns.c 

nécessitent 

libgcc_s_seh-1.dll

libstdc++-6.dll

libwinpthread-1.dll
de minGW

et [librairies  de SUNDIALS]

High dimension monte carlo : Chada, N. K., Schillings, C., & Weissmann, A. (2018). "Dimension-independent Markov chain Monte Carlo based on parallel tempering

Méthodologie : Si la variable $x$ a des bornes $[a, b]$, appliquer la transformation $y = \text{logit}((x - a) / (b - a))$. La proposition MCMC se fait sur $y \in \mathbb{R}$. La probabilité d'acceptation cible doit être multipliée par le déterminant du Jacobien de cette transformation.Référence d'appui : Carpenter, B., et al. (2017). "Stan: A Probabilistic Programming Language". (Bien qu'orienté HMC, la méthode de Constraint Transform est le standard technique utilisé pour traiter rigoureusement les frontières dans tout MCMC complexe)


Si le gradient est incalculable (boîte noire), la marche aléatoire simple doit être remplacée par des "stretch moves". Plusieurs "marcheurs" sont utilisés pour chaque température. La proposition pour un marcheur se fait le long de la ligne le reliant à un autre marcheur aléatoire. Cela s'adapte automatiquement à la matrice de covariance, même en grande dimension.

Référence d'appui : Vousden, W. D., Farr, W. M., & Mandel, I. (2016). "Dynamic temperature selection for parallel tempering in Markov chain Monte Carlo simulations". L'algorithme (implémenté via le package ptemcee) démontre l'utilisation de Goodman & Weare (2010) au sein d'une architecture PT.




