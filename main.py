import pandas as pd
import acp.ACP as acp
import grafice as g
import numpy as np


tabel = pd.read_csv('dataIN/DateUE.csv', index_col=0)
print(tabel)

# creare lista variabile observate
varNume = list(tabel.columns)[1:]
print(varNume)
# creare lista etichete observatii
obsNume = list(tabel.index)
print(obsNume)
# extragere matrice valori variabile observate
X = tabel[varNume].values
print(X)

# instatinere clasa ACP
acp_model = acp.ACP(X)

# salvare X standardizat in fisier CSV
Xstd = acp_model.getXstd()
Xstd_df = pd.DataFrame(data=Xstd, index=obsNume, columns=varNume)
Xstd_df.to_csv('dataOUT/Xstd.csv')

# creare grafic varianta explicata de catre componentele principale
valProp = acp_model.getValProp()
g.componentePrincipale(valoriProprii=valProp)
# g.afisare()

# creare corelograma factori de corelatie
Rxc = acp_model.getRxc()
Rxc_df = pd.DataFrame(data=Rxc, index=varNume, columns=('C'+str(k+1) for k in range(len(varNume))))
Rxc_df.to_csv('dataOUT/FactoriCorelatie.csv')
g.corelograma(matrice=Rxc_df, dec=2, titlu='Corelograma factorilor de corelatie')
# g.afisare()

# creare corelograma a scorurilor
scoruri = acp_model.getScoruri()
scoruri_df = pd.DataFrame(data=scoruri, index=obsNume, columns=('C'+str(k+1) for k in range(len(varNume))))
scoruri_df.to_csv('dataOUT/Scoruri.csv')
g.corelograma(matrice=scoruri_df, dec=2, titlu='Corelograma scorurilor (componentele principale standardizate)')
# g.afisare()

# creare corelograma a calitatii reprezentarii observatiilor pe axele componentelor principale
calObs = acp_model.getCalObs()
calObs_df = pd.DataFrame(data=calObs, index=obsNume, columns=('C'+str(k+1) for k in range(len(varNume))))
calObs_df.to_csv('dataOUT/CalitateObservatii.csv')
g.corelograma(matrice=calObs_df, dec=2, titlu='Corelograma a calitatii reprezentarii observatiilor pe axele componentelor principale')
# g.afisare()

# creare corelograma contributiei observatiilor la varianta axelor componentelor principale
betha = acp_model.getBetha()
betha_df = pd.DataFrame(data=betha, index=obsNume, columns=('C'+str(k+1) for k in range(len(varNume))))
betha_df.to_csv('dataOUT/Betha.csv')
g.corelograma(matrice=betha_df, dec=2, titlu='Corelograma contributiei observatiilor la varianta axelor componentelor principale')
# g.afisare()

# creare corelograma comunitati (regasirea componentelor principale in variabilele initiale)
comunitati = acp_model.getComun()
comunitati_df = pd.DataFrame(data=comunitati, index=varNume, columns=('C'+str(k+1) for k in range(len(varNume))))
comunitati_df.to_csv('dataOUT/Comunitati.csv')
g.corelograma(matrice=comunitati_df, dec=2, titlu='Corelograma comunitati (regasirea componentelor principale in variabilele initiale)')
# g.afisare()

# cercul corelatiilor intre varaibilelel initiale si componentele C1 si C2
# g.cerculCorelatiilor(matrice=Rxc, titlu='Cercul corelatiilor intre varaibilelel initiale si componentele C1 si C2')
g.cerculCorelatiilor(matrice=Rxc_df, titlu='Cercul corelatiilor intre varaibilelel initiale si componentele C1 si C2')
# g.afisare()

# distributia observatiilor in spatiul componentele C1 si C2
scor_max = np.max(scoruri)
scor_min = np.min(scoruri)
print('Scor maxim, folosit ca raza pentru cercul corelatiilor: ', scor_max)
g.cerculCorelatiilor(matrice=scoruri_df, raza=scor_max, valMin=scor_min, valMax=scor_max,
                     titlu='Distributia observatiilor in spatiul componentele C1 si C2')
g.afisare()