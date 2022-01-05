#todo
#1 plot eins gegen Y
#2 plot zwei gegen Y
#3 Plot 4 gegen Y
#-> Jeweils Modelle vergleichen (R2,AIC,..)
# Mögliche Fehler analysieren.

#3) OLS R2, AIC, etc. untersuchen
#ggf zeug auslagern??
#import wbdata as wd
from Vergleich import *
import numpy as np
import wbgapi as wb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits import  mplot3d
#import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.stats.stattools import durbin_watson
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf

#values von 'WA76' bis 'WA99' sind ok.

dataY = pd.read_csv("/users/jonagrimm/Desktop/ExportAuswahl.csv", sep=";")
dataX = pd.read_csv("/users/jonagrimm/Desktop/ExportInputs.csv", sep=";")
print(dataX.head )
print(dataY.head )
#erzeuge zeigt uns, dass die Variante 2,1-also Distance+CurGDP das optimale R2 hat.
#erzeuge('WA77', dataX, dataY)

#zunächst wollen wir nach Kollinearität zwischen Distance+CurGDP schauen.
#Das wäre insofern  auch theoretisch nachvollziehbar, weil Nähe zu Deutschland mit Lage in Europa und damit einem
#relativ hohen GDP/capita korelliert. Andererseits sind die Bevölkerungszahlen europäischer Staaten vhm. gering.

# wir schauen deshalb zunächst in den scatter-plot
mischer=create_mischer('WA87',dataX,dataY)
erzeuge('WA87', dataX, dataY)


#plt.scatter(mischer[['Distance']], mischer[['GDP(curUSDmil)']] )
#plt.xlabel('Distance')
#plt.ylabel('GDPcurUSDmil')
# plt.show()


#wir könnten das jetzt für alle 10 Kombinationen von zwei Variablen machen, aber das wäre zu anstrengend, deshalb machen wir nur VIF

# IST DAS RICHTIG SO??? - das mit dem Adden scheint nicht nötig, weil ich ja schon die konstanten Länder davor habe??
mischer_vif=add_constant(mischer)
ColumnNames=mischer_vif.columns.values
print(ColumnNames)

for g in range(3,7):
    for k in range(2,g):
        varF1=variance_inflation_factor(mischer_vif[[ColumnNames[g],ColumnNames[k]]].values,1)
        print('Der VIF ist für  '+ColumnNames[g]+ColumnNames[k]+' folgendes: ' +str(varF1))

# Die VIFs sind alle kleiner als 3. Deshalb wird nicht von Multikollinearität ausgegangen




#->Wir testen mit Durbin-Watson auf AutoKorr der Residuen:
ols=linear_model.LinearRegression()
ols.fit(mischer[['Distance','GDP(curUSDmil)']],mischer[['Wert']])
predictsY=ols.predict(mischer[['Distance','GDP(curUSDmil)']])
erg_list=list()
for i in range(0, len(mischer['Wert'])) :
    erg_list.append((predictsY[i])[0]-(mischer['Wert'])[i])
print(erg_list)
print(durbin_watson(erg_list))

#Die DS Statistik zeigt uns eine akzeptable 1,6 an. Ansonsten könnten wir mit folgendem auch für die anderen Modelle die AKs bekommen:
#DurWat_alle(mischer)



# Varianz residuen konstant? "heteroskedastisch"-> breusch pegan test:

#HIER ACHTUNG--------------------------
mischer=mischer.rename({'GDP(curUSDmil)':'GDPcurUSDmil'},axis=1 )
print(mischer['GDPcurUSDmil'])
fitT = smf.ols('Wert ~ Distance+GDPcurUSDmil', data=mischer).fit()
print(fitT.summary())

test = sms.het_breuschpagan(fitT.resid, fitT.model.exog)
print(test)

#wir erhalten p Wert von 0,07. Ausgehend von alpha=5% lehnen wir damit homoskedastie nicht ab--> varianzen konst.


#ToDo: 1) Beziehung zwischen exo+endogen linear?
# X/Y Ausreißer?



# AUSWERTUNG: wir haben soweit ein funktionales Modell, bei dem die Nullhypothese: keine der typischen Problemstellungen
#zumindest nicht offensichlich abgelehnt werden muss. Die Passgenauigkeit erscheint befriedigend, aber ggf. noch ausbaufähig durch Hinzunahme weiterer Parameter.
# Außerdem könnte noch eine visuelle Überprüfung der Linearität mittels 3d Bild erfolgen.


