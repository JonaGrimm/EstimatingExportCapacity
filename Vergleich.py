import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from statsmodels.stats.stattools import durbin_watson


def testing():
    print('huhu')

#good ist Warenkennung, muss also WA75..WA99 sein
def erzeuge(goods_selected,dataX,dataY):

    mischer=create_mischer(goods_selected, dataX,dataY)
    # die urspr., aber etwas alberne Analyse:
    ursp_analyse(mischer)


    x_pred = np.linspace(0, 350, 70)
    y_pred = np.linspace(0, 2000000, 50)
    xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
    model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

    ols = linear_model.LinearRegression()
    ColNames = ['Population', 'GDP(curUSDmil)', 'Distance', 'GDPgrowth', 'ResidentPatents', 'Wert']
    tryY = mischer['Wert']


    plt.scatter(mischer['Distance'], mischer['GDP(curUSDmil)'])
   # plt.show()

    #fit verschiedener Modelle zeigen
    for i in range(5):
        for s in range(i):
            tryX = mischer[[ColNames[i], ColNames[s]]]
            model = ols.fit(tryX.values, tryY)
            predicted = model.predict(model_viz)
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(mischer[ColNames[i]], mischer[ColNames[s]], mischer['Wert'], color='red')
            ax.scatter3D(xx_pred.flatten(), yy_pred.flatten(), predicted)

            print('Wir haben folgends R2:'+ str(model.score(tryX.values,tryY))+'Für Modell  ',str(i)+'   '+ str(s) +ColNames[i]+' - '+ColNames[s])

    print('__________________________________________')

    # plt.show(block=True)

   # print(model.score(tryX.values, tryY))











def create_mischer(goods_selected, dataX,dataY):
    #Daten nach Groesse von Population (nicht) aussortieren
    #Dann in einen DataFrame packen
    red_X = dataX[dataX['Population'] < 280][
        ['Country', 'Population', 'Distance', 'GDP(curUSDmil)', 'GDPgrowth', 'ResidentPatents']]
    red_Y = dataY[dataY['Kennung'] == goods_selected][['Wert', 'Land']]
    mischer = pd.merge(red_X, red_Y, left_on='Country', right_on='Land', how='inner')
    mischer['Wert'] = pd.to_numeric(mischer['Wert'], downcast='integer')
    return mischer



def ursp_analyse(mischer):
    Varianten = [['Distance', 'GDP(curUSDmil)', 'GDPgrowth', 'ResidentPatents'],
                 ['Population', 'GDP(curUSDmil)', 'GDPgrowth', 'ResidentPatents'],
                 ['Population', 'Distance', 'GDPgrowth', 'ResidentPatents'],
                 ['Population', 'Distance', 'GDP(curUSDmil)', 'ResidentPatents'],
                 ['Population', 'Distance', 'GDP(curUSDmil)', 'GDPgrowth']
                 ]
    finalX = mischer[Varianten[0]]
    finalY = mischer['Wert']
    print(finalX)
    print(finalY)
    linear_regressor = linear_model.LinearRegression()
    linear_regressor.fit(finalX, finalY)

    print(linear_regressor.intercept_)
    print(linear_regressor.coef_)

def DurWat_alle(mischer):
    colNam = mischer.columns.values

    for g in range(0, 6):
        for k in range(1, g):
            mod = linear_model.LinearRegression()
            mod.fit(mischer[[colNam[g], colNam[k]]], mischer['Wert'])
            predY = mod.predict(mischer[[colNam[g], colNam[k]]])
            erg_out = list()
            for i in range(0, len(mischer['Wert'])):
                erg_out.append(predY[i] - (mischer['Wert'])[i])
            print('Für Modell' + colNam[g] + '    ' + colNam[k] + '  ist die Durbin-W-Stat:  ' + str(
                durbin_watson(erg_out)))

    # mischer.plot.scatter(x='Distance',y='Wert',c='blue')
    # mischer.plot.scatter(x='GDP(curUSDmil)',y='Wert',c='red')
    # mischer.plot.scatter(x='Population',y='Wert',c='green')
    # mischer.plot.scatter(x='GDPgrowth',y='Wert',c='brown')
    # mischer.plot.scatter(x='ResidentPatents',y='Wert',c='blue')



#add country label- müsste aber im main stehen.
#for x in mischer['Country']:
 #   x_val=mischer[mischer['Country'] == x]['Distance']
 #   print(x_val)
 #   y_val=mischer[mischer['Country'] == x]['GDP(curUSDmil)']
 #   plt.text(x_val,y_val,s=x)