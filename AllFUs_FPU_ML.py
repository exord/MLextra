import os
import time
import numpy as np
import sklearn
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from astropy.io import ascii
from astropy.time import Time

def deltara2sky(deltara, dec):
    '''
    Delta ra_angle to sky_angle
    
    It is almost equivalent to : real_delta_ra = delta_ra * cos(dec)

    :param deltara: [deg] Delta ra
    :param dec: [deg] absolute dec
    :return: [deg]
    '''
    return np.rad2deg(np.arccos(1. + (np.cos(np.deg2rad(deltara)) - 1.0)*np.cos(np.deg2rad(dec))**2.)     )


def AllFUs(fpu):

    datafile = 'FPU{}_ML_2021-03-22.csv'.format(fpu)
    
    data = ascii.read(os.path.join('datasets', datafile), delimiter=',')
    
    df = pd.read_csv(os.path.join('datasets/', datafile))

    
    df = pd.read_csv('datasets/FPU{}_ML_2021-03-22.csv'.format(fpu))
    
    # Columns have extra spaces; let's correct that (and remove field, just to keep a fully numeric dataframe; why?)
    df = pd.DataFrame(df.values[:, 1:], columns=df.columns.str.strip(' ')[1:], dtype='float64')

    # offset ha
    df['ha'] = df['ha'] 
    
    # Compute X and Y differences for each arm
    for i in range(1, 6):
        for d in ['X', 'Y']:
            df['FU{}_d{}'.format(i, d)] = df['FU{}_{}_observed'.format(i, d)] - df['FU{}_{}_calculated'.format(i, d)]

    if fpu == 1:
        raise SystemError('No data')
    elif fpu == 2:
        reference00 = {1: [716961, 23439970], 2: [848553, 22862347], 3: [377384, 23407169], 4: [-270410, 23304840], 5: [29727, 23524875]}
    elif fpu == 3:
        reference00 = {1: [110172, 23584097], 2: [525673, 23515499], 3: [685832, 23188744], 4: [-362080, 22780065], 5: [-741491, 23781604]}

    # Compute calculated X and Y delta to center
    for i in range(1, 6):
        df['FU{}_Delta_racosdec'.format(i)] = deltara2sky(df['FU{}_Delta_ra'.format(i)], (df['FU{}_dec'.format(i)] + df['dec'])/2 )
        df['FU{}_Delta_X_calculated'.format(i)] = df['FU{}_X_calculated'.format(i)] - reference00[i][0]
        df['FU{}_Delta_Y_calculated'.format(i)] = df['FU{}_Y_calculated'.format(i)] - reference00[i][1]
        df['FU{}_Radius_calculated'.format(i)] = np.sqrt(df['FU{}_Delta_X_calculated'.format(i)]**2 + df['FU{}_Delta_Y_calculated'.format(i)]**2)
        
    cond = df.alt >= 0.0

    X = df.loc[cond, ('ha', 'dec', 'temperature', 'airmass', 'alt',
                      'FU1_Delta_racosdec', 'FU1_Delta_dec', 'FU1_Delta_X_calculated', 'FU1_Delta_Y_calculated', 'FU1_Radius_calculated', 'FU1_Delta_Teff',
                      'FU2_Delta_racosdec', 'FU2_Delta_dec', 'FU2_Delta_X_calculated', 'FU2_Delta_Y_calculated', 'FU2_Radius_calculated', 'FU2_Delta_Teff',
                      'FU3_Delta_racosdec', 'FU3_Delta_dec', 'FU3_Delta_X_calculated', 'FU3_Delta_Y_calculated', 'FU3_Radius_calculated', 'FU3_Delta_Teff',
                      'FU4_Delta_racosdec', 'FU4_Delta_dec', 'FU4_Delta_X_calculated', 'FU4_Delta_Y_calculated', 'FU4_Radius_calculated', 'FU4_Delta_Teff',
                      'FU5_Delta_racosdec', 'FU5_Delta_dec', 'FU5_Delta_X_calculated', 'FU5_Delta_Y_calculated', 'FU5_Radius_calculated', 'FU5_Delta_Teff',
                      )]
    t = df.loc[cond, ('FU1_dX', 'FU1_dY',
                      'FU2_dX', 'FU2_dY',
                      'FU3_dX', 'FU3_dY',
                      'FU4_dX', 'FU4_dY',
                      'FU5_dX', 'FU5_dY')]

    # Split train test
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.01, random_state=1234)

    # Scale
    scaler = StandardScaler()
    X_train2 = scaler.fit_transform(X_train)
    X_test2 = scaler.transform(X_test)
    X_scaled2 = scaler.transform(X)

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train2)
    X_test_poly = poly.transform(X_test2)
    #pr = LinearRegression(fit_intercept=False)
    pr = Ridge(alpha=29.8) 

    #params = {'alpha': np.logspace(-3, 2, 20)}
    #prcv = GridSearchCV(pr, params, cv=5, scoring='neg_mean_squared_error')
    #prcv.fit(X_train_poly, t_train)
    
    pr.fit(X_train_poly, t_train)
    y_train = pr.predict(X_train_poly)
    y_test =  pr.predict(X_test_poly)
    print('Polinomial Regression RMSE (train): {:.2f}'.format(np.sqrt(mean_squared_error(t_train, y_train))))
    print('Polinomial Regression RMSE (test): {:.2f}'.format(np.sqrt(mean_squared_error(t_test, y_test))))

    polynomialRegression = {}
    polynomialRegression['estimator'] = pr
    polynomialRegression['poly'] = poly
    polynomialRegression['scaler'] = scaler
    

    import pickle
    f = open('2021-04-01_FPU{}_ML_v0.pickle'.format(fpu), 'wb')
    pickle.dump(polynomialRegression, f)
    f.close()

    X_scaled = poly.transform(X_scaled2)
    y_pr = pr.predict(X_scaled)

    '''
    # Feature importances
    params = X.columns
    importances = rscv.best_estimator_.feature_importances_
    simportances = np.std([tree.feature_importances_ for tree in rscv.best_estimator_], axis=0)
    indices = np.argsort(rscv.best_estimator_.feature_importances_)[::-1]
    for ii in range(len(params)):
        print('{}. {} : {:6.4f} +/- {:6.4f}'.format(ii + 1, params[indices][ii], importances[indices][ii], simportances[indices][ii]))
    # Plot the impurity-based feature importances of the forest
    f = plt.figure(figsize=(10, 7))
    ax = f.add_axes([0.15, 0.15, 0.8, 0.8])
    ax.set_title('Feature importances')
    ax.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=simportances[indices], align="center")
    plt.xticks(range(X.shape[1]), params[indices])
    plt.xticks(rotation=25, ha='right')
    ax.set_xlim([-1, X.shape[1]])
    plt.show()
    f.savefig('AllFUsFigures/FPU{}_FeatureImportances_LR.pdf'.format(fpu))
    '''
    
    for iFu in range(1, 6):
        f = plt.figure(figsize=(10, 10))
        ax = f.add_axes([0.1, 0.1, 0.85, 0.85])
        ax.plot(df['FU{}_dX'.format(iFu)]/1e3, df['FU{}_dY'.format(iFu)]/1e3, 'o', color='0.8', ms=5, label='data', alpha=1.0)

        # All sample
        dX = y_pr[:, 2 * (iFu - 1)]
        dY = y_pr[:, 2 * (iFu - 1) + 1]
        calculatedX = df['FU{}_X_calculated'.format(iFu)] + dX
        calculatedY = df['FU{}_Y_calculated'.format(iFu)] + dY
        ocX = df['FU{}_X_observed'.format(iFu)] - calculatedX 
        ocY = df['FU{}_Y_observed'.format(iFu)] - calculatedY
        ocR = np.sqrt(ocX**2 + ocY**2)
        std = np.std(ocR)
        print('All sample FU{} rms : {:.3f} microns'.format(iFu, std/1e3))
        #ax.plot(ocX/1e3, ocY/1e3, 'o', ms=5, label='Random Forest', alpha=1.0)
        ax.set_title(r'FU{}, $\sigma$ : {:.1f} $\mu$m, max : {:.1f} $\mu$m'.format(iFu, std/1e3, np.max(ocR)/1e3))
        '''
        inds = np.arange(len(ocR))[ocR > 5000]
        for ind in inds:
            print('    {:.1f} um: {}, {}, {}'.format(ocR[ind]/1e3, data['field'][ind], data['time'][ind],
                    Time(data['time'][ind], format='unix', scale='utc').to_datetime().strftime('%Y-%m-%dT%H:%M:%S')))
        '''
        # train
        dX = y_train[:, 2 * (iFu - 1)]
        dY = y_train[:, 2 * (iFu - 1) + 1]
        calculatedX = X_train['FU{}_Delta_X_calculated'.format(iFu)] + reference00[iFu][0] + dX
        calculatedY = X_train['FU{}_Delta_Y_calculated'.format(iFu)] + reference00[iFu][1] + dY
        ocX = ((X_train['FU{}_Delta_X_calculated'.format(iFu)] + reference00[iFu][0]) + t_train['FU{}_dX'.format(iFu)]) - calculatedX 
        ocY = ((X_train['FU{}_Delta_Y_calculated'.format(iFu)] + reference00[iFu][1]) + t_train['FU{}_dY'.format(iFu)]) - calculatedY
        ocR = np.sqrt(ocX**2 + ocY**2)
        std = np.std(ocR)
        print('Train FU{} rms : {:.3f} microns'.format(iFu, std/1e3))
        ax.plot(ocX/1e3, ocY/1e3, 'o', ms=5, label='train', alpha=1.0)

        # test
        dX = y_test[:, 2 * (iFu - 1)]
        dY = y_test[:, 2 * (iFu - 1) + 1]
        calculatedX = X_test['FU{}_Delta_X_calculated'.format(iFu)] + reference00[iFu][0] + dX
        calculatedY = X_test['FU{}_Delta_Y_calculated'.format(iFu)] + reference00[iFu][1] + dY
        ocX = ((X_test['FU{}_Delta_X_calculated'.format(iFu)] + reference00[iFu][0]) + t_test['FU{}_dX'.format(iFu)]) - calculatedX 
        ocY = ((X_test['FU{}_Delta_Y_calculated'.format(iFu)] + reference00[iFu][1]) + t_test['FU{}_dY'.format(iFu)]) - calculatedY
        ocR = np.sqrt(ocX**2 + ocY**2)
        std = np.std(ocR)
        print('Test FU{} rms : {:.3f} microns'.format(iFu, std/1e3))
        ax.plot(ocX/1e3, ocY/1e3, 'o', ms=5, label='test', alpha=1.0)
        
        circle = plt.Circle((0, 0), 20.0, edgecolor='k', facecolor='none', zorder=10, linewidth=0.5)
        ax.add_artist(circle)
        ax.set_xlabel(r'X$_{\rm O-C}$ [$\mu$m]')
        ax.set_ylabel(r'Y$_{\rm O-C}$ [$\mu$m]')
        ax.set_aspect('equal')
        ax.legend(loc='best')
        f.savefig('FPU{}_FU{}_PR.pdf'.format(fpu, iFu))
    



    
