#Python file containing all custom functions used in the mod2.ipyn file
import statsmodels.api as sm
import numpy as np
def fillNull(df, columns, numericalData = True, naive = True, showInfo = False):
    '''
        This function takes in a DataFrame and column names of the DataFrame which 
        have null values. It also has an option 'naive' which takes a bool. If set 
        to true, fillNull will fill in all null numerical data with the average of 
        the column the data was pulled from.
        
        Parameters:
        
            df - A pandas DataFrame object (pd.DataFrame())

            columns - A list of column names from the columns of a pandas DataFrame 
                      object (pd.DataFrame()[column])
                      
            numericalData - A bool value that types the list of columns being fed.
                            If true, the mean value of the column will be used to fill
                            in missing values. If false, the null values will be made
                            into their own column.

            naive - A bool value that determines the method for filling in the null 
                    values. Default is True.

            showInfo - A bool value that determines if the df[x].value_counts() command
                       will be displayed upon executing the function.
                   
        Returns:
        
            None
            
    '''
    if numericalData == True:
        for x in columns:
            fillIn = df[x].mean()
            df[x] = df[x].fillna(fillIn)
            if showInfo == True:
                display(f'Column {x} now has {df[x].isna().sum()} null values')
                display(f'The mean of column {x} = {fillIn}')
                display(df[x].value_counts())
    else:
        for x in columns:
            df[x] = df[x].astype('object')
            fillIn = 'Null'
            df[x] = df[x].fillna(fillIn)
            df[x] = df[x].astype('category')
            if showInfo == True:
                display(f'Column {x} now has {df[x].isna().sum()} null values')
                display(f'The mean of column {x} = {fillIn}')
                display(df[x].value_counts())

def removeFeatures(df,features, target, RThreshold = .8, pThreshold = .05):
    loopCount = 0
    tempFeatures = features.copy()
    succeededCols = []
    removedCols = []
    pendingCols = []
    Y = df[target]
    X = df[tempFeatures]
    X = sm.add_constant(X)
    model = sm.OLS(endog = Y, exog = X).fit()
    R = model.rsquared
    print(f' The starting R-value for the model is {R}')
    print('\n')
    while True:
        
        
#         Y = df[target]
#         X = df[tempFeatures+pendingCols]
#         X = sm.add_constant(X)
#         model = sm.OLS(endog = Y, exog = X).fit()
#         R = model.rsquared
        
        badP = model.pvalues[tempFeatures].loc[model.pvalues > .05]
        # print(badP)
        if len(badP) > 0:
            print(f'The following columns have p-values above the threshold of {pThreshold}: {list(badP.index)}\n')
            for x in list(badP.index):
                tempFeatures.remove(x)
                pendingCols.append(x)
                
#         print(pendingCols)
#         print()
#         print(succeededCols)
        absModelParams = abs(model.params.drop(index = ['const']+pendingCols+succeededCols))
        leastImpactfulCoeff = absModelParams.loc[absModelParams == absModelParams.min()]
        leastImpactful = leastImpactfulCoeff.index[0]
        print(f'{leastImpactful} was removed with a coefficient of {leastImpactfulCoeff[0]}\n' )
#         print(leastImpactful)
        tempFeatures.remove(leastImpactful)
        pendingCols.append(leastImpactful)
        
        Y = df[target]
        X = df[tempFeatures+succeededCols]
        X = sm.add_constant(X)
        model = sm.OLS(endog = Y, exog = X).fit()
        R = model.rsquared
        
        if( R < RThreshold) & (len(pendingCols) > 0):
            print(f'''The R-squared value has dropped to {R} which is below the acceptable threshold of {RThreshold}. Adding feature: {pendingCols[-1]} back into the model and continuing \n\n\n''')
            succeededCols.append(pendingCols.pop())
            Y = df[target]
            X = df[tempFeatures+succeededCols]
            X = sm.add_constant(X)
            model = sm.OLS(endog = Y, exog = X).fit()
            R = model.rsquared
        
        
        if (R > RThreshold) & (len(pendingCols) > 0):
            removedCols = removedCols + pendingCols
            pendingCols = []
        
        loopCount += 1

        if len(tempFeatures) == 0:
          break
                

    return model, removedCols