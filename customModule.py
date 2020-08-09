#Python file containing all custom functions used in the mod2.ipyn file
import statsmodels.api as sm
import numpy as np
import pandas as pd
import math

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
    # for numerical data we use the mean to replace the null values
    if numericalData == True:
        for x in columns:
            fillIn = df[x].median()
            df[x] = df[x].fillna(fillIn)
            if showInfo == True:
                display(f'Column {x} now has {df[x].isna().sum()} null values')
                display(f'The median of column {x} = {fillIn}')
                display(df[x].value_counts())
    # For categorical data we fill in the string 'Null'
    else:
        for x in columns:
            df[x] = df[x].astype('object')
            fillIn = 'Null'
            df[x] = df[x].fillna(fillIn)
            df[x] = df[x].astype('category')
            if showInfo == True:
                display(f'Column {x} now has {df[x].isna().sum()} null values')
                display(f'The median of column {x} = {fillIn}')
                display(df[x].value_counts())

def dropNumericOutliers(df, stdRange = 3, outliersPerRow = 1):
    '''
        This function takes in a pandas dataframe and operates on the numerical data columns to remove column outliers
        that lie more than three standard deviations away from the column mean. Note that the outliers are removed at the dataframe
        so there is a potential to exclude a large percentage of data if the row indecies don't overlap for the extream values in each column
    '''
    try:
        dfNumeric = df.select_dtypes('number').drop(columns = 'id').copy()
    except:
        dfNumeric = df.select_dtypes('number').copy()
        print('id column was not detected and was not dropped')
    indiciesRemoved = []
    for x in dfNumeric.columns:
        mean = dfNumeric[x].mean()
        std = dfNumeric[x].std()
        outliers = dfNumeric[x].loc[(dfNumeric[x] < mean - stdRange*std) | (dfNumeric[x] > mean + stdRange*std)].copy()
        for y in outliers.index:
            indiciesRemoved.append(y)
        # numberCanRemove = math.floor(dropThreshold*dfNumeric.shape[0])
        # numberRemoved = 0
        

    # print(indiciesRemoved)
    dictionaryOfIndiceies = {}
    for x in indiciesRemoved:
        dictionaryOfIndiceies[x] = dictionaryOfIndiceies.get(x, 0) + 1

    
        
    # display(dictionaryOfIndiceies) 
    dropIndex = []
    for key,value in dictionaryOfIndiceies.items():
        if value > outliersPerRow:
            dropIndex.append(key)
    print(f'The input parameters caused a {round(len(dropIndex)/df.shape[0]*100,2)}% drop of the total rows.')
    df = df.drop(index = dropIndex)
    return df

def reduceModel(df,features, target, rThreshold = .8, pThreshold = .05, removeFeatures = False):
    '''
        The reduceModel function takes in a dataframe with lists of features and target variables. The function creates a linear regression
        model and subsequently performs a step-by-step removal of features from the original model until the point at which removing any additional
        will result is a degredation of the model below the desired R_squared value. Features are removed on a basis of the magnitude of their
        linear coefficients. After the feature is removed, the model is run again to confirm the integrity of the model. In the event that removing 
        a feature increases the p-values of other columns due to marginal coolinearity, those columns will be removed in addition to the feature with
        the least impactful coefficient.  Should the model fall below the quality threshold, the last removed feature is re-added to the 
        model but excluded from the list of feature eliagable for removal.

        Parameter:
            df - A pandas DataFrame object (pd.DataFrame()).
        
            features - A non-empty list of columns in the provided dataframe that 
                       will be treated as the independant variables in the linear models.
            
            target - A non-empty list of columns in the provided dataframe that
                     will be treated as the independant variables in the linear models.
            
            rThreshold - A float that represents the bottom threshold limit for the 
                         integrity of the linearmodel.
            
            pThreshold - A float that represents the significance threshold for removing
                         one of the features from the model without regards to that features
                         coefficient. 

        Returns:
            model - A linear model from statsmodels.api with the final list of features.

            removedCols - A list of all the columns that were removed during the fuction operation. 

    '''
    # Create a copy of the input array so we don't change the original
    tempFeatures = features.copy()

    #create buckets for the filtering process
    succeededCols = []
    removedCols = []
    pendingCols = []

    # Run the first model to establish an entry point into the loop
    Y = df[target]
    X = df[tempFeatures]
    X = sm.add_constant(X)
    model = sm.OLS(endog = Y, exog = X).fit()
    R = model.rsquared
    print(f' The starting R-value for the model is {R}')
    print('\n')

    # Begin the loop to remove columns
    while removeFeatures == True:
        # Push any columns with insignificant pvalues from the previous iteration into the pending bucket
        badP = model.pvalues[tempFeatures].loc[model.pvalues > pThreshold]
        if len(badP) > 0:
            print(f'The following columns have p-values above the threshold of {pThreshold}: {list(badP.index)}\n')
            for x in list(badP.index):
                tempFeatures.remove(x)
                pendingCols.append(x)

        # Find the parameter from the previous iteration with the smallest magnitude coeff that is in the tempFeatures bucket
        modelParams = model.params.drop(index = ['const']+pendingCols+succeededCols)
        leastImpactfulCoeff = modelParams.loc[abs(modelParams) == abs(modelParams).min()]
        leastImpactful = leastImpactfulCoeff.index[0]
        print(f'{leastImpactful} was removed with a coefficient of {leastImpactfulCoeff[0]}\n' )
        # Remove the smallest magnitude parameter from the tempFeatures bucket and push it into the pending bucket. 
        tempFeatures.remove(leastImpactful)
        pendingCols.append(leastImpactful)
        
        #Run the model again with the tempFeatures and succeded bucket. 
        Y = df[target]
        X = df[tempFeatures+succeededCols]
        X = sm.add_constant(X)
        model = sm.OLS(endog = Y, exog = X).fit()
        R = model.rsquared
        
        # When the model fails to pass the set threshold, add the last param that was pushed into the pending bucket into the succeded bucket
        # we do this because despite the magnitude of the param, removing it caused an unacceptable drop in rsquared and thus is is likely
        # important to include. 
        if( R < rThreshold) & (len(pendingCols) > 0):
            print(f'''The R-squared value has dropped to {R} which is below the acceptable threshold of {rThreshold}. Adding feature: {pendingCols[-1]} back into the model and continuing \n\n\n''')
            succeededCols.append(pendingCols.pop())
            pendingCols = []
            # Run the model again having added in the last column pushed into the pending bucket. 
            Y = df[target]
            X = df[tempFeatures+succeededCols]
            X = sm.add_constant(X)
            model = sm.OLS(endog = Y, exog = X).fit()
            R = model.rsquared
        
        #If running the model without the params in the pending column didn't cause an unacceptable drop in rsquared then we purge the pending bucket
        if (R > rThreshold) & (len(pendingCols) > 0):
            removedCols = removedCols + pendingCols
            pendingCols = []
        
        # When all columns have been sorted into the removed bucket or succeeded bucket we will the loop
        if len(tempFeatures) == 0:
          break

    # one last check to make sure that there aren't any insignificant pvalues in our model
    badP = model.pvalues[tempFeatures+succeededCols].loc[model.pvalues > pThreshold]
    while len(badP) > 0:
        print(f'The following columns have p-values above the threshold of {pThreshold}: {list(badP.index)}\n')
        for x in badP.index:
            if removeFeatures == False:
                tempFeatures.remove(x)
                removedCols.append(x)
            else:
                tempFeatures+succeededCols.remove(x)
                removedCols.append(x)
        Y = df[target]
        X = df[succeededCols+tempFeatures]
        X = sm.add_constant(X)
        model = sm.OLS(endog = Y, exog = X).fit()
        R = model.rsquared
        badP = model.pvalues[tempFeatures+succeededCols].loc[model.pvalues > pThreshold]

    return model, removedCols