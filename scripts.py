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
    # for numerical data we use the mean to replace the null values
    if numericalData == True:
        for x in columns:
            fillIn = df[x].mean()
            df[x] = df[x].fillna(fillIn)
            if showInfo == True:
                display(f'Column {x} now has {df[x].isna().sum()} null values')
                display(f'The mean of column {x} = {fillIn}')
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
                display(f'The mean of column {x} = {fillIn}')
                display(df[x].value_counts())

def dropNumericOutliers(df):
    '''
        This function takes in a pandas dataframe and operates on the numerical data columns to remove column outliers
        that lie more than three standard deviations away from the column mean. Note that the outliers are removed at the dataframe
        so there is a potential to exclude a large percentage of data if the row indecies don't overlap for the extream values in each column
    '''
    outlierIndicies = []
    try:
        dfNumeric = df.select_dtypes(['int64','float64']).drop(columns = 'id')
    except:
        dfNumeric = df.select_dtypes(['int64','float64'])
    for x in dfNumeric.columns:
        mean = dfNumeric[x].mean()
        std = dfNumeric[x].std()
        indecies = dfNumeric.loc[(dfNumeric[x] < mean - 3*std) | (dfNumeric[x] > mean + 3*std)].index
        print(indecies)
        for index in indecies:
            if index in outlierIndicies:
                pass
            else:
                outlierIndicies.append(index)
    print(outlierIndicies)
    print('\n')
    print(f'Combined all the outliers make up {dfNumeric.shape[0]/len(outlierIndicies)}% of the data')
    
    return df.drop(index = outlierIndicies)

def removeFeatures(df,features, target, rThreshold = .8, pThreshold = .05):
    '''
        The removeFeatures function takes in a dataframe with lists of features and target variables. The function creates a linear regression
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
    while True:
        # Push any columns with insignificant pvalues from the previous iteration into the pending bucket
        badP = model.pvalues[tempFeatures].loc[model.pvalues > .05]
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
                

    return model, removedCols