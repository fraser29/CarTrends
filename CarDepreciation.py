"""
DataSci project to understand how a car depreciates relative to its
review on www.autoexpress.co.uk

Read car data from www.autoexpress.co.uk
Read car prices from www.themoneycalculator.com


Fraser M. Callaghan
Janurary 2019
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import os
import sys
import pickle
import statsmodels.api as sm
# from PIL import Image
# from StringIO import StringIO

import plotly.offline as pyo
import plotly.graph_objs as go



## RESOURCES:
# Car review data: https://www.autoexpress.co.uk
# car depreciation data: https://www.themoneycalculator.com/vehicle-finance/calculators/car-depreciation-by-make-and-model

# Tips on scraping: https://blog.hartleybrody.com/web-scraping/

workingDir = os.path.join(os.path.expanduser('~'), 'CarDepreciation')
try:
    os.mkdir(workingDir)
except IOError:
    pass # dir exists
csvf = os.path.join(workingDir, 'data.csv')

def getDepreciationData():
    """
    Builds dictionary, key=make, value=list of model/depreciation_3yr_% tuples
    :return: dict
    """
    depreciation_root = 'https://www.themoneycalculator.com'
    depreciation_url = depreciation_root+'/vehicle-finance/calculators/car-depreciation-by-make-and-model'
    depreciation_html = urlopen(depreciation_url)
    autoexpress_soup = BeautifulSoup(depreciation_html, 'lxml')
    depreciationDict = {}
    allLinks = autoexpress_soup.find_all('a')
    modelsCount = 0
    for link in allLinks:
        linkName = link.get('href')
        # print(linkName)
        if ('car-depreciation-by-make-and-model' in linkName):
            iMake = linkName.split('/')[-2]
            if 'depreciation' in iMake:
                continue
            iMake_soup = BeautifulSoup(urlopen(depreciation_root+linkName), 'lxml')
            iMakeLinks = iMake_soup.find_all('a')
            for iMakeLink in iMakeLinks:
                iMakeLinkName = iMakeLink.get('href')
                if 'car-depreciation-by-make-and-model/'+iMake in iMakeLinkName:
                    iModel = iMakeLinkName.split('/')[-2]
                    try:
                        iModel_soup = BeautifulSoup(urlopen(depreciation_root+iMakeLinkName), 'lxml')
                    except HTTPError:
                        print('### Link not found for %s'%(depreciation_root+iMakeLinkName))
                        continue
                    iModelText = iModel_soup.find_all('p')
                    for iModeText_i in iModelText:
                        thisText = iModeText_i.text
                        if 'models on average depreciate' in thisText.lower():
                            IDdepreciate = thisText.find('depreciate ')
                            val = int(thisText[IDdepreciate+10:IDdepreciate+13])
                            print('%s %s - depreciation %d%%'%(iMake, iModel, val))
                            depreciationDict.setdefault(iMake, []).append((iModel, val))
                            modelsCount += 1
    print('Models found with depreciation values: %d'%(modelsCount))
    return depreciationDict


def getAutoExpressReviews():
    """
    Builds dictionary, key=make, value=list of model/rating tuples
    :return: dict
    """
    autoexpressroot_url = 'https://www.autoexpress.co.uk'
    autoexpress_url = autoexpressroot_url+'/car-reviews'
    autoexpress_html = urlopen(autoexpress_url)
    autoexpress_soup = BeautifulSoup(autoexpress_html, 'lxml')
    carMakeList = []
    allLinks = autoexpress_soup.find_all('a')
    for link in allLinks:
        linkName = link.get('href')
        try:
            if '/reviews' in linkName:
                carMakeList.append(linkName.split('/')[1])
        except TypeError:
            pass # may return none
    carMakeList = list(set(carMakeList))
    print('Found %d car makes'%(len(carMakeList)))

    carModels_dict = {}
    modelsCount = 0
    for iMake in carMakeList:
        iMakeReviews_url = autoexpressroot_url+'/'+iMake
        iMake_soup = BeautifulSoup(urlopen(iMakeReviews_url), 'lxml')
        all_links = iMake_soup.find_all('a')
        for link in all_links:
            linkName = link.get('href')
            try:
                linkParts = linkName.split('/')
                if (len(linkParts) == 3) & (linkParts[1] == iMake):
                    iModel = linkParts[2]
                    iModel_soup = BeautifulSoup(urlopen(autoexpressroot_url+'/'+iMake+'/'+iModel), 'lxml')
                    # iModel_soup = BeautifulSoup(urlopen('https://www.autoexpress.co.uk/kia/rio'), 'lxml')
                    reviewText = iModel_soup.get_text()
                    ratingValueID = reviewText.find('ratingValue":')
                    numberID = ratingValueID+13
                    # print(reviewText[ratingValueID:ratingValueID+15])
                    iModelRating = int(reviewText[numberID])
                    print('%s %s rating is : %d'%(iMake, iModel, iModelRating))
                    carModels_dict.setdefault(iMake, []).append((iModel, iModelRating))
                    modelsCount += 1
            except IndexError:
                pass # at the if statement - just haven't got to right link yet
            except ValueError:
                pass # No rating so error on the int conversion
            except AttributeError: # This is an explicit break - just noticed a link return None after all models listed
                break
    # print(carModels_dict)
    print('Models found with ratings: %d'%(modelsCount))
    return carModels_dict



def getWhatCarReviews():
    whatcar_url = 'https://www.whatcar.com'
    whatcar_html = urlopen(whatcar_url)
    whatcar_soup = BeautifulSoup(whatcar_html, 'lxml')
    carMakeList = []
    all_links = whatcar_soup.find_all("a")
    for link in all_links:
        linkName = link.get("href")
        if 'make/' in linkName:
            link_parts = linkName.split('/')
            carMakeList.append(link_parts[2])
    carMakeList = list(set(carMakeList))

    print('Carmakes found: %d'%(len(carMakeList)))
    carModels_dict = {}
    modelsCount = 0
    for iMake in carMakeList:
        print(iMake)
        iMakeReviews_url = whatcar_url+'/make/'+iMake
        iMake_soup = BeautifulSoup(urlopen(iMakeReviews_url), 'lxml')
        all_links = iMake_soup.find_all('a')
        for link in all_links:
            linkName = link.get('href')
            try:
                if '/review/' in linkName:
                    link_parts = linkName.split('/')
                    carModels_dict.setdefault(iMake, []).append('%s_%s'%(link_parts[2], link_parts[3]))
                    # get image to check stars
                    modelsCount += 1
            except TypeError:
                pass # sometimes a None type as link - just skip
        break
        # print(carModels_dict)
    print('Models found: %d'%(modelsCount))
    ## NOT COMPLETE
    # Review score not mentioned - just an image given.

def getData():

    ratingsPkl = os.path.join(workingDir, 'RatingsDict.pkl')
    depreciationPkl = os.path.join(workingDir, 'DepreciationDict.pkl')
    if not os.path.isfile(ratingsPkl):
        ratingsDict = getAutoExpressReviews()
        with open(ratingsPkl, 'wb') as fid:
            pickle.dump(ratingsDict, fid, protocol=2)
    else:
        if sys.version_info[0] == 3:
            opts = {"encoding":'latin1'}
        with open(ratingsPkl, 'rb') as fid:
            ratingsDict = pickle.load(fid, **opts)

    if not os.path.isfile(depreciationPkl):
        depreciationDict = getDepreciationData()
        with open(depreciationPkl, 'wb') as fid:
            pickle.dump(depreciationDict, fid, protocol=2)
    else:
        if sys.version_info[0] == 3:
            opts = {"encoding":'latin1'}
        with open(depreciationPkl, 'rb') as fid:
            depreciationDict = pickle.load(fid, **opts)
    rD2 = {}
    for ikey, val in ratingsDict.items():
        rD2[ikey.lower()] = val
    dD2 = {}
    for ikey, val in depreciationDict.items():
        dD2[ikey.lower()] = val

    ratingsDict = rD2
    depreciationDict = dD2
    return ratingsDict, depreciationDict

def buildCsv():
    ratingsDict, depreciationDict = getData()

    rMake = set(ratingsDict.keys())
    print(len(rMake))
    dMake = set(depreciationDict.keys())
    print(len(dMake))
    commonMake = rMake.intersection(dMake)
    print('%d makes with rating and depreciation value'%(len(commonMake)))
    likeModels = 0
    dfD = {}
    for iMake in commonMake:
        modelsRatings = [i[0].lower() for i in ratingsDict[iMake]]
        modelsDeprc = [i[0].lower() for i in depreciationDict[iMake]]
        for k1, iModel in enumerate(modelsRatings):
            if iModel in modelsDeprc:
                likeModels += 1
                dfD.setdefault('Make', []).append(iMake)
                dfD.setdefault('Model', []).append(iModel)
                dfD.setdefault('Rating', []).append(ratingsDict[iMake][k1][1])
                mDID = modelsDeprc.index(iModel)
                dfD.setdefault('Depreciation', []).append(depreciationDict[iMake][mDID][1])

    df = pd.DataFrame.from_dict(dfD)
    df.to_csv(csvf)
    print('%d models with rating nad depreciation value'%(likeModels))


def scatterInPlotly():
    df = pd.read_csv(csvf)

    makesWithMultRankings = [i for i in set(df['Make']) if sum(df['Make']==i)>15]
    print(makesWithMultRankings)
    c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 2, len(makesWithMultRankings))]
    allScatters = []
    xx = np.linspace(0,5,100)
    for k1, iMake in enumerate(makesWithMultRankings):
        idf = df[df['Make']==iMake]
        regressionModel = getRegression(idf)
        iScatter = go.Scatter(
            x=idf['Rating'],
            y=idf['Depreciation'],
            mode='markers',
            marker=dict(size=14,
                        line=dict(width=1),
                        color=c[k1],
                        opacity=0.7
                        ), name=iMake,
            text='Regression R-squared=%2.2f'%regressionModel.rsquared)  # The hover text
        allScatters.append(iScatter)
        x2 = sm.add_constant(xx)
        yy = regressionModel.predict(x2)
        print(iMake, yy[0], yy[55],yy[-1])
        iLine = go.Scatter(
            x=xx,
            y=yy,
            mode='lines',
            marker=dict(size=10,
                        line=dict(width=1),
                        color=c[k1],
                        opacity=0.3
                        ), name=iMake,
            text='Regression R-squared=%2.2f'%regressionModel.rsquared)  # The hover text
        allScatters.append(iLine)

    layout = go.Layout(
        title='Car Rating Depreciation Scatterplot',  # Graph title
        xaxis=dict(title='Ratings'),  # x-axis label
        yaxis=dict(title='Depreciation'),  # y-axis label
        hovermode='closest'  # handles multiple points landing on the same vertical
    )
    fig = go.Figure(data=allScatters, layout=layout)
    pyo.plot(fig, filename='CarDepreciation_withFit.html')

def scatterInPlotlyAll():
    idf = pd.read_csv(csvf)

    regressionModel = getRegression(idf)
    iScatter = go.Scatter(
        x=idf['Rating'],
        y=idf['Depreciation'],
        mode='markers',
        marker=dict(size=14,
                    line=dict(width=1),
                    opacity=0.7
                    ), name='All',
        text='Regression R-squared=%2.2f'%regressionModel.rsquared)  # The hover text
    allScatters = [iScatter]
    xx = np.linspace(0,5,100)
    x2 = sm.add_constant(xx)
    yy = regressionModel.predict(x2)
    iLine = go.Scatter(
        x=xx,
        y=yy,
        mode='lines',
        marker=dict(size=10,
                    line=dict(width=1),
                    opacity=0.3
                    ), name='All',
        text='Regression R-squared=%2.2f'%regressionModel.rsquared)  # The hover text
    allScatters.append(iLine)

    layout = go.Layout(
        title='Car Rating Depreciation Scatterplot',  # Graph title
        xaxis=dict(title='Ratings'),  # x-axis label
        yaxis=dict(title='Depreciation'),  # y-axis label
        hovermode='closest'  # handles multiple points landing on the same vertical
    )
    fig = go.Figure(data=allScatters, layout=layout)
    pyo.plot(fig, filename='CarDepreciation_withFit.html')

def getRegression(iidf):
    X = iidf["Rating"]
    y = iidf["Depreciation"]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    return model

def analysis():
    df = pd.read_csv(csvf)
    print(df.columns)
    df2 = df.groupby('Make')
    colsWithMany = [i for i in set(df['Make']) if sum(df['Make']==i)>50]
    print(colsWithMany)
    # sns.lmplot(x="Rating", y="Depreciation", data=df, fit_reg=False, hue='Make', legend=False)
    # plt.legend(loc='lower right')
    # plt.show()
    #
    # df.boxplot('Depreciation','Rating')
    # plt.show()

    for iMake in colsWithMany:
        idf = df[df['Make']==iMake]

        model = getRegression(idf)
        # print(model.summary())
        print(iMake, model.rsquared)



def main():
    buildCsv()
    analysis()
    # scatterInPlotly()
    scatterInPlotlyAll()


if __name__ == '__main__':
    main()




