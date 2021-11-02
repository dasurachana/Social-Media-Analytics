"""
Social Media Analytics Project
Name:
Roll Number:
"""

import hw6_social_tests as test

project = "Social" # don't edit this

### PART 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]
df={}

'''
makeDataFrame(filename)
#3 [Check6-1]
Parameters: str
Returns: dataframe
'''
def makeDataFrame(filename):
    political_Data_df=pd.read_csv(filename)
    return political_Data_df


'''
parseName(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseName(fromString):
    start = fromString.find(" ")
    startSlicing=fromString[start:]
    end = startSlicing.find("(")
    name=startSlicing[:end]
    name=name.strip()
    #print(name)
    return name



'''
parsePosition(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parsePosition(fromString):
    position_Start=fromString.find("(")
    positionSlicing=fromString[position_Start+1:]
    position_End=positionSlicing.find(" from")
    position=positionSlicing[:position_End]
    position=position.strip()
    #print(position)
    return position


'''
parseState(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''

def parseState(fromString):
    state_Start=fromString.find("from")
    stateSlicing=fromString[state_Start+4:]
    state_End=stateSlicing.find(")")
    state=stateSlicing[:state_End]
    state=state.strip()
    #print(state)
    return state


'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    tags=[]
    message_List= message.split("#")
    for each in message_List[1:]:
        n_Str=""
        for char in each:
            if char in endChars:
                break
            n_Str=n_Str+char
        n_Str="#"+n_Str
        tags.append(n_Str)
    return tags
'''
getRegionFromState(stateDf, state)
#6 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    #print(stateDf)
    region=stateDf.loc[stateDf['state']==state,'region']
    #print(region)
    return region.values[0]

'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    names=[]
    positions=[]
    states=[]
    regions=[]
    hashtags =[]
    for index, row in data.iterrows():
        row_Value= row["label"]
        names.append(parseName(row_Value))
        positions.append(parsePosition(row_Value))
        states.append(parseState(row_Value))
        regions.append(getRegionFromState(stateDf, parseState(row_Value)))
        textValue= row["text"]
        hashtags.append(findHashtags(textValue))
    data["name"]=names
    data["position"]=positions
    data["state"]=states
    data["region"]=regions
    data["hashtags"]=hashtags
    return

### PART 2 ###

'''
findSentiment(classifier, message)
#1 [Check6-2]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if score < -0.1 :
        return "negative"
    elif score > 0.1:
        return "positive"
    else:
        return "neutral"

'''
addSentimentColumn(data)
#2 [Check6-2]
Parameters: dataframe
Returns: None
'''
def addSentimentColumn(data):
    
    classifier = SentimentIntensityAnalyzer()
    sentiments=[]
    for index, row in data.iterrows():
        find_Sentiment=findSentiment(classifier,row["text"])
        sentiments.append(find_Sentiment)
    data["sentiment"]=sentiments
    #print(data["sentiment"])
    return


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    dict_count={}
    # print(data["state"])
    if dataToCount=="" and colName=="":
        for index,row in data.iterrows():
            if row["state"] not in dict_count:
                dict_count[row["state"]] = 1
            else:
                dict_count[row["state"]]+=1
    else:
        for index,row in data.iterrows():
            if dataToCount == row[colName] :
                if row["state"] not in dict_count:
                    dict_count[row["state"]] = 1
                else:
                    dict_count[row["state"]]+=1
    return dict_count

'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    nested_dict={}
    for index,row in data.iterrows():
        nested_dict[row["region"]]={}
       #print(row[colName])
    for index,row in data.iterrows():
        if row[colName] not in nested_dict[row["region"]]:
            nested_dict[row["region"]][row[colName]]=1
        else:
            nested_dict[row["region"]][row[colName]]+=1
    return nested_dict

'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    dict_Hashtags={}
    for index,row in data.iterrows():
        for each in row["hashtags"]: 
            if each not in dict_Hashtags.keys():
                dict_Hashtags[each]=1
            else:
                dict_Hashtags[each]+=1
    return dict_Hashtags


'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    common_Hashtags={}
    most_Common_Hashtag={}
    common_Hashtags=sorted(hashtags.items(),key= lambda x:x[1],reverse=True)
    #print(common_Hashtags)
    for each in common_Hashtags[0:count]:
        most_Common_Hashtag[each[0]]=each[1]
    return most_Common_Hashtag
'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    hashtag_list=[]
    count=0
    all_Sentiments=[]
    for index,row in data.iterrows():
        # print(row["sentiment"])
        hashtag_list=findHashtags(row["text"])
        if hashtag in hashtag_list:
            count+=1
            hashtag_Sentiment= row["sentiment"]
            if hashtag_Sentiment == "positive" : all_Sentiments.append(1)
            elif hashtag_Sentiment == "negative" : all_Sentiments.append(-1)
            else : all_Sentiments.append(0)
    avg_Hashtag_Sentiment= sum(all_Sentiments)/count
    
    return avg_Hashtag_Sentiment



### PART 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    return


'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    return


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    return


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    return


#### PART 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":

    # test.testMakeDataFrame()
    # test.testParseName()
    # test.testParsePosition()
    # test.testParseState()
    # test.testFindHashtags()
    # test.testGetRegionFromState()
    # test.testAddColumns()
    # test.testFindSentiment()
    # test.testAddSentimentColumn()
    df = makeDataFrame("data/politicaldata.csv")
    stateDf = makeDataFrame("data/statemappings.csv")
    addColumns(df, stateDf)
    addSentimentColumn(df)
    # print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    # test.week1Tests()
    # print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    # test.runWeek1()
    test.testGetDataCountByState(df)
    test.testGetDataForRegion(df)
    test.testGetHashtagRates(df)
    test.testMostCommonHashtags(df)
    test.testGetHashtagSentiment(df)

    ## Uncomment these for Week 2 ##
    """print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()"""

    ## Uncomment these for Week 3 ##
    """print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()"""
