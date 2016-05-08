__author__ = 'paulthompson'

import sys, json, pandas, numpy as np, copy, math, os, collections
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.svm import SVC
from scipy import sparse
from scipy.sparse import hstack

# This python file is used for creating content-based Steam game recommendations. The data was pulled from the Steam
# API and other sources. The functions below are for three main purposes: feature selection, model cross-validation,
# and recommendation creation.

game_path = ""
user_path = ""
appID_path = ""

def create_game_profile_df(game_data_path):
    '''
    This function is for game feature selection. Each game has a json of a variety of attributes including description.
    :param game_data_path: Game_path above. File with game json data.
    :return:
    '''
    print "Creating Game Profiles"
    gamecount = 0
    with open(game_data_path) as f:
        #Initialization
        gameFeatDicts = []; gameCount = 0; gameNameDict = {}; gameNameList = []; gameDescDict = {}; gameDescList = []

        #Game File Feature Extraction
        for line in f:
            gamecount += 1
            record = json.loads(line)
            if 'data' in record['details'][record['details'].keys()[0]].keys():
                gameFeatDict = {}
                if record['details'][record['details'].keys()[0]]['data']['type'] in ['demo','dlc','movie',
                                                                                      'advertising','video']:
                    continue
                # try:
                    # print record['details'][record['details'].keys()[0]]['data']
                gameFeatDict['steam_appid'] = record['appid']
                gameNameDict[record['appid']] = record['name']
                gameDescDict[record['appid']] = record['details'][record['details'].keys()[0]]['data']['detailed_description']
                # except:
                #     pass
                try:
                    gameFeatDict['mac'] = record['details'][record['details'].keys()[0]]['data']['platforms']['mac']
                except:
                    pass
                try:
                    gameFeatDict['windows'] = record['details'][record['details'].keys()[0]]['data']['platforms']['windows']
                except:
                    pass
                try:
                    gameFeatDict['linux'] = record['details'][record['details'].keys()[0]]['data']['platforms']['linux']
                except:
                    pass
                try:
                    gameFeatDict['type'] =  record['details'][record['details'].keys()[0]]['data']['type']
                except:
                    pass
                try:
                    gameFeatDict['releaseYear'] = int(record['details'][record['details'].keys()[0]]['data']['release_date']['date'][-4:])
                except:
                    pass
                try:
                    gameFeatDict['isFree'] = record['details'][record['details'].keys()[0]]['data']['is_free']
                except:
                    pass
                try:
                    gameFeatDict['metacriticScore'] = record['details'][record['details'].keys()[0]]['data']['metacritic']['score']
                except:
                    pass
                try:
                    gameFeatDict['developer'] = record['details'][record['details'].keys()[0]]['data']['developers'][0]
                except:
                    pass
                try:
                    gameFeatDict['requiredAge'] = int(record['details'][record['details'].keys()[0]]['data']['required_age'])
                except:
                    pass
                try:
                    categories = record['details'][record['details'].keys()[0]]['data']['categories']
                    allowedCategories = ['Single-player', 'Co-op', 'Multi-player', 'MMO', 'Local Co-op']
                    for category in categories:
                        if str(category['description']) in allowedCategories:
                            gameFeatDict[str(category['description'])] = 'True'
                except:
                    pass
                try:
                    gameFeatDict['fullPrice'] = \
                        record['details'][record['details'].keys()[0]]['data']['price_overview']['initial']
                except:
                    pass
                if record['tags'] == []:
                    try:
                        for genre in record['details'][record['details'].keys()[0]]['data']['genres']:
                            gameFeatDict[genre['description']] = 'True'
                    except:
                        pass
                if record['tags'] <> []:
                    try:
                        for tag in record['tags']:
                            gameFeatDict[tag] == 'True'
                    except:
                        pass
                gameFeatDicts.append(gameFeatDict)
                gameCount += 1
            else:
                pass

        vec = DictVectorizer()
        gameFeatures = vec.fit_transform(gameFeatDicts).toarray()
        gameFeaturesNames = vec.get_feature_names()
        gameFeaturesDF = pandas.DataFrame(gameFeatures, columns = gameFeaturesNames)
        gameFeaturesDF.index = gameFeaturesDF['steam_appid']
        for id in gameFeaturesDF.index:
            gameNameList.append(gameNameDict[id])
            gameDescList.append(gameDescDict[id])
        print "Game Count", gamecount

        print "Getting LDA features..."
        n_features = 2000
        n_topics = 20
        n_top_words = 20
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
        tf = tf_vectorizer.fit_transform(gameDescList)
        lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online', learning_offset=50.,
                                    random_state=0)
        gameDescTopics = sparse.coo_matrix(lda.fit_transform(tf))

        print "Transforming X and Y game content feature matrices.."
        gameFeaturesDF = gameFeaturesDF.drop(['steam_appid'], axis=1)
        Y_train = pandas.Series(np.zeros(len(gameFeaturesDF)), index = gameFeaturesDF.index)
        X_train_sparse = sparse.coo_matrix(gameFeaturesDF)
        X_train_sparse2 = hstack([X_train_sparse, gameDescTopics])

        X_train = pandas.DataFrame(X_train_sparse2.todense(), index=gameFeaturesDF.index)

        print "Finished Creating Game Profiles"
        return Y_train, X_train_sparse2, X_train

def getAppNames():
    '''
    Game attributes are kept track of by AppID throughout the modeling process. This function ties the AppID back to a
     name.
    :return:
    '''
    with open(appID_path) as f:
        records = json.load(f)
        IDNameDict = {}
        for game in records['applist']['apps']['app']:
            IDNameDict[str(game['appid'])] = game['name']
    return IDNameDict

def get_user_games(user_data_path, numToRetrieve):
    '''
    Gets game ids, names and playtimes from user's steam profile data
    :param user_data_path: file with json of user steam profile data
    :param numToRetrieve: Number of users to retrieve
    :return: list of user ids, list of user games
    '''
    print "Getting User Games"
    with open(user_data_path) as f:
        linecount = 0; userGamesList = {}; userIDList = []

        for line in f:
            record = json.loads(line)

            if record['ownedGames']['response'].keys():
                gamesPlayedList = []
                for game in record['ownedGames']['response']['games']:
                    if game['playtime_forever'] <> 0:
                        gameinfo = {}
                        gameinfo['playtime_forever'] = game['playtime_forever']
                        gameinfo['name'] = game['name']
                        print game['name']
                        gameinfo['appid'] = game['appid']
                        gamesPlayedList.append(gameinfo)
                linecount += 1
                userIDList.append(record['user'])
                userGamesList[record['user']] = gamesPlayedList

            if linecount == 1:
                return userIDList, userGamesList


def CrossValUsingLinReg(userGames, Y_train, X_train, X_train_sparse, gameAverages = None):
    '''
    This function is used for comparing models. It uses hold one out cross-validation where
    it keeps one game out of a user's games creates a model. It then counts the number of held out games
     recommended after going through all the user's games.
    :param userGames:
    :param Y_train:
    :param X_train:
    :param X_train_sparse:
    :param gameAverages:
    :return:
    '''
    IDNameDict = getAppNames()
    IDindexDict = {}
    # "Adding user playtime to Response"
    appIDs = []
    ErrorList = [2430]
    for game in userGames:
        if game['appid'] not in ErrorList:
            try:
                Y_train[game['appid']] = np.log(game['playtime_forever'] + 1.01)
                appIDs.append(game['appid'])
                # if game['playtime_forever'] > 60:
                #     Y_train[game['appid']] = 1.0
                #     appIDs.append(game['appid'])
            except:
                pass
    gameCount = len(appIDs)
    recAccuracyCount = 0

    for i, id in enumerate(Y_train.index):
        IDindexDict[id] = i

    print "Linear Regression Cross-Validation"
    print ""
    for i in range(gameCount):
        # "Splitting Out Training and Test Y and X"
        tempAppIDs = copy.copy(appIDs)
        del tempAppIDs[i]

        Y_temp_train = copy.deepcopy(Y_train)
        Y_temp_train[appIDs[i]] = 0
        X_test = X_train.drop(tempAppIDs, axis=0)
        unPlayedGames = list(X_test.index)
        X_test_sparse = sparse.coo_matrix(X_test)

        # "Fitting Linear Model"
        # regr = linear_model.ElasticNet(alpha=5.0)
        regr = SVC(kernel='linear', probability=True)
        regr.fit(X_train_sparse, Y_temp_train)

        # "Getting Predictions"
        predictions = list(regr.predict_proba(X_test_sparse)[:,1])
        gamePredictions = []
        for k in range(len(unPlayedGames)):
            gamePredictions.append([unPlayedGames[k],predictions[k]])

        print IDNameDict[str(appIDs[i])]
        topRecommendations = sorted(gamePredictions, key= lambda x: x[1], reverse = True)[0:50]

        if appIDs[i] in np.array(topRecommendations)[:,0]:
            print "Success"
            recAccuracyCount += 1
            print recAccuracyCount, "out of", i, "given total of", gameCount
            print ""
        else:
            print "Failure"
            RecList =[]
            for id in np.array(topRecommendations)[0:10,0]:
                RecList.append(IDNameDict[str(int(id))])
            print RecList
            print ""
    print ""
    print recAccuracyCount, "of the user's games out of", gameCount, "recommended."

def returnTopLinRegRecs(userGames, Y_train, X_train, X_train_sparse, nRecs = 30, printRecs = True):
    '''
    Returns recommendations for a user according specified model
    :param userGames:
    :param Y_train:
    :param X_train:
    :param X_train_sparse:
    :param nRecs:
    :return:
    '''
    IDindexDict = {}
    appIDs = []
    ErrorList = [2430]
    for game in userGames:
        if game['appid'] not in ErrorList:
            try:
                Y_train[game['appid']] = np.log(game['playtime_forever'] + 1.01)
                appIDs.append(game['appid'])
            except:
                pass

    for i, id in enumerate(Y_train.index):
        IDindexDict[id] = i

    print "Running Linear Regression"
    print ""

    X_test = X_train.drop(appIDs, axis=0)
    unPlayedGames = list(X_test.index)
    X_test_sparse = sparse.coo_matrix(X_test)
    regr = linear_model.LinearRegression()
    regr.fit(X_train_sparse, Y_train)
    predictions = list(regr.predict(X_test_sparse))
    gamePredictions = []
    for k in range(len(unPlayedGames)):
        gamePredictions.append([unPlayedGames[k], predictions[k]])
    topRecommendations = sorted(gamePredictions, key= lambda x: x[1], reverse = True)[0:nRecs]

    if printRecs:
        IDNameDict = getAppNames()
        topRecNames = []
        for id in np.array(topRecommendations)[:,0]:
            topRecNames.append(IDNameDict[str(int(id))])
        print topRecNames

    return topRecommendations


if __name__ == '__main__':
    userIDList, userGamesList = get_user_games(user_path, numToRetrieve = 2)
    PlayTimeZeros, GameDF_sparse, GameDF = create_game_profile_df(game_path)
    selectedUserGames = userGamesList[userIDList[0]]

    CrossVal = False
    if CrossVal:
        CrossValUsingLinReg(userGames=selectedUserGames, Y_train=PlayTimeZeros, X_train=GameDF,
                            X_train_sparse=GameDF_sparse)

    getRecs = True
    if getRecs:
        topRecs = returnTopLinRegRecs(userGames=selectedUserGames, Y_train=PlayTimeZeros, X_train=GameDF,
                                  X_train_sparse=GameDF_sparse, nRecs = 30)
