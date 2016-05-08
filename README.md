## Steam Game Content-based Recommendations

### Intro
The purpose of this project was to create the best list of recommendations possible given a user's game playing history on Steam and the attributes of games. I used cross-validation to compare a number of different models and combinations of game features. 

### Data
A fellow masters student at USF pulled a wide variety of attributes for games from the Steam API including achievements, price, genres, developers, publishers, free to play, categories, year published, game description, name, and appID. Information for users included a list of games and play time history in their libraries.

### Method
A single user was selected to compare model variations. The user was selected based on their relatively large amount of games with play times above zero (209). The initial set of features tested include operating system, release year, price, metacritic’s score, developer, required age, application category/type: 'Game', developer-designated genre(s), user-designated genre(s). The goal of testing was to find features and models that would be most likely to recommend games the user likes. Therefore, hold-one-out cross-validation was used with a model trained for each of the user’s 209 games with each game held out of the training set in turn. The number of games in the top 50 recommended out of the total was recorded.

### Experiment Results
The following are key experiment results (not all experiments were included for brevity):


| Model Type  | Experiment Description                                           | # Rec. | Out of |
|-------------|------------------------------------------------------------------|--------|--------|
| Lin. Regr.  | Initial Features                                                 | 16     | 209    |
| Lin. Reg.   | Including Unigram/Bigram TF/IDF on Titles                        | 14     | 209    |
| Lin. Reg.   | Latent Dirichlet Allocation on description with TF/IDF on Titles | <22    | 209    |
| Lin. Reg.   | Latent Dir. Allocation on description (Used going forward)       | 22     | 209    |
| Lin. Reg.   | Response: Playtime / Average game playtime across users          | 18     | 209    |
| Lin. Reg.   | Response: Log(Playtime / Average + 1.01)                         | 36     | 209    |
| Lin. Reg.   | Response: Log(playtime + 1.01)                                   | 45     | 209    |
| Elastic Net | Response: Log(playtime + 1.01)                                   | 38     | 209    |
| Log. Reg.   | (C=1.0) - Response: 1 if playtime > 30                           | 18     | 201    |
| Log. Reg.   | (C=1.0) - Response: 1 if playtime > 60                           | 19     | 195    |
| Log. Reg.   | (C=50.0) - Response: 1 if playtime > 60                          | 41     | 195    |
| SVM         | (C=50.0; Kernel = RBF),- Response: 1 if playtime > 60            | 2      | 25     |


### Conclusion
Two different models were the best and did close to equally well: Logistic Regression with very little regularization and Linear Regression with a response of Log(playtime + 1.01). Any model with regularization did badly. SVMs (tried both linear and RBF kernels) took a long time to run and produced not great results. The chosen model from the experiments is Linear Regression with the log-transformed response. It deals with the long tail (some play times can be much higher than others) while shifting weights towards the kinds of games that the user plays more often.