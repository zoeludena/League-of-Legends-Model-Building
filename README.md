# League-of-Legends-Model-Building
This is a project for DSC80 where we create different models using DecisionTreeClassifier(), sklearn, python, and pandas. We also conduct fairness analysis on the model using permutation testing.

**Names:** Zoe Ludena and Johel Tutak

If you would like to look at out previous project where we analyzed League of Legends Souls affect on winrate look here: <a href="https://zoeludena.github.io/League-Of-Legends-Soul-Analysis/">League-of-Legends-Soul-Analysis</a>.

## Cleaning our Data
Our dataset is on all of the professional League of Legends games that have taken place in 2023. The dataset contains 12 rows per game, one row per player and 2 rows of summary statistics (one for each team). Furthermore, there are over 100 columns of nearly all the data you could collect on a League of Legends match.

We took this dataset and isolated columns to help us answer our prediction problem. We kept: `"result"`, `"cspm"`, `"goldat10"`, `"xpat10"`, `"goldat15"`, `"xpat15"`, and `"vspm"`. Furthermore, we only kept the summary rows for both the winning and the losing team. Below you can find what these columns mean and the first five rows of our DataFrame.

**First five rows of our cleaned DataFrame:**

|   result |    cspm |   goldat10 |   xpat10 |   goldat15 |   xpat15 |   vspm |
|---------:|--------:|-----------:|---------:|-----------:|---------:|-------:|
|        1 | 29.5406 |      14612 |    18745 |      22384 |    29220 | 7.1669 |
|        0 | 28.1623 |      14537 |    18901 |      22914 |    30891 | 6.9832 |
|        0 | 32.9064 |      15969 |    19120 |      24771 |    30084 | 7.3399 |
|        1 | 33.5714 |      16330 |    18838 |      24098 |    29554 | 8.1281 |
|        1 | 33.5455 |      14794 |    17060 |      22945 |    27423 | 8.8485 |

**Columns:**
- `"result"` is the outcome of a League of Legends match. It is a 1 to symbolize a win or a 0 to symbolize a loss. There is a win and loss per game.
- `"cspm"` is the creep score per minute. A creep is a minion to be killed for gold.
- `"goldat10"` is the gold the team possessed at the 10 minute mark of the game.
- `"xpat10"` is the xp the team possesed at the 10 minute mark of the game. Xp is experience and allows the champions to level up, which allows for stronger abilities.
- `"goldat15"` is the gold the team possessed at the 15 minute mark of the game.
- `"xpat15"` is the xp the team possessed at the 15 minute mark of the game. XP is experience and allows the champtions to level up, which allows for stronger abilities.

## Problem Identification
Our prediction problem is trying to predict whether a team won their game of League of Legends, given only their post game statistics (the two rows corresponding to each team in the DataFrame), excluding the results obviously. Note that we do not include the results of the other team in our analysis, as we want to focus on only this particular team’s statistics. This is a binary classification problem, with the two outcomes being a win or a loss. Our response variable is the “result” column, which is 1 if a team won a game, and 0 otherwise. We chose this because we found predicting the winner of a match the most interesting among the samples at the bottom of the instructions, and the most applicable to our gaming tendencies. We are evaluating our model through accuracy. We did not find the F1-score to be particularly important in this scenario, as there are no repercussions for an imbalanced amount of Type I and Type II errors.

## Baseline Model
For our Baseline model we took our cleaned DataFrame and further isolated two features. The features of our model are `"cspm"`, meaning the total creep score per minute, and `"xpat10"`, meaning the xp at 10 minutes. We converted `"xpat10"` to xp gained per minute in a new column called `"xp_10_1"`. Below you will find the first five rows of the data we used in our Baseline Model where we tested `"cspm"` and `"xp_10_1"` and tried to predict `"result"`.

|   result |    cspm |   xp_10_1 |
|---------:|--------:|----------:|
|        1 | 29.5406 |    1874.5 |
|        0 | 28.1623 |    1890.1 |
|        0 | 32.9064 |    1912   |
|        1 | 33.5714 |    1883.8 |
|        1 | 33.5455 |    1706   |

`"cspm"` and `"xp_10_1"` are both quantitative variables, and therefore we did not need to encode any of our features. Our model had an accuracy of 57.79% on our test data, which while not ideal still gives us a reasonably higher chance to predict a winner given a results screen. We do not believe that the current model is “good,” as it is missing many key features that are available to us, and can greatly improve the accuracy of our model, such as the gold difference at difference times, and the xp at 15, which might prove to be more useful than xp at 10, as it is further along in the match.

## Final Model
### Hyperparameters
The modeling algorithm we chose was a decision tree, as we have a classification problem. The hyperparameters that worked the best for our model were max_depth = 20, `train_test_split` test_size = .25, and criterion = “gini”. Our best `train_test_split` test_size kept choosing values from 0.25 to 0.5 (so 0.25, 0.3, 0.35, 0.4, 0.45, or 0.5) as the most optimal values between runs, likely due to variance in the different training data we got for each run. The other values stayed constant, with max_depth staying at 20 and criterion staying at "gini." We found our optimal hyperparameters through a self implemented, slightly modified version of grid search. We took each permutation of max_depth, `train_test_split` test_size, and criterion, found the test error, and then chose the permutation with the lowest test error.

**Here are our desired hyper parameters as shown in a DataFrame:**

|   train_err |   tree_depth |   test_size | criterion   |
|------------:|-------------:|------------:|:------------|
|   0.0646074 |           20 |         0.4 | gini        |

### Features
The features we added were `"goldat10"`, `"goldat15"`, `"xpat15"` and `"vspm"`. `"goldat10"` and `"goldat15"` is the amount of gold the team had after 10 and 15 minutes respectively, and `"xpat15"` is the amount of xp the team had at the 15 minute mark. `"vspm"` is the vision score per minute, which roughly corresponds to their wards placed and vision control over the map throughout the game. We used all of these columns to help us predict `"result"`.

**Here is what our DataFrame looks like:**

|   result |    cspm |   goldat10 |   xpat10 |   goldat15 |   xpat15 |   vspm |
|---------:|--------:|-----------:|---------:|-----------:|---------:|-------:|
|        1 | 29.5406 |      14612 |    18745 |      22384 |    29220 | 7.1669 |
|        0 | 28.1623 |      14537 |    18901 |      22914 |    30891 | 6.9832 |
|        0 | 32.9064 |      15969 |    19120 |      24771 |    30084 | 7.3399 |
|        1 | 33.5714 |      16330 |    18838 |      24098 |    29554 | 8.1281 |
|        1 | 33.5455 |      14794 |    17060 |      22945 |    27423 | 8.8485 |

We chose to include `"goldat10"`, `"goldat15"` and `"xpat15"` as they are all statistics of how many resources the teams collected throughout the game. We hypothesized that the greater the amount of resources, the greater a teams chance of winning, and therefore these were relevant features to include. We included `"vspm"` as this would indicate how well a team had control over the match. A team with a higher vspm would likely have greater ability to make plays around the map, meaning they would have a higher chance of winning. We did some feature engineering to convert things recorded at 10 and 15 minutes to minutes to keep all of our time variables the same. We did this using a pipeline, which is why one does not see that in our DataFrame above. We chose statistics that were per minute or at certain intervals to avoid bias from game length, as if it was just overall gold, overall xp, and overall vision score, then higher numbers would have a high correlation with longer games, reducing the accuracy of our model.


### Final Model VS Baseline Model
Our Final model had an accuracy of roughly 67.84%, which is a roughly a 10% increase from our Baseline model's accuracy of 57.79%. This is a substantial improvement to our baseline, giving us a ⅔ chance to get the correct result. This increase is likely due to access to a wider range of data, allowing for the decision tree to make more accurate decisions based on more variables/information.

### Fairness Analysis
Our group X is teams that got the dragon soul for that game, and our group Y is teams that did not have dragon soul. Our evaluation metric is whether our classifier’s accuracy is the same between the two groups. The null hypothesis is “Our classifier's accuracy is the same for both if a team had dragon soul or if they did not have dragon soul. Any difference was due to random chance.” The alternative hypothesis is “Our classifier's accuracy is not the same. Its accuracy for if the team had soul was greater than if the team did not have soul.” The test statistic we chose is the difference in accuracy between the two groups, with a significance level of 0.05. Our resulting P-value was 0.682, meaning we failed to reject the null hypothesis, and that any difference between the two groups can be explained by random chance. There is no evidence that the classifier's accuracy is different depending on if the team got soul or did not get soul.
