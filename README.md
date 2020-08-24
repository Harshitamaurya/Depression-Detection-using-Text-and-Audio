# Depression-Detection-using-Text-and-Audio
The project is implemented in two parts that are as follows: 
 
PART I.  Depression Detection through Text Analysis 
 
In this we aimed at detecting whether the person is depressed or not via social media. These days people are very active on online platforms such as Twitter and usually express their thoughts through messages and blogs they most. Therefore, we considered this data generated as a source for analysing the presence of depression in them. It was implemented in following phases: 
 
 Data Collection: The dataset has been generated combining part of the Sentiment140 [6] (8,000 positive tweets), and another one for depressive tweets (2,314 tweets), with a total of 10,314 tweets. The Sentiment140 dataset contains 1,600,000 tweets extracted using the twitter API. The tweets have been annotated (0 = negative, 2 = neutral, 4 = positive) and they can be used to detect sentiment. It contains the following 6 fields: target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive) ids: The id of the tweet (2087) date: the date of the tweet (Sat May 16 23:58:44 UTC 2009) flag: The query. If there is no query, then this value is NO_QUERY. user: the user that tweeted. text: the text of the tweet. 
 
For our experiment, we have taken a sample of 8,000 tweets with polarity of 4, the positive ones. Depressive Tweets have been Web scraped with TWINT [7]. The entire data has been extracted from [8]. The final dataset has three attributes tweet-id, tweet content and label that is 0 or 1 depending on whether the tweet is positive or negative respectively. 
 
 Word-Cloud Analysis: A word-cloud is created using the python in-built function for depressive and positive words by including all the words present in tweets with label 1 and label 0 respectively. 
 
 Data Pre-processing: Data pre-processing has been done for stemming and removing stop words and then tokenizing the tweets by using whitespace as the delimiter. As the keyword matching strategy is widely used, words must have unified representations regardless of the tense and voice. For example-―marrying and ―married should be represented as ―marri uniformly. For the purpose of stemming porter stemming algorithm has been used.                                                   5 

 
 Classification: Using Bayes theorem for analysing if a person writing a particular tweet can be diagnosed with depression or not. The probabilities have been calculated using the probability of each word in a tweet to belong to a particular category and then multiple importance of word. 
 
 
PART II.  Depression Detection through Audio Features 
 
Though the social media provides a means to capture an individual‟s present state of mind, but sometimes a person‟s feeling or thoughts might depend on one or other indirect causes and thus this data can‟t solely be used for depression detection. Therefore, we extended our approach by further analysing the audio features of each individual being interviewed by a virtual interviewer called Ellie, controlled by a human interviewer in another room. Each of the session lasted for an average of 16 minutes. Also, prior to the interview, each participant completed a psychiatric questionnaire (PHQ-8) [5], from which a binary "truth" classification (depressed, not depressed) was derived. It was implemented in following phases: 
 
 Data Collection: The data collected include audio and video recordings and extensive questionnaire responses. Data has been transcribed and annotated for a variety of verbal and non-verbal features. All audio recordings and associated depression metrics were provided by the DAIC-WOZ Database [9], which was compiled by USC's Institute of Creative Technologies and released as part of the 2016 Audio/Visual Emotional Challenge and Workshop (AVEC 2016). The dataset consists of 189 sessions. The audio data provided by AVEC consisted of an audio file of entire interview with the participant, pre-extracted features using the COVAREP toolbox at 10-ms intervals over the entire recording (F0, NAQ, QOQ, H1H2, PSP, MDQ, peak Slope, Rd, Rd conf, MCEP 0-24, HMPDM 1-24, HMPDD 1-12, and Formants 1-3), Transcript file containing speaking times and values of participant and the virtual interviewer, and a formant file. 
 
 Data Pre-processing: Data pre-processing has been done for removing the rows that are having 50% or more of the values as zeroes as it is of no use. Further, PHQ-8 score column is added in each of the files for training of the model.  
 
 Handling Large Data: Data has been divided into 11 separate folders for training purpose. Each folder is separately trained on the model and overall results are averaged for testing purpose. Only 10 percent of the data has been utilised for training purposes which was randomly selected from the data of every patient. Also, the data types of data         
are reduced from 64- bit float to 32-bit float. After training every model data frame are removed to evacuate the space so that enough space is available on RAM for the training purpose of each of the folders generated. 
 
 Data Imbalance: Data pre-processing has been done for removing the rows with having 50% or more of the values as zeroes as it is of no use. Further,189 sessions of interactions were ranging between 7-33min (with an average of 16min) therefore, biasing can occur. Also, a larger volume of signal from an individual may emphasize some characteristics that may be specific to a person. In the data-set, the number of non-depressed subjects is about four times larger than that of depressed ones. To rectify this imbalance, undersampling has been done on the dataset. 
 
Data Correlation: Correlation matrix is generated to identify the relationship between the different audio features and how they can impact each other. The correlation coefficient values obtained lies between 0 and 0.4 (Fig-6.7). This indicates that features are independent of each other. Further, the impact of each of the features on the target variable of score prediction is also analysed (Fig. 6.8). 
 
 Classification: Using Support Vector Regression (kernel-sigmoid), Support Vector Regressor with (kernel linear), Random Forest Regressor with 40 estimators and Random Forest Regressor with 400 estimators, for predicting the PHQ-8 score of the tested participants against the proposed model. Further, for analysing the accuracy of the model the hypothesis is considered that person with depression scale value>=10 is depressed and otherwise not depressed. The binary classification column is added addressing the depressed person as “1”.The measure of how well the model performs was taken as the similarity in results of the model implemented and the questionnaire. If majority of the answers in the questionnaire in a domain were marked” yes” and the model too gave a higher value for that category, then the case was taken as positives. Any conflict resulted in a negative case. The models have been evaluated by measuring the root mean square error and mean absolute error when predicting the PHQ8 score of a patient (Fig-6.9). 
 
 Implementation 
 
 We aim at providing a multimodal feature extraction and decision level fusion approach for depression detection. Thus, we have considered depression detection analysis using both text and speech.  
 
 Approach for Depression Detection through Text Analysis: The project starts with collection of dataset and then preprocessing the data for removing stop words and stemming. Word cloud an in-built python library has been used to create clusters of similar words from depressive (Fig-6.1) and positive tweets (Fig-6.2). Then Bayes theorem has been applied for analyzing if a person writing a particular tweet can be diagnosed with depression or not. Bag of words and Term frequency – Inverse Term Frequency have been used to quantify the dataset. The dataset has been collected from a github repository [8]. 
 
Data set description: The dataset has been generated combining part of the Sentiment140 [6] (8,000 positive tweets), and another one for depressive tweets [7] (2,314 tweets), with a total of 10,314 tweets. The important features used are: Tweet text and label. The final dataset has three attributes tweet-id, tweet content and label that is 0 or 1 depending on whether the tweet is positive or negative respectively. 
 
We have then preprocessed this data and applied naïve bayes classifier. Data preprocessing has been done for stemming and removing stop words and then tokenizing the tweets. For the purpose of stemming porter stemmer has been used. Bag of words and tf-idf i.e. Term Frequency and Inverse Document Frequency have been used for conversion of words into vectors. 
 
We have then used Bayes theorem to predict if a person is diagnosed with depression or not (Fig-6.5) and then measured the results through F-score, precision, recall and accuracy. (Fig-6.3 and 6.4) 
 
 
 
 Approach for Depression Detection through Speech: The project starts with collection of audio data audio data [9] (made available from DAIC-WOZ (Distress Analysis Interview Corpus-Wizard-of-Oz database).  
 
  
Data set description: All audio recordings and associated depression metrics were provided by the DAICWOZ Database [9], which was compiled by USC's Institute of Creative Technologies and released as part of the 2016 Audio/Visual Emotional Challenge and Workshop (AVEC 2016). The dataset consists of 189 sessions, averaging 16 minutes, between a participant and virtual interviewer called Ellie, controlled by a human interviewer in another room via a "Wizard of Oz" approach. Prior to the interview, each participant completed a psychiatric questionnaire (PHQ-8)[5], from which a binary "truth" classification (depressed, not depressed) was derived. 
 
The audio file used has 74 columns consisting of each of the features below as shown by labelling of A, B, C………….BW in the csv file. A- F0, B- NAQ, C- QOQ, D- H1H2, E- PSP, F- MDQ, G - peak Slope, H- Rd, I-Rd conf, J – AH -MCEP 0-24, AI – BF - HMPDM 1-24, BG – BR -HMPDD 1-12, BS – BV - Formants 1-3. Pre-processing of the data is done by removing unnecessary rows and adding the PHQ-8 score column.  Further, the large data is handled for efficient processing by diving it into 11 separate folders and feeding each of them to the model for training and taking the average of the results for testing purpose.  Also, data has imbalance having number of non-depressed participants four times larger than the depressed ones. It is addressed using under sampling. The correlation between different variables is also analysed.  We trained four regression models namely Support Vector Regressor with kernel linear, Support Vector Regressor with kernel sigmoid, Random Forest Regressor with 40 estimators and Random Forest Regressor with 400 estimators. We compared the models on the basis of root mean square error (rmse)and mean absolute error (mae) when calculating PHQ-8 score of a patient. Support Vector Regressor(kernel sigmoid) performed better than Random Forest Regressor(estimators=40 and estimators=400), former had rmse of 5.394 and latter of 6.233 (Fig-6.6). The Support Vector Regression Model applied helps in predicting the PHQ-8 score value of the tested participants. For calculating accuracy, the binary scaling is added marking „1‟ for depressed and „0‟ for non- depressed. The accuracy for all the models was similar, so on the basis of rmse and mae score we chose Support Vector Machine with kernel sigmoid for our final implementation. 

Dataset Source: [6] Go, A., Bhayani, R. and Huang, L., 2009. Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(2009), p.12. 
 
Online Source: [7] Cody Zacharias, TWINT Project,.[Online].Available: https://github.com/twintproject 
 
[8] Virdiana Romero, May 13 2019, Detecting Depression in tweets 
 
Audio Dataset :               
[9] Jonathan Gratch, Ron Artstein, Gale Lucas, Giota Stratou, Stefan Scherer, Angela Nazarian,               Rachel Wood, Jill Boberg, David DeVault, Stacy Marsella, David Traum, Skip Rizzo, Louis               Philippe Morency, ―The Distress Analysis Interview Corpus of human and computer interviews,               in Proceedings of Language Resources and Evaluation Conference (LREC),               2014.[Online].Available:http://dcapswoz.ict.usc.edu/ 
