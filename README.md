# Studying Party Differences in Opinion Writing on the Supreme Court
- Alyssa Eisenberg and Seung Ham
- August, 2018

In this project, we try to find political party difference in opinion on the Supreme Court. We used https://www.courtlistener.com/api/bulk-data to download ~22,000 opinion texts from 1792 to 2018 which have political party of author at the time of writing available.

## Getting Data:
1. In **data** folder, run ``. data_download.sh`` which will download all the necessary data.
2. Run ``python data_process.py`` which will create **data.p** file containing both party information and tokenized opinion texts.
3. In EDA.ipynb, there is a line of code saving vocabulary class to **vocab.p** file which can be used later for modeling (i.e. Baseline.ipynb).

## Preprocessing Data
### Sentence Tokenization
Most of text data was available in HTML format, so we first converted it into normal text using lxml.html package. After that, we used NLTK sentence tokenizer to separate each opinion document into sentences. Since there were many sentences with numberings and titles in these legal documents, we used NLTK tagger to filter only those sentences with both noun/pronoun and verb included.

### Word Tokenization
Once we get the set of sentences to use, we tokenized them into words based on the vocabulary used in Google pre-trained word2vec (https://code.google.com/archive/p/word2vec/). As part of this process, we removed all punctuations from sentences and labeled unknown words as '<unk>'.
  
## Baseline Model
For the baseline model, we used logistic regression with TF-IDF matrix as features. For TF-IDF, we used top 1000 words as stop words. The model results were surprisingly good, and we didn't have to tune many hyperparameters for the model. In terms of cross-validation, we used 80/10/10 split for train/dev/test data.

## Neural Net Model
To see if we can improve the model performance, we followed the methods described in the paper by Tang, Qin, and Liu (http://aclweb.org/anthology/D15-1167), and the model has the below structure:
        
1. Word vectors using word2vec
2. Different filters for CNN with different widths.
   For each filter, there are:
    - Linear layer
    - Average pooling
    - tanh transformation
3. Average of outputs from three filters to find the sentence representation.
4. Input the CNN sentence representations to the bi-directional gated RNN.
5. Outputs from forward RNN and backward RNN are concatenated for each sentence,
   and the concatenated vectors are averaged to give the document representation.
6. Input the document representation to softmax layer.



## Final Report:
<a href="https://www.overleaf.com/18030683ttmhjwjsvysx#/68288176/" target="_blank">Click here</a>

## Proposal:

In this more partisan age, the judicial branch still depends on their reputation for objectivity. However, we know that Supreme Court justices appointments are politically motivated and that Supreme Court opinions have become more semantically distinctive from the broader judicial body of work of the federal appellate courts1. We want to extend this by examining differences in Supreme Court opinions by party affiliation of the authoring justice. We will identify key differences in word usage and examine changes over time in a model predicting party classification.

Our data set consists of the opinion texts from Supreme Court cases from 1951 - 2007, matched to their author’s party affiliation2, and has been used in prior research1. There is additional metadata about each Supreme Court case if needed through the SCDB3.

First, we will use a simple logistic regression on tfidfVectorization of the opinions to classify the author’s party. This is easily interpretable and we can identify words with the largest weights for each class, exemplifying key differences in language usage.

Additionally, we will build a more complex classification model based on learned document representation. We will follow Tang, Qin, and Liu4 to use an LSTM to get sentence representations, and combine those to the document level using a gated RNN.

We will evaluate our hypothesis that partisanship has been increasing over time. We anticipate our model accuracy will increase over time, indicating that opinions from different parties will become increasingly distinctive and easier to predict. We will also compare document similarity measures (cosine or Jaccard distance) on the document representations for the parties over time, anticipating that each party will become closer internally and further apart from the other party.

If we have time, we will run the same analysis using a different document representation strategy from Le and Mikolov5. We can also explore the usage of different types of word embeddings such as learning a ‘party affiliation’ vector with the word embeddings6.

## References:
1. Livermore, Michael A.; Riddell, Allen; Rockmore, Daniel. “The Supreme Court and the Judicial Genre.” Arizona Law Review 59 (2017): 837 - 901. Retrieved June 17 2018 from  https://ssrn.com/abstract=2740126.
2. CourtListener from Free Law Project. https://www.courtlistener.com/api/bulk-info/
3. The Supreme Court Database from Washington University Law. http://supremecourtdatabase.org/
4. Tang, Duyu; Qin, Bing; Liu, Ting. “Document Modeling with Gated Recurrent Neural Network for Sentiment Classification.” Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (2015): 1422-1432. Retrieved June 17 2018 from http://aclweb.org/anthology/D15-1167
5. Le, Quoc; Mikolov, Tomas. “Distributed Representations of Sentences and Documents.” Proceedings of the 31st International Conference on Machine Learning (2014): 1188-1196. Retrieved June 17 2018 from https://arxiv.org/pdf/1405.4053.pdf
6. Nay, John J. “Gov2Vec: Learning Distributed Representations of Institutions and Their Legal Text.” Proceedings of 2016 EMNLP Workshop on Natural Language Processing and Computational Social Science (2016): 49-54. Retrieved June 17 2018 from http://www.aclweb.org/anthology/W16-5607
