# Studying Party Differences in Opinion Writing on the Supreme Court
- Alyssa Eisenberg and Seung Ham
- July, 2018

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
