

# 📰 **NEWS DETECTIONS FOR FAKE AND CLUSTERING OF TYPES OF NEWS**

### 📊 **Machine Learning Applications**  
### 🧠 **Course 2024 -2025**














Carolina López De La Madriz 100475216
Álvaro Martín Ruiz   
Emma Rodriguez Hervas 
Jaime Salafranca Pardo 100475216


### Table of Contents

1. Introduction	
2. Data and preprocessing	
2.1.  Problem statement	
2.2.  Where does the dataset come from	
2.3.  Preprocessing	
2.4.  Exploratory Data Analysis	
3. Natural Language Processing	
4. Machine Learning	
4.1. Classification task: Fake News Detection	
4.2. Clustering: Types of News	
4.3. Recommendation Systems: Real News Recommendations after Fake News	
5. Dashboard	
6. Conclusions	
























### Introduction

The rapid advancement of Natural Language Processing (NLP) and text generation technologies has contributed to the growing prevalence of fake news. With the help of generative multimedia, such content has become increasingly convincing, making it difficult to distinguish real from fake information. This project explores whether the authenticity of news articles can be determined using only their textual content.
The project uses a dataset containing news articles and their corresponding labels (REAL or FAKE) and is structured around three main objectives. The first is to develop a classification model capable of predicting the veracity of a given news article. The second involves identifying the genre or topic of each article, offering further context and insights into news patterns. The final task is to implement a recommendation of news that, upon detecting a fake article, suggests a related real article to provide users with accurate alternatives.
These features are integrated into a dashboard interface, allowing users to input news content, verify its authenticity, explore its genre, and access reliable related information in case of misinformation.
Is it necessary to have the whole text or context of the news to predict the veracity of the news? Can we just predict the veracity with the length of the news or the length of the title? Is the title enough to predict the veracity of the news, or the type of news? The answers to those questions are the principal target of the work and not only the prediction itself. It is known that having a powerful model that can predict the veracity of the news is a great tool, and it is achievable, nevertheless, there are several computational and energetic issues with large models, which can consume a lot of resources. It is for that reason that checking how different methods work is an interesting approach, as reducing the amount of data, parameters, and weights is very relevant for the good implementation in large-scale models.


#### Data and preprocessing
This part of the report explains how the data has been obtained and preprocessed to achieve a clean dataset useful for the different tasks. The observation and exploratory analysis held, and the decisions taken for the data preprocessing, are also explained in this section.

#### 2.2.  Where does the dataset come from

The extraction of news is a critical point in the analysis of fake and real news. This and difficulties faced will be explained, as well as how the news can be obtained for more precise and future work. In an ideal case, the dataset could be created from recent news and applied to a certain sector. There are several API’s that work very well to obtain news with title and text. 
	For the scope of this project, we have used the GoogleNews API and an APINews for extracting some news. The problem with these APIs is the amount of news that can be scraped per day and person. In the free version, this amount is limited to less than a thousand. As there is no financial support to construct the dataset, we have used the code to understand how the data would be scraped, but we have used a dataset available on Kaggle, which replicates the results we would have. To build the dataset, we have used a FAKE news dataset with source publication date, title, and text. A similar dataset with REAL news was merged, creating a binary target value.

#### 2.3.  Preprocessing and Exploratory Data Analysis

In the created dataset, we have a target variable as well as a title and a text variable for approximately 4000 news articles for Real and 4000 for Fake news. But those variables hide some other ones as the length of the news and the number of word variables that we are going to create. Once created, those variables will be used in preprocessing to see if it is interesting to work with.

We can observe in Figure 1 and Figure 2, the count of words per article and the length of articles. As the distribution of the fake news and real news seems different, we can draw the hypothesis that those variables are very interesting, and that we could predict the veracity due to this. We will investigate it and compare models that also use the content of the news, and not only the metadata. 


![Figure1&2](img/Figure1&2)


Figure 1: Distribution of Length		   Figure 2: Distribution of word count


The rest of the preprocessing of the data consists of only checking if all the news is completed, that is, not having NANs in the text and of title. In this case, as we have decided not to spend money and time scraping the data, our imported data does not have NAN values in any case and is nearly already preprocessed. It is also interesting to look simply at the most common words for both classes to get some interesting information for Natural Language Processing. It is that way we were able to discover a problem in the dataset. All the “ ‘ “ had been deleted. This is a big problem for solving contractions.

#### Natural Language Processing

Once the dataset has been preprocessed, we may start to do the Natural Language Processing, This step is critical as it will determine the accuracy of our future model, but also the computing times and costs. To do the Natural Language Processing, a pipeline will be defined, and some different techniques will be tested and contrasted to make the decision. The base skeleton of the pipeline will be inspired by the notebooks used in class and as homework. Even if all the steps all the pipeline will be executed together, they will be divided for explanation purposes in this report.

## 3.1. Step 1: Text Processing

In our text processing pipeline, we will first remove the HTML structure and convert it to text using BeautifulSoup. The raw text is then parsed with the lxml parser to remove any embedded HTML tags, preserving only the visible textual content. Then we will extract the URL. This is a critical decision because some fake news could have url to possibly fraudulent or virus-infected web pages, which could be of great help for deciding whether the news is false or true. Nevertheless, we have decided to delete them as we want the text just to be the content of it and not metadata such as links or pictures. When reading news, the links are usually not visible but inserted in words, which makes them ‘invisible’ to readers.  Moreover, using the same library, we will recreate all the lost contractions by placing “ ‘ “ where there should be one.
Then the library contractions are used to remove and fix the contraction. This is a good technique usually used to improve text processing. The NLP function, which is the en_core_web_sm from Spacy, is used, which performs tokenization, part-of-speech tagging, and lemmatization. Tokens are reduced to their base forms (lemmas), converted to lowercase, and filtered to exclude stop words, punctuation, and non-alphabetic tokens. A second lemmatization pass is applied using the WordNet Lemmatizer (wnl), which can offer improved normalization based on WordNet’s lexical database. A final filtering step removes any remaining stop words defined in a custom or standard stopword list.

Even if the Spacy function already provides the necessary preprocessing, it is interesting to redo some steps to verify that no stop words are added. This pipeline will be applied separately to the Title and the rest of the text, this will be like that for the whole project, as the content of the title and the one in the text is very different, and we do not want to merge them. Once we have the Text processing, we can start the Text Vectorization.

After this first Pipeline, we introduced several additional improvements. First, we performed named entity recognition (NER) during preprocessing using SpaCy to identify and optionally retain key named entities such as individuals, organizations, and geopolitical locations, which could enhance thematic analysis. We also applied bigram and trigram detection using Gensim’s Phrases model to capture multi-word expressions like "climate change" or "white house", which are semantically richer than their individual components. In addition, we tested language detection and filtering to remove non-English documents and ensure consistency across the corpus.


## 3.2. Step 2: Text Vectorization

For the text vectorization, we have explored several techniques, including BoW and TFIDF. Initially, we built a dictionary from the processed text using the Gensim library. During dictionary construction, we removed extremely common words that appeared in more than 70% of the documents, as these offered limited discriminative power. At the other end of the spectrum, we also filtered out rare words that occurred in fewer than five documents, which tend to introduce noise. With this refined vocabulary, we computed Bag-of-Words (BoW) vectors that represented the raw frequency of each term in each document. We also calculated TF-IDF (Term Frequency-Inverse Document Frequency) vectors, which adjust term frequencies based on how uniquely they are distributed across the corpus. These two classical representations were compared both visually and quantitatively.
In addition to these techniques, we implemented Doc2Vec, which learns fixed-length embeddings for entire documents rather than individual words. The model was trained on our corpus and compared with averaged Word2Vec and GloVe embeddings in terms of topic separation and clustering performance. Doc2Vec was particularly useful for capturing document-level semantics in a unified framework.
To further expand our analysis, we tested Non-negative Matrix Factorization (NMF) as an alternative topic modeling method, which decomposes TF-IDF vectors into interpretable topic components. We also experimented with BERTopic and Top2Vec, two recent topic modeling techniques that combine transformer-based embeddings (e.g., BERT) with clustering algorithms like HDBSCAN and dimensionality reduction via UMAP. These methods were especially promising in identifying semantically coherent topics from embedding spaces, often outperforming traditional LDA in both interpretability and coherence.
We visualized the resulting document vectors and topic assignments using t-SNE and UMAP to reduce the data to two or three dimensions. These visualizations allowed us to observe topic clusters, overlaps, and outliers, providing valuable insights into the structure of the dataset. In some cases, HDBSCAN was applied to discover dense regions of documents that naturally clustered around a common theme, further validating the quality of the vector representations.
Each vectorization strategy was assessed qualitatively (in terms of interpretability and topic coherence) and quantitatively (using coherence scores, silhouette coefficients, and classification performance). The comparison between methods allowed us to identify the most effective representation for thematic analysis in this corpus.
How are we comparing the methods? We are computing the similarity between all the documents of the corpus. What do we expect, for clustering we want to have the lowest similarity average between documents while for regression we want a large similarity for same class elements and a low one for different class elements. 
Based on that we will select the GloVe embeding for text and Doc2Vec for Titles while we will select BoW or TFIDF for a LDA analysis.



Machine Learning
Once the natural Processing is realized, our Machine Learning work and analysis can start, as previously mentioned, the idea is first to realize a classification by selecting different variables and testing several models. Then we will use clustering to see if we can create groups of news and understand this clustering. This could be interesting to understand if we can detect populist news or just topic-related news. In this case, the results we are looking for are not as topic-related as our LDA model already does well, but rather trying to detect some tendencies of speech in news.
Finally, the recommended systems has for objective to detect similarities between real and fake news and recommended in case of detecting that a new iis fake, the 2 most similar Real news, This is interesting as it not only destroys the propagation of Fake news but also directly changes the mindset of the reader and in a quick period it can introduce the real point of view in to the readers mind.

4.1. Classification task: Fake News Detection

In this classification task, our objective is to optimize the accuracy of the classification while minimizing the computational cost and time. It is, of course, interesting to say that the more interesting metrics in this case are the recall, as we want to detect all the Fake news, even if we classify as fake the True news.

Our first model and benchmark is creating a Linear regression that uses the Length of the title in words or characters, for detecting the veracity of the news. Then, a model will be created using only the title and not the text. Finally, a third model will be built using the content of the text only. Exploring the results of those three models will enable us to combine the necessary variables to create the definitive model. Moreover, several Machine learning techniques such as Random Forest, SVM, and Neural Networks will be used.


4.2. Clustering: Types of News 

This clustering task is an exploration idea in which we want to understand is we can detect some tendencies in the news, a clustering in function of the topic is not the main objective for us, but more a sentiment clustering which could be able to detect Right wing, Neutral, or Left Wing influenced news. This could also be interesting if found, as for a reader it could be interesting to have recommended news similar to the one that he is reading, but from the opposite political spectrum.

If we use an elbow technique over the number of clusters to decide how many of them to use, we obtain 5 clusters. And plotting the obtained clusters over the PCA1 and PCA2 we get the following images.




This is interesting if we compare it to the the distribution of False and TRUE news over PCA1 and PCA2.




In the second step of the analysis, we transformed the preprocessed texts into numerical vector representations to enable machine learning models to process and analyze them. We explored a wide range of vectorization strategies, starting with traditional statistical methods and progressing toward modern embedding-based models.

Using both BoW and TF-IDF representations, we trained Latent Dirichlet Allocation (LDA) models to uncover latent thematic structures within the corpus. To determine the optimal number of topics, we employed coherence scoring using the c_v metric. We tested a range of topic counts, typically from 5 to 50, and selected the number of topics that maximized coherence. While LDA is traditionally applied using BoW vectors, we also experimented with training LDA using TF-IDF representations, as recent literature suggests this may improve the semantic clarity of topics in some cases.
Beyond classical models, we also explored word embeddings using Word2Vec and GloVe, both of which generate dense vector representations for individual words. These embeddings were either pre-trained (e.g., on Google News or Common Crawl) or trained on our own corpus. To obtain document-level vectors, we tested multiple aggregation methods. The simplest was to average the word vectors of each document. However, we also applied TF-IDF weighted averaging, which places greater emphasis on informative words. In both cases, we applied Principal Component Analysis (PCA) or Singular Value Decomposition (SVD) to reduce the dimensionality of the resulting vectors and suppress noise. These steps aimed to produce document embeddings that preserved meaningful semantic structure while being computationally efficient.




Dashboard




Conclusions

