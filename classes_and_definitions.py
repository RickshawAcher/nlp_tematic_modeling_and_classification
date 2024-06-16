from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
from nltk.tokenize import NLTKWordTokenizer
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from pymystem3 import Mystem
import warnings
import nltk
import re
import regex
import requests

# nltk.download('stopwords')
from nltk.corpus import stopwords

class TextTransformer:
    '''
    Class to transform text data.
    Allows optional POS tagging.
    
    '''
    def __init__(self, dictionary=None, tokenizer=NLTKWordTokenizer()):
        '''
        Initializes the TextTransformer with optional dictionary of allowed lemmas and tokenizer.
        
        Args:
            dictionary (set): Set of allowed words. Only lemmas in this set will be included in the output.
            tokenizer: Tokenizer for text processing. Defaults to NLTKWordTokenizer.
            
        '''
        self.stop_words = self._update_stopwords()
        self.mystem = Mystem()
        self.mapping = self._fetch_mapping()
        self.dictionary = dictionary
        self.tokenizer = tokenizer

    def transform(self, X, tag_flag=False):
        '''
        Main function.
        Transforms the text data by tokenizing, lemmatizing, and optionally tagging.
        
        Args:
            X (str, pd.Series, or np.ndarray): Input text data.
            tag_flag (bool): Whether to include POS tags. Defaults to False.

        Returns:
            list of list of str: Transformed text data.
            
        '''
        tokenizer = self.tokenizer
        if isinstance(X, (pd.Series, np.ndarray)):
            texts = X if isinstance(X, np.ndarray) else X.values
            results = [self._process_text(text, tokenizer, tag_flag) for text in texts]
            return results
        elif isinstance(X, str):
            return np.array([self._process_text(X, tokenizer, tag_flag)])
        else:
            raise ValueError("Input must be a string, numpy array, or pandas Series.")

    def _remove_dates_and_replace_numbers(self, text):
        '''
        Removes dates and replaces numbers in the text.
        
        Args:
            text (str): Input text.

        Returns:
            str: Processed text with dates removed and numbers replaced.
            
        '''
        date_pattern = r'\b\d{1,2}[.-]\d{1,2}[.-]\d{4}\b|\b\d{4}[.-]\d{1,2}[.-]\d{1,2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}/\d{1,2}/\d{1,2}\b|\b\d{1,2}[.-]\d{1,2}\b|\b\d{1,2}[.-]\d{1,2}\b|\b\d{1,2}-\d{1,2}/\d{1,2}\b'
        time_pattern = r'\b\d{1,2}:\d{2}(:\d{2})?\b'
        number_pattern = r'\b\d+\b'

        text = re.sub(date_pattern, '', text)
        text = re.sub(time_pattern, 'время', text)
        text = re.sub(number_pattern, 'число', text)

        return text

    def _remove_punctuation(self, tokens):
        '''
        Removes punctuation from tokens.
        
        Args:
            tokens (list of str): List of tokens.

        Returns:
            list of str: List of tokens with punctuation removed.
            
        '''
        pattern = r'(?![\w\s\)])[\p{P}]|(?::\)|:-\)|:\(|:-\()'
        return [regex.sub(pattern, '', token) for token in tokens]

    def _process_text_with_tokenizer(self, text, tokenizer):
        '''
        Processes the text with the tokenizer.
        
        Args:
            text (str): Input text.
            tokenizer: Tokenizer for text processing.

        Returns:
            list of str: List of processed tokens.
            
        '''
        tokens = tokenizer.tokenize(self._remove_dates_and_replace_numbers(text))
        return [token.lower() for token in self._remove_punctuation(tokens)]

    def _update_stopwords(self):
        '''
        Updates the stopwords list with important words.
        Uses stopwords from  nltk.corpus.
        
        Returns:
            set: Set of updated stopwords.
            
        '''
        stop_words = set(stopwords.words('russian'))
        important_words = ["не", "нет", "опять", "ничего", "совсем", "никогда", "наконец", "хорошо", "лучше", "нельзя", "всегда"]
        for word in important_words:
            stop_words.discard(word)
        return stop_words

    def _lemmatize_text(self, tokens):
        '''
        Lemmatizes the tokens.
        
        Args:
            tokens (list of str): List of tokens.

        Returns:
            list of str: List of lemmatized tokens.
            
        '''
        text = ' '.join(tokens)
        lemmatized_text = self.mystem.lemmatize(text)
        if self.dictionary is not None:
            return [lemma.strip() for lemma in lemmatized_text if lemma.strip() in self.dictionary]
        return [lemma.strip() for lemma in lemmatized_text if lemma.strip()]

    def _tag_mystem_with_mapping(self, text):
        '''
        Tags the text with POS tags using Mystem.
        
        Args:
            text (str): Input text.

        Returns:
            tuple: Tuple containing lists of lemmas and tagged lemmas.
            
        '''
        processed = self.mystem.analyze(text)
        lemmas, tagged = [], []
        for w in processed:
            if "analysis" in w and w["analysis"]:
                try:
                    lemma = w["analysis"][0]["lex"].lower().strip()
                    if self.dictionary is not None and lemma not in self.dictionary:
                        continue
                    pos = w["analysis"][0]["gr"].split(',')[0].split('=')[0].strip()
                    tagged.append(lemma + '_' + self.mapping.get(pos, 'X'))
                    lemmas.append(lemma)
                except IndexError:
                    continue
        return lemmas, tagged

    def _fetch_mapping(self):
        '''
        Fetches the mapping for POS tags.
        
        Returns:
            dict: Dictionary mapping POS tags to their universal equivalents.
            
        '''
        url = 'https://raw.githubusercontent.com/akutuzov/universal-pos-tags/4653e8a9154e93fe2f417c7fdb7a357b7d6ce333/ru-rnc.map'
        r = requests.get(url)
        return dict(pair.split() for pair in r.text.split('\n') if len(pair.split()) == 2)

    def _process_text(self, text, tokenizer, tag_flag):
        '''
        Processes the text by tokenizing, removing stopwords, and lemmatizing.
        
        Args:
            text (str): Input text.
            tokenizer: Tokenizer for text processing.
            tag_flag (bool): Whether to include POS tags.

        Returns:
            list of str: List of processed tokens or tagged lemmas.
            
        '''
        tokens = self._process_text_with_tokenizer(text, tokenizer)
        tokens = [token for token in tokens if token not in self.stop_words]
        lemmas = self._lemmatize_text(tokens)
        if tag_flag:
            _, lemmas_with_tags = self._tag_mystem_with_mapping(' '.join(tokens))
            return lemmas_with_tags
        return lemmas

class SentimentClassifier:
    '''
    Support follow approaches:
    - classification over dataset got by 'bag_of_words' or 'tfidf'
    - classification over dataset got as the mean of embeddings of words in samples
    - classification over dataset got as the weighted mean (by 'tfidf') of embeddings of words in samples
    
    '''
    
    def __init__(self, X, y):
        '''
        Initialize the TextClassifier with the given data.

        Parameters:
        X : Input data.
        y : Labels.
        
        '''
        
        self.X = X
        self.y = y
        self.models = {}
        self.vectorizers = {}

    def fit_classifier(self, task_type, classifier_name, embedding_vectorizer=None):
        '''
        Fit the classifier with the given task type and classifier name.

        Parameters:
        task_type (str): Type of task ('bag_of_words', 'tfidf', 'mean_embedding_vectorizer', 'tfidf_embedding_vectorizer').
        classifier_name (str): Name of the classifier ('svm', 'logreg', 'catboost').
        embedding_vectorizer: Embedding vectorizer for 'mean_embedding_vectorizer' and 'tfidf_embedding_vectorizer'.
        
        '''
        
        
        if task_type == "bag_of_words":
            vectorizer = CountVectorizer(analyzer=analyzer)
            X_processed = vectorizer.fit_transform(self.X)
            self.vectorizers[task_type] = vectorizer
        elif task_type == "tfidf":
            vectorizer = TfidfVectorizer(analyzer=analyzer)
            X_processed = vectorizer.fit_transform(self.X)
            self.vectorizers[task_type] = vectorizer
        elif task_type == "mean_embedding_vectorizer":
            if not embedding_vectorizer:
                raise ValueError("Need embedding_vectorizer for mean_embedding_vectorizer")
            dim = len(list(embedding_vectorizer[1]))          
            X_processed = np.array([
                    np.mean([embedding_vectorizer[w] for w in words if w in embedding_vectorizer]
                    or [np.zeros(dim)], axis=0) for words in self.X])
            self.vectorizers[task_type] = embedding_vectorizer
        elif task_type == "tfidf_embedding_vectorizer":
            if not embedding_vectorizer:
                raise ValueError("Need embedding_vectorizer for tfidf_embedding_vectorizer")
            tfidf = TfidfVectorizer(analyzer=analyzer)
            tfidf.fit(self.X)
            max_idf = max(tfidf.idf_)
            tfidf_weights = defaultdict(
                lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
            dim = len(list(embedding_vectorizer[1])) 
            X_processed = np.array([
                np.mean([embedding_vectorizer[w] * tfidf_weights[w]
                         for w in words if w in embedding_vectorizer] or
                        [np.zeros(dim)], axis=0) for words in self.X])
            self.vectorizers[task_type] = {'embedding_vectorizer': embedding_vectorizer,
                                           'tfidf_weights': tfidf_weights}
        else:
            raise ValueError("Invalid task type")

        if classifier_name == "svm":
            classifier = SVC(probability=True)
        elif classifier_name == "logreg":
            classifier = LogisticRegression()
        elif classifier_name == "catboost":
            classifier = CatBoostClassifier(verbose=False)
        else:
            raise ValueError("Invalid classifier name")

        classifier.fit(X_processed, self.y)
        self.models[f"{task_type}_{classifier_name}"] = classifier

    def _process_data(self, X, task_type):
        '''
        Private method processing the input data according to the given task type.

        Parameters:
        task_type (str): Type of task ('bag_of_words', 'tfidf', 'mean_embedding_vectorizer', 'tfidf_embedding_vectorizer').
        X: Input data.

        Returns:
        np.ndarray: Processed data.
        
        '''
        if task_type == "bag_of_words" or task_type == 'tfidf':
            return self.vectorizers[task_type].transform(X)
        elif task_type == "mean_embedding_vectorizer":
            embedding_vectorizer = self.vectorizers[task_type]
            dim = len(list(embedding_vectorizer[1])) 
            return np.array([
                    np.mean([embedding_vectorizer[w] for w in words if w in embedding_vectorizer]
                    or [np.zeros(dim)], axis=0) for words in X])
        elif task_type == "tfidf_embedding_vectorizer":
            embedding_vectorizer = self.vectorizers[task_type]['embedding_vectorizer']
            tfidf_weights = self.vectorizers[task_type]['tfidf_weights']
            dim = len(list(embedding_vectorizer[1])) 
            return np.array([
                np.mean([embedding_vectorizer[w] * tfidf_weights[w] for w in words if w in embedding_vectorizer]
                        or [np.zeros(dim)], axis=0) for words in X])
        else:
            raise ValueError("Invalid task type")

    def predict_proba(self, X, task_type, classifier_name):
        '''
        Predict probabilities for the input data.

        Parameters:
        task_type (str): Type of task ('bag_of_words', 'tfidf', 'mean_embedding_vectorizer', 'tfidf_embedding_vectorizer').
        classifier_name (str): Name of the classifier ('svm', 'logreg', 'catboost').
        X: Input data.

        Returns:
        np.ndarray: Predicted probabilities.
        
        '''
        task_classifier_name = task_type + '_' + classifier_name
        if task_classifier_name not in self.models:
            raise ValueError("Model not trained yet")
        X_processed = self._process_data(X, task_type)
        return self.models[task_classifier_name].predict_proba(X_processed)[:, 1]

    def predict(self, X, task_type, classifier_name, threshold=0.5):
        '''
        Predict labels for the input data.

        Parameters:
        task_type (str): Type of task ('bag_of_words', 'tfidf', 'mean_embedding_vectorizer', 'tfidf_embedding_vectorizer').
        classifier_name (str): Name of the classifier ('svm', 'logreg', 'catboost').
        X: Input data.
        threshold (float): Threshold for predicting labels.

        Returns:
        np.ndarray: Predicted labels.
        
        '''
        task_classifier_name = task_type + '_' + classifier_name
        if task_classifier_name not in self.models:
            raise ValueError("Model not trained yet")
        X_processed = self._process_data(X, task_type)
        return (self.models[task_classifier_name].predict_proba(X_processed)[:, 1] > threshold).astype(int)

    def evaluate_model(self, X, y, task_type, classifier_name, threshold=0.5):
        '''
        Evaluate the model performance on the given data.

        Parameters:
        task_type (str): Type of task ('bag_of_words', 'tfidf', 'mean_embedding_vectorizer', 'tfidf_embedding_vectorizer').
        classifier_name (str): Name of the classifier ('svm', 'logreg', 'catboost').
        X: Input data.
        y: True labels.
        threshold (float): Threshold for predicting labels.
        
        '''
        task_classifier_name = task_type + '_' + classifier_name
        if task_classifier_name not in self.models:
            raise ValueError("Model not trained yet")
        y_pred_proba = self.predict_proba(X, task_type, classifier_name)
        y_pred = self.predict(X, task_type, classifier_name, threshold)

        print(f"ROC AUC Score: {roc_auc_score(y, y_pred_proba):.3f}")
        print(f"Precision: {precision_score(y, y_pred):.3f}")
        print(f"Recall: {recall_score(y, y_pred):.3f}")
        print(f"Accuracy: {accuracy_score(y, y_pred):.3f}")

        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        tick_marks = np.arange(len(set(y)))
        class_labels = ['Negative', 'Positive']
        plt.xticks(tick_marks, class_labels)
        plt.yticks(tick_marks, class_labels)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.show()
           
class CategoryClassifier(SentimentClassifier):
    
    def fit_classifier(self, task_type, classifier_name, embedding_vectorizer=None, sentiment = None) -> None:
        '''
        Fit the classifier with the given task type and classifier name.

        Parameters:
        task_type (str): Type of task ('bag_of_words', 'tfidf', 'mean_embedding_vectorizer', 'tfidf_embedding_vectorizer').
        classifier_name (str): Name of the classifier ('svm', 'logreg', 'catboost').
        embedding_vectorizer: Embedding vectorizer for 'mean_embedding_vectorizer' and 'tfidf_embedding_vectorizer'.
        sentiment (np.ndarray): Sentiment feature to be concatenated with the processed data.
        
        '''
        
        if task_type == "bag_of_words":
            vectorizer = CountVectorizer(analyzer=analyzer)
            X_processed = vectorizer.fit_transform(self.X)
            self.vectorizers[task_type] = vectorizer
        elif task_type == "tfidf":
            vectorizer = TfidfVectorizer(analyzer=analyzer)
            X_processed = vectorizer.fit_transform(self.X)
            self.vectorizers[task_type] = vectorizer
        elif task_type == "mean_embedding_vectorizer":
            if not embedding_vectorizer:
                raise ValueError("Need embedding_vectorizer for mean_embedding_vectorizer")
            dim = len(list(embedding_vectorizer[1]))          
            X_processed = np.array([
                    np.mean([embedding_vectorizer[w] for w in words if w in embedding_vectorizer]
                    or [np.zeros(dim)], axis=0) for words in self.X])
            self.vectorizers[task_type] = embedding_vectorizer
        elif task_type == "tfidf_embedding_vectorizer":
            if not embedding_vectorizer:
                raise ValueError("Need embedding_vectorizer for tfidf_embedding_vectorizer")
            tfidf = TfidfVectorizer(analyzer=analyzer)
            tfidf.fit(self.X)
            max_idf = max(tfidf.idf_)
            tfidf_weights = defaultdict(
                lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
            dim = len(list(embedding_vectorizer[1])) 
            X_processed = np.array([
                np.mean([embedding_vectorizer[w] * tfidf_weights[w]
                         for w in words if w in embedding_vectorizer] or
                        [np.zeros(dim)], axis=0) for words in self.X])
            self.vectorizers[task_type] = {'embedding_vectorizer': embedding_vectorizer,
                                           'tfidf_weights': tfidf_weights}
        else:
            raise ValueError("Invalid task type")

        if classifier_name == "svm":
            classifier = SVC(probability=True)
        elif classifier_name == "logreg":
            classifier = LogisticRegression()
        elif classifier_name == "catboost":
            classifier = CatBoostClassifier(verbose=False)
        else:
            raise ValueError("Invalid classifier name")
            
        if sentiment is not None:
            X_processed = np.hstack((X_processed.toarray(), sentiment[:, np.newaxis]))
            sentiment_flg = '_sentiment'
        else:
            sentiment_flg = ''

        classifier.fit(X_processed, self.y)
        
        self.models[f"{task_type}_{classifier_name}" + sentiment_flg] = classifier

    def predict_proba(self, X_test, task_type, classifier_name, sentiment=None):
        '''
        Predict class probabilities for the test data.

        Parameters:
        X_test: Test data.
        task_type (str): Type of task used in training.
        classifier_name (str): Name of the classifier used in training.
        sentiment (np.ndarray): Sentiment feature to be concatenated with the processed data.

        Returns:
        np.ndarray: Predicted class probabilities.
        
        '''
        X_processed = super()._process_data(X_test, task_type)
        if sentiment is not None:
            X_processed = np.hstack((X_processed.toarray(), sentiment[:, np.newaxis]))
            sentiment_flg = '_sentiment'
        else:
            sentiment_flg = ''
            
        return self.models[f"{task_type}_{classifier_name}" + sentiment_flg].predict_proba(X_processed)

    def predict(self, X_test, task_type, classifier_name, sentiment=None):
        '''
        Predict class labels for the test data.

        Parameters:
        X_test: Test data.
        task_type (str): Type of task used in training.
        classifier_name (str): Name of the classifier used in training.
        sentiment (np.ndarray): Sentiment feature to be concatenated with the processed data.

        Returns:
        np.ndarray: Predicted class labels.
        
        '''
        X_processed = self._process_data(X_test, task_type)
        if sentiment is not None:
            X_processed = np.hstack((X_processed.toarray(), sentiment[:, np.newaxis]))
            sentiment_flg = '_sentiment'
        else:
            sentiment_flg = ''
        
        return self.models[f"{task_type}_{classifier_name}" + sentiment_flg].predict(X_processed).astype(int)

    def evaluate_model(self, X_test, y_test, task_type, classifier_name, sentiment=None):
        '''
        Evaluate the model on the test data.

        Parameters:
        X_test: Test data.
        y_test: True labels for the test data.
        task_type (str): Type of task used in training.
        classifier_name (str): Name of the classifier used in training.
        sentiment (np.ndarray): Sentiment feature to be concatenated with the processed data.

        Prints:
        ROC AUC Score, Precision, Recall, Accuracy, and displays a confusion matrix.
        
        '''
        X_processed = self._process_data(X_test, task_type)
        if sentiment is not None:
            X_processed = np.hstack((X_processed.toarray(), sentiment[:, np.newaxis]))
            sentiment_flg = '_sentiment'
        else:
            sentiment_flg = ''
            
        y_pred_proba = self.models[f"{task_type}_{classifier_name}" + sentiment_flg].predict_proba(X_processed)
        y_pred = self.models[f"{task_type}_{classifier_name}" + sentiment_flg].predict(X_processed).astype(int)

        print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro'):.3f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.3f}")
        print(f"Recall: {recall_score(y_test, y_pred, average='macro'):.3f}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        tick_marks = np.arange(len(set(y_test)))
        class_labels = np.unique(y_test)
        plt.xticks(tick_marks, class_labels)
        plt.yticks(tick_marks, class_labels)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.show()

class ResponseDecomposition:
    '''
    Class to decompose responses into sentiment, category, and topic

    '''

    def __init__(self, sentiment_classifier, category_classifier, topic_modeling, text_transformer):
        '''
        Args:
            sentiment_classifier: Classifier for sentiment analysis.
            category_classifier: Classifier for category prediction.
            topic_modeling: Dictionary containing topic modeling data.
            text_transformer: Instance of TextTransformer.
            
        '''
        self.sentiment_classifier = sentiment_classifier
        self.category_classifier = category_classifier
        self.topic_modeling = topic_modeling
        self.dictionary = topic_modeling['credits'][0]['dictionary']
        self.text_transformer = text_transformer

    def _process_X(self, X):
        '''
        Processes the input data using the text transformer.

        Args:
            X (str or pd.Series): Input text data.

        '''
        return self.text_transformer.transform(X, False)

    def _map_category(self, predictions):
        '''
        Maps numeric category predictions to category names.

        Args:
            predictions (list or array of int): Numeric predictions of categories.

        '''
        mapping = {
            1: 'credits',
            2: 'deposits',
            3: 'creditcards',
            4: 'hypothec',
            5: 'autocredits',
            6: 'debitcards'
        }
        return np.array([mapping[prediction] for prediction in predictions])

    def _predict_topic(self, X, mapping, categories, sentiment):
        '''
        Predicts topics based on the input data, categories, and sentiment.

        Args:
            X (np.array): Processed text data.
            mapping (dict): Mapping of topic numbers to topic names.
            categories (np.array): Array of predicted categories.
            sentiment (np.array): Array of predicted sentiments.

        Returns:
            np.array: Array of predicted topics.
            
        '''
        corpus = [self.dictionary.doc2bow(text) for text in X]
        topics = []
        for i in range(len(corpus)):
            category = categories[i]
            grade = sentiment[i]
            vec = self.topic_modeling[category][grade]['lda_model'][corpus[i]][0]
            topic_number = max(vec, key=lambda x: x[1])[0]
            topic = mapping[category][grade][topic_number]
            topics.append(topic)
        return np.array(topics)

    def transform(self, X,
                  need_topic=True,
                  need_sentiment=True,
                  need_category=True,
                  task_type_sentiment='tfidf', classifier_name_sentiment='svm',
                  task_type_category='tfidf', classifier_name_category='logreg', sentiment_for_category_classifier=True,
                  threshold_for_sentiment=0.58,
                  topic_mapping=None):
        '''
        Transforms the input data into sentiment, category, and topic predictions.

        Args:
            X (text or pd.Series): Input text data.
            need_topic (bool): Whether topic prediction is needed. Defaults to True.
            need_sentiment (bool): Whether sentiment prediction is needed. Defaults to True.
            need_category (bool): Whether category prediction is needed. Defaults to True.
            task_type_sentiment (str): Task type for sentiment classifier. Defaults to 'tfidf'.
            classifier_name_sentiment (str): Classifier name for sentiment. Defaults to 'svm'.
            task_type_category (str): Task type for category classifier. Defaults to 'tfidf'.
            classifier_name_category (str): Classifier name for category. Defaults to 'logreg'.
            sentiment_for_category_classifier (bool): Whether sentiment prediction is needed for category classification. Defaults to True.
            threshold_for_sentiment (float): Threshold for sentiment prediction. Defaults to 0.58.
            topic_mapping (dict): Mapping for topics. Required if need_topic is True.

        Returns:
            pd.DataFrame: DataFrame containing the predicted sentiment, category, and topic.
            
        '''
        X_processed = self._process_X(X)
        result = {}

        if need_sentiment or (need_category and sentiment_for_category_classifier) or need_topic:
            sentiment_prediction = self.sentiment_classifier.predict(
                X_processed, task_type_sentiment, classifier_name_sentiment, threshold_for_sentiment)
            if need_sentiment:
                result['sentiment'] = sentiment_prediction

        if (need_topic or need_category) and sentiment_for_category_classifier:
            category_prediction = self._map_category(self.category_classifier.predict(
                X_processed, task_type_category, classifier_name_category, sentiment_prediction))
        elif need_category or need_topic:
            category_prediction = self._map_category(self.category_classifier.predict(
                X_processed, task_type_category, classifier_name_category))

        if need_category:
            result['category'] = category_prediction

        if need_topic:
            if topic_mapping is None:
                raise ValueError(
                    "Topic mapping cannot be None if need_topic is True.")
            topic_prediction = self._predict_topic(
                X_processed, topic_mapping, category_prediction, sentiment_prediction)
            result['topic'] = topic_prediction

        return pd.DataFrame(result)    
        
def analyzer(x):
    '''
    It's best time to smile :)
    
    '''
    return x