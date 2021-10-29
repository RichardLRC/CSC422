import os
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, X_train, Y_train):
        X_column_names = X_train.columns

        labels = []
        probas = []

        for value, count in Y_train.value_counts().iteritems():
            # count how many documents are from each topic
            labels.append(f"class={value}")
            probas.append(count / len(Y_train))

            indices = Y_train[Y_train == value].index.tolist()
            X = X_train.iloc[indices]
            total_word_count = X.size

            for x_column_name in X_column_names:
                # for each topic, count how many times a word is 
                # in documents of this topic
                x = X[x_column_name]
                x_word_count = x.size
                labels.append(f"{x_column_name}|class={value}")
                probas.append((x_word_count + 1) / (total_word_count + 1 * 6))

        # save model
        output = pd.DataFrame([probas], columns = labels)
        output.to_csv(os.path.join(ROOT_PATH, 'models', 'nb_model.csv'))

    def predict(self, X_test):
        model = pd.read_csv(os.path.join(ROOT_PATH, 'models', 'nb_model.csv')).to_dict(orient='index')[0]
        classes = [key.replace('class=', '') for key in model.keys() if key.startswith('class=')]
        output = []

        for _, row in X_test.iterrows():
            predicted_class = None
            highest_proba = -1

            for cl in classes:
                proba = model[f"class={cl}"]
                for x in row.keys().values:
                    count = row[x]
                    p = model[f"{x}|class={cl}"] ** count
                    proba *= p
                
                if proba > highest_proba:
                    highest_proba = proba
                    predicted_class = cl
                
            output.append(predicted_class)
        
        return output

    def name(self):
        return 'Naive Bayes'


class MultinomialNaiveBayes:
    def __init__(self):
        self.classifier = MultinomialNB()

    def fit(self, X_train, Y_train):
        self.classifier.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.classifier.predict(X_test)

    def name(self):
        return 'Multinomial Naive Bayes'
