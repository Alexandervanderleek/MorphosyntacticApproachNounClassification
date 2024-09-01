from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

"""
Contains Classes for TFIDF, TF, KNN, DT(DECISION TREE), SVM
used for quickly testing multiple settings with our vectorizers and supervised learning techniques
"""


#Basic Term Frequency Inverse Frequency Vectorizer class
class TFIDFVectorizer:
    def __init__(self, ngram_range, analyzer='char', lowercase=False):
        self.range = ngram_range
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer, lowercase=lowercase)

    def fit_transform(self, data):
        return self.vectorizer.fit_transform(data)

    def transform(self, data):
        return self.vectorizer.transform(data)

#Basic Term Frequency vectorizer class
class TFVectorizer:
    def __init__(self, ngram_range, analyzer='char', lowercase=False):
        self.range = ngram_range
        self.vectorizer = CountVectorizer(ngram_range=ngram_range, analyzer=analyzer, lowercase=lowercase)

    def fit_transform(self, data):
        return self.vectorizer.fit_transform(data)

    def transform(self, data):
        return self.vectorizer.transform(data)

#KNN classification model class, training & evaluation capabilities
class KNNClassifier:
    
    #Load initial data to be used (testing and training)
    def __init__(self, train_nouns, train_classes, test_nouns, test_classes):
        self.train_nouns = train_nouns
        self.train_classes = train_classes
        self.test_nouns = test_nouns
        self.test_classes = test_classes


    def train_and_evaluate(self, vectorizer, num_neighbors):
        # Fit and transform the training data
        X_train = vectorizer.fit_transform(self.train_nouns)
        y_train = self.train_classes

        # Transform the testing data
        X_test = vectorizer.transform(self.test_nouns)
        y_test = self.test_classes

        # Train the KNN classifier
        knn = KNeighborsClassifier(n_neighbors=num_neighbors)
        knn.fit(X_train, y_train)

        # Evaluate the model
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy, y_pred

    #function for testing multiple settings different NN and vectorizer values, will return the best (results, vectorizer, nn-values ect)
    def test_multiple_settings_return_best(self, vectorizers, num_neighbors_list):
        results = []
        best_result = 0
        y_pred_final = None
        num_neighbors_final = None
        vectorizer_final = None

        for vectorizer in vectorizers:
            for num_neighbors in num_neighbors_list:
                accuracy, y_pred = self.train_and_evaluate(vectorizer, num_neighbors)
                results.append((vectorizer, num_neighbors, accuracy))
                if(accuracy>best_result):
                    best_result = accuracy
                    y_pred_final = y_pred
                    vectorizer_final = vectorizer
                    num_neighbors_final = num_neighbors

        return results, best_result, y_pred_final, vectorizer_final, num_neighbors_final

#Decision tree classification model class, training & evaluation capabilities
class DTreeClassifier:
    def __init__(self, train_nouns, train_classes, test_nouns, test_classes, random_state=None, prune_tree=False):
        self.train_nouns = train_nouns
        self.train_classes = train_classes
        self.test_nouns = test_nouns
        self.test_classes = test_classes
        self.random_state = random_state
        self.prune_tree = prune_tree

    def train_and_evaluate(self, vectorizer, max_depth):
        # Fit and transform the training data
        X_train = vectorizer.fit_transform(self.train_nouns)
        y_train = self.train_classes

        # Transform the testing data
        X_test = vectorizer.transform(self.test_nouns)
        y_test = self.test_classes

        # Train the Decision Tree classifier
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=self.random_state)
        dt.fit(X_train, y_train)

        #if we want to apply prunning
        if self.prune_tree:

            path = dt.cost_complexity_pruning_path(X_train, y_train)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities

            # Train a series of decision trees with different alpha values
            pruned_models = []
            for ccp_alpha in ccp_alphas:
                pruned_model = DecisionTreeClassifier(criterion="gini",max_depth=max_depth, ccp_alpha=ccp_alpha, random_state=self.random_state)
                pruned_model.fit(X_train, y_train)
                pruned_models.append(pruned_model)

            # Find the model with the best accuracy on test data
            best_accuracy = 0
            best_pruned_model = None
            for pruned_model in pruned_models:
                accuracy = pruned_model.score(X_test, y_test)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_pruned_model = pruned_model

            # Model Accuracy after pruning
            y_pred = best_pruned_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            return accuracy, y_pred, best_pruned_model

        # Evaluate the model
        y_pred = dt.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy, y_pred, dt
        
    #function for testing multiple settings different depth and vectorizer values, will return the best (results, vectorizer, depth ect)
    def test_multiple_settings_return_best(self, vectorizers, max_depth_list):
        results = []
        best_result = 0
        y_pred_final = None
        max_depth_final = None
        vectorizer_final = None
        dt_final = None

        for vectorizer in vectorizers:
            for max_depth in max_depth_list:
                accuracy, y_pred, dt = self.train_and_evaluate(vectorizer, max_depth)
                results.append((vectorizer, max_depth, accuracy))
                if accuracy > best_result:
                    dt_final = dt
                    best_result = accuracy
                    y_pred_final = y_pred
                    vectorizer_final = vectorizer
                    max_depth_final = max_depth

        return results, best_result, y_pred_final, vectorizer_final, max_depth_final, dt_final

#Support vector classification model class, training & evaluation capabilities
class SVMClassifier:
    def __init__(self, train_nouns, train_classes, test_nouns, test_classes):
        self.train_nouns = train_nouns
        self.train_classes = train_classes
        self.test_nouns = test_nouns
        self.test_classes = test_classes
    
    def train_and_evaluate(self, vectorizer, kernel, C, gamma='scale'):
        # Fit and transform the training data
        X_train = vectorizer.fit_transform(self.train_nouns)
        y_train = self.train_classes

        # Transform the testing data
        X_test = vectorizer.transform(self.test_nouns)
        y_test = self.test_classes

        # Train the SVM classifier
        svm = SVC(kernel=kernel, C=C, gamma=gamma)
        svm.fit(X_train, y_train)

        # Evaluate the model
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy, y_pred

    #function for testing multiple settings different gamma, C, kernels and vectorizer values, will return the best (results, vectorizer, C ect)
    def test_multiple_settings_return_best(self, vectorizers, kernel_list, C_list, gamma_list=['scale']):
        results = []
        best_result = 0
        y_pred_final = None
        vectorizer_final = None
        kernel_final = None
        C_final = None
        gamma_final = None

        for vectorizer in vectorizers:
            for kernel in kernel_list:
                for C in C_list:
                    for gamma in gamma_list:
                        accuracy, y_pred = self.train_and_evaluate(vectorizer, kernel, C, gamma)
                        results.append((vectorizer, kernel, C, gamma, accuracy))
                        if accuracy > best_result:
                            best_result = accuracy
                            y_pred_final = y_pred
                            vectorizer_final = vectorizer
                            kernel_final = kernel
                            C_final = C
                            gamma_final = gamma

        return results, best_result, y_pred_final, vectorizer_final, kernel_final, C_final, gamma_final