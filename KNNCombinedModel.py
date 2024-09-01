import os
import time
import random
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from Classifiers import KNNClassifier, TFIDFVectorizer, TFVectorizer
from DataLoader import DataLoader
from ReportGenerators import ClassificationReportWriter
from inputGathering import DualToSingleInputGatherer

'''
KNN classification model using dual to single class models
'''

def main():
    #Gather inputs for model
    (
        language_choice,ngram_start_dual, ngram_end_dual, nearest_neighbours_dual,
        ngram_start_single, ngram_end_single, nearest_neighbours_single,
        want_compression, compression_type, want_random_time, want_tf,want_tf2, want_report
    ) = DualToSingleInputGatherer()

    compression = compression_type

    random_mtime = get_random_time() if want_compression and want_random_time else 0

    #Load the correct dataloader
    if language_choice == 1:
        data_loader_dual = DataLoader("data/80_20_Split_DualNorthSotho/NorthernSothoDual80%TrainSet.xlsx", 
                                  'data/80_20_Split_DualNorthSotho/singleAsDual.xlsx', 
                                  mtime=random_mtime if want_compression else None)

        data_loader_single = DataLoader('data/80_20_Split_SingleNorthSotho/NorthernSothoSingle80%TrainSet.xlsx', 
                                    'data/80_20_Split_SingleNorthSotho/NorthernSothoSingle20%TestSet.xlsx',
                                    mtime=random_mtime if want_compression else None)
    else:
        data_loader_dual = DataLoader("data/80_20_Split_Dual_Representative/ZuluNounsDual80%TrainSet.xlsx", 
                                  'data/80_20_Split_Dual/ZuluNounsDual20%TestSet.xlsx', 
                                  mtime=random_mtime if want_compression else None)

        data_loader_single = DataLoader('data/80_20_Split_Single/ZuluNouns80%TrainSet.xlsx', 
                                    'data/80_20_Split_Single/ZuluNouns20%TestSet.xlsx',
                                    mtime=random_mtime if want_compression else None)

    # --- Dual Model ---
    #load the data
    train_nouns_dual, train_classes_dual = load_data(data_loader_dual, want_compression, is_train=True,  compression=compression)
    test_nouns_dual, test_classes_dual = load_data(data_loader_dual, want_compression, is_train=False, compression=compression)
    #create the vectorizer
    vectorizer_dual = create_vectorizer(want_tf, ngram_start_dual, ngram_end_dual)
    #create the classifer
    knn_classifier_dual = KNNClassifier(train_nouns_dual, train_classes_dual, test_nouns_dual, test_classes_dual)
    results, best, y_pred_dual, vectorizer_final, num_neighbors_final = knn_classifier_dual.test_multiple_settings_return_best([vectorizer_dual], [nearest_neighbours_dual])

    # --- Single Model ---
    #load the data
    train_nouns_single, train_classes_single = load_data(data_loader_single, want_compression, is_train=True,  compression=compression)
    test_nouns_single, test_classes_single = load_data(data_loader_single, want_compression, is_train=False,  compression=compression)
    #create the vecotorizer
    vectorizer_single = create_vectorizer(want_tf2, ngram_start_single, ngram_end_single)

    X_train = vectorizer_single.fit_transform(train_nouns_single)
    y_train = train_classes_single

    X_test = vectorizer_single.transform(test_nouns_single)
    y_test = test_classes_single

    #create the classifer
    knn = KNeighborsClassifier(n_neighbors=nearest_neighbours_single)
    knn.fit(X_train, y_train)

    #process the dual to single predictions
    final_y_pred = process_predictions(y_pred_dual, X_test, knn)

    accuracy = accuracy_score(y_test, final_y_pred)
    print(f"Final Accuracy: {accuracy}")

    #create report if required
    if want_report:
        write_report(want_tf, ngram_start_dual, ngram_end_dual, nearest_neighbours_dual,
                     ngram_start_single, ngram_end_single, nearest_neighbours_single,
                     final_y_pred, test_classes_single, test_nouns_single, accuracy, want_compression)

#function to get random mtime 
def get_random_time():
    current_time = time.time()
    random_offset = random.randrange(0, int(current_time))
    return current_time - random_offset

#function to load the data
def load_data(data_loader, want_compression, is_train=True, compression='gzip'):
    if want_compression:
        return data_loader.load_train_data_compressed(compression) if is_train else data_loader.load_test_data_compressed(compression)
    else:
        return data_loader.load_train_data() if is_train else data_loader.load_test_data()

#function to create the vecorizer
def create_vectorizer(want_tf, ngram_start, ngram_end):
    return TFVectorizer(ngram_range=(ngram_start, ngram_end)) if want_tf else TFIDFVectorizer(ngram_range=(ngram_start, ngram_end))

#function process prediction dual to single
def process_predictions(y_pred, X_test, knn):
    final_y_pred = []
    for pred, noun in zip(y_pred, X_test):
        probs = knn.predict_proba(noun)
        classes = list(knn.classes_)
        if "/" in pred:
            split = pred.split("/")
            if "-" in split:
                index = 1 if split[0] == "-" else 0
                final_y_pred.append(str(split[index]))
            else:
                index1 = classes.index(split[0])
                index2 = classes.index(split[1])
                decision = split[0] if probs[0][index1] > probs[0][index2] else split[1]
                final_y_pred.append(str(decision))
        else:
            final_y_pred.append(str(pred))
    return final_y_pred

#function to write the report
def write_report(want_tf, ngram_start_dual, ngram_end_dual, nearest_neighbours_dual,
                 ngram_start_single, ngram_end_single, nearest_neighbours_single,
                 final_y_pred, test_classes, test_nouns, accuracy, want_compression):
    vectorizer_param = "TF" if want_tf else "TFIDF"
    filename = f'KNNDualToSingle{"Compressed" if want_compression else ""}{vectorizer_param}ng-{ngram_start_dual}-{ngram_end_dual}NN-{nearest_neighbours_dual}ng-{ngram_start_single}-{ngram_end_single}NN-{nearest_neighbours_single}.xlsx'
    
    if want_compression:
        test_nouns = DataLoader.decompress_array(test_nouns)
    
    report_writer = ClassificationReportWriter(final_y_pred, test_classes, test_nouns, accuracy, filename)
    report_writer.write_report()

if __name__ == "__main__":
    main()