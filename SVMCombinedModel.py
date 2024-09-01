import time
import random
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from Classifiers import SVMClassifier, TFIDFVectorizer, TFVectorizer
from DataLoader import DataLoader
from ReportGenerators import ClassificationReportWriter
from inputGathering import SVMDualToSingleInputGatherer

'''
SVM dual to single class model
'''

def main():
    #input for model
    (
        language_choice, 
        ngram_start_dual, ngram_end_dual, kernel_dual, C_dual, gamma_dual,
        ngram_start_single, ngram_end_single, kernel_single, C_single, gamma_single,
        want_compression, compression_type, want_random_time, want_tf, want_tf2, want_report
    ) = SVMDualToSingleInputGatherer()

    compression = compression_type

    random_mtime = get_random_time() if want_compression and want_random_time else 0

    #get data loader
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
    train_nouns_dual, train_classes_dual = load_data(data_loader_dual, want_compression, is_train=True, compression=compression)
    test_nouns_dual, test_classes_dual = load_data(data_loader_dual, want_compression, is_train=False, compression=compression)
    #create the vecotrizer
    vectorizer_dual = create_vectorizer(want_tf, ngram_start_dual, ngram_end_dual)
    #create the classifer
    svm_classifier_dual = SVMClassifier(train_nouns_dual, train_classes_dual, test_nouns_dual, test_classes_dual)
    results, best, y_pred_dual, vectorizer_final, kernel_final, C_final, gamma_final = svm_classifier_dual.test_multiple_settings_return_best(
        [vectorizer_dual], [kernel_dual], [C_dual], [gamma_dual]
    )


    # --- Single Model ---
    #load the data
    train_nouns_single, train_classes_single = load_data(data_loader_single, want_compression, is_train=True, compression=compression)
    test_nouns_single, test_classes_single = load_data(data_loader_single, want_compression, is_train=False, compression=compression)
    #create the vecorizer
    vectorizer_single = create_vectorizer(want_tf2, ngram_start_single, ngram_end_single)

    X_train = vectorizer_single.fit_transform(train_nouns_single)
    y_train = train_classes_single

    X_test = vectorizer_single.transform(test_nouns_single)
    y_test = test_classes_single

    #create the classifer
    svm = SVC(kernel=kernel_single, C=C_single, gamma=gamma_single, probability=True)
    svm.fit(X_train, y_train)

    #process the predictions dual to single
    final_y_pred = process_predictions(y_pred_dual, X_test, svm)


    accuracy = accuracy_score(y_test, final_y_pred)
    print(f"Final Accuracy: {accuracy}")

    #if you want the report generate it
    if want_report:
        write_report(want_tf, ngram_start_dual, ngram_end_dual, kernel_dual, C_dual, gamma_dual,
                     ngram_start_single, ngram_end_single, kernel_single, C_single, gamma_single,
                     final_y_pred, test_classes_single, test_nouns_single, accuracy, want_compression)
        
#function get the random m_time for compression
def get_random_time():
    current_time = time.time()
    random_offset = random.randrange(0, int(current_time))
    return current_time - random_offset

#load the data funciton
def load_data(data_loader, want_compression, is_train=True, compression='gzip'):
    if want_compression:
        return data_loader.load_train_data_compressed(compression) if is_train else data_loader.load_test_data_compressed(compression)
    else:
        return data_loader.load_train_data() if is_train else data_loader.load_test_data()

#function create the vectorizer
def create_vectorizer(want_tf, ngram_start, ngram_end):
    return TFVectorizer(ngram_range=(ngram_start, ngram_end)) if want_tf else TFIDFVectorizer(ngram_range=(ngram_start, ngram_end))

#function process predictions dual to single
def process_predictions(y_pred, X_test, svm):
    final_y_pred = []
    for pred, noun in zip(y_pred, X_test):
        probs = svm.predict_proba(noun)
        classes = list(svm.classes_)
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

#function write report and assign correct filename
def write_report(want_tf, ngram_start_dual, ngram_end_dual, kernel_dual, C_dual, gamma_dual,
                 ngram_start_single, ngram_end_single, kernel_single, C_single, gamma_single,
                 final_y_pred, test_classes, test_nouns, accuracy, want_compression):
    vectorizer_param = "TF" if want_tf else "TFIDF"
    filename = f'SVMDualToSingle{"Compressed" if want_compression else ""}{vectorizer_param}ng-{ngram_start_dual}-{ngram_end_dual}K-{kernel_dual}C-{C_dual}G-{gamma_dual}ng-{ngram_start_single}-{ngram_end_single}K-{kernel_single}C-{C_single}G-{gamma_single}.xlsx'
    
    if want_compression:
        test_nouns = DataLoader.decompress_array(test_nouns)
    
    report_writer = ClassificationReportWriter(final_y_pred, test_classes, test_nouns, accuracy, filename)
    report_writer.write_report()

if __name__ == "__main__":
    main()