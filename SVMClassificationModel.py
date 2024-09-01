from inputGathering import SVMInputGatherer
import random
from Classifiers import SVMClassifier, TFIDFVectorizer, TFVectorizer
from DataLoader import DataLoader
from ReportGenerators import ClassificationReportWriter
import pandas as pd
import re
import time

'''
SVM classifcation model
'''

def main():
    #gather inputs for model
    (
        class_type, language_choice, single_run, ngram_start, ngram_end,
        kernel_list, C_list, gamma_list, want_tf, want_report,
        want_compression, compression_type, want_random_time
    ) = SVMInputGatherer()

    random_mtime = get_random_time() if want_compression and want_random_time else (0 if want_compression else None)

    #get dataloader
    data_loader = get_data_loader(class_type, language_choice, random_mtime)
    
    #load the data
    train_nouns, train_classes, test_nouns, test_classes = load_data(data_loader, want_compression, compression_type)
    
    #create the vecotorizers
    vectorizers = create_vectorizers(want_tf, single_run, ngram_start, ngram_end)
    
    #create the classifer
    svm_classifier = SVMClassifier(train_nouns, train_classes, test_nouns, test_classes)

    #test and return best result
    results, best, y_pred, vectorizer_final, kernel_final, C_final, gamma_final = svm_classifier.test_multiple_settings_return_best(
        vectorizers, kernel_list, C_list, gamma_list
    )

    #output results
    print_results(results)

    #if a report is required
    if want_report:
        write_report(class_type == 1, want_tf, vectorizer_final, kernel_final, C_final, gamma_final, y_pred, test_classes, test_nouns, best, want_compression, compression_type)

#Print the results in a formated manner
def print_results(results):
    dict = {
        "ngram-range": [],
        "kernel": [],
        "C": [],
        "gamma": [],
        "accuracy": []
    }
    for vectorizer, kernel, C, gamma, accuracy in results:
        dict["ngram-range"].append(re.sub(r'\s+', '', str(vectorizer.vectorizer.ngram_range)))
        dict["kernel"].append(kernel)
        dict["C"].append(C)
        dict["gamma"].append(gamma)
        dict["accuracy"].append("{:.4f}".format(accuracy))
    df = pd.DataFrame(dict)
    print(df.to_string(index=False))

#get a random compression time if required
def get_random_time():
    current_time = time.time()
    random_offset = random.randrange(0, int(current_time))
    return current_time - random_offset

#get the data loader 
def get_data_loader(class_type,language_choice, random_mtime):
    if language_choice == 1:
        if class_type == 1:
            return DataLoader("data/80_20_Split_DualNorthSotho/NorthernSothoDual80%TrainSet.xlsx",
                            'data/80_20_Split_DualNorthSotho/NorthernSothoDual20%TestSet.xlsx',
                            mtime=random_mtime)
        elif class_type == 2:
            return DataLoader('data/80_20_Split_SingleNorthSotho/NorthernSothoSingle80%TrainSet.xlsx', 
                            'data/80_20_Split_SingleNorthSotho/NorthernSothoSingle20%TestSet.xlsx', 
                            mtime=random_mtime)
    elif language_choice == 2:
        if class_type == 1:
            return DataLoader('data/80_20_Split_Dual_Representative/ZuluNounsDual80%TrainSet.xlsx', 
                            'data/80_20_Split_Dual_Representative/ZuluNounsDual20%TestSet.xlsx', 
                            mtime=random_mtime)
        elif class_type == 2:
            return DataLoader('data/80_20_Split_Single/ZuluNouns80%TrainSet.xlsx', 
                            'data/80_20_Split_Single/ZuluNouns20%TestSet.xlsx', 
                            mtime=random_mtime)

#function to load the data
def load_data(data_loader, want_compression, compression):
    if want_compression:
        train_nouns, train_classes = data_loader.load_train_data_compressed(compression=compression)
        test_nouns, test_classes = data_loader.load_test_data_compressed(compression=compression)
    else:
        train_nouns, train_classes = data_loader.load_train_data()
        test_nouns, test_classes = data_loader.load_test_data()
    return train_nouns, train_classes, test_nouns, test_classes

#function to create the vectorizer
def create_vectorizers(want_tf, single_run, ngram_start, ngram_end):
    VectorizerClass = TFVectorizer if want_tf else TFIDFVectorizer
    if single_run:
        return [VectorizerClass(ngram_range=(ngram_start, ngram_end))]
    else:
        return [VectorizerClass(ngram_range=(ngram_start, x)) for x in range(ngram_start, ngram_end+1)]

#function to write the report
def write_report(class_type, want_tf, vectorizer_final, kernel_final, C_final, gamma_final, y_pred, test_classes, test_nouns, best, want_compression, compression_type):
    vectorizer_param = "TF" if want_tf else "TFIDF"
    filename = f'{"dualClass" if class_type else "singleClass"}{"Compressed" if want_compression else ""}{vectorizer_param}ngram-{vectorizer_final.range[0]}-{vectorizer_final.range[1]}SVM-{kernel_final}-C{C_final}-gamma{gamma_final}.xlsx'
    
    if want_compression:
        test_nouns = DataLoader.decompress_array(test_nouns, compression_type)
    
    report_writer = ClassificationReportWriter(y_pred, test_classes, test_nouns, best, filename)
    report_writer.write_report()

if __name__ == "__main__":
    main()