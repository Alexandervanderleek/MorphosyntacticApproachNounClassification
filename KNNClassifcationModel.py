from inputGathering import KNNInputGatherer
import random
from Classifiers import KNNClassifier, TFIDFVectorizer, TFVectorizer
from DataLoader import DataLoader
from ReportGenerators import ClassificationReportWriter
import pandas as pd
import re
import time

'''
KNN classifcation model
'''

def main():

    #Input gathering for model
    (
        class_type,
        language_choice,
        single_run,
        ngram_start,
        ngram_end,
        nearest_neighbours,
        want_tf,
        want_report,
        want_compression,
        compression_type,
        want_random_time
    ) = KNNInputGatherer()

    compression = compression_type

    random_mtime=None

    if want_compression and want_random_time:
        random_mtime = get_random_time()
    
    if want_compression and not want_random_time:
        random_mtime=0

    # Create an instance of the DataLoader class
    data_loader = get_data_loader(class_type,language_choice, random_mtime)

    # Load the training and testing data
    train_nouns, train_classes, test_nouns, test_classes = load_data(data_loader, want_compression, compression=compression)

    # Create instances of the TextVectorizer
    vectorizers = create_vectorizers(want_tf, single_run, ngram_start, ngram_end)

    # Create an instance of the KNNClassifier
    knn_classifier = KNNClassifier(train_nouns, train_classes, test_nouns, test_classes)

    # Test multiple settings for vectorizers and num_neighbors
    results, best, y_pred, vectorizer_final, num_neighbors_final = knn_classifier.test_multiple_settings_return_best(vectorizers, nearest_neighbours)

    # Print results
    print_results(results)

    #if want a report
    if want_report:
        write_report(class_type==1,want_tf, vectorizer_final, num_neighbors_final, y_pred, test_classes, test_nouns, best, want_compression, compression)

#get a random m_time for compression
def get_random_time():
    current_time = time.time()
    random_offset = random.randrange(0, int(current_time))
    return current_time - random_offset

#function to get data loader
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

#function to print results formated
def print_results(results):
    dict = {
        "ngram-range": [],
        "NN-value": [],
        "accuracy": []
    }
    for vectorizer, num_neighbors, accuracy in results:
        dict["ngram-range"].append(re.sub(r'\s+', '', str(vectorizer.vectorizer.ngram_range)))
        dict["NN-value"].append(num_neighbors)
        dict["accuracy"].append("{:.4f}".format(accuracy))
    df = pd.DataFrame(dict)
    print(df.to_string(index=False))

#function to write report and assign correct filename
def write_report(class_type,want_tf, vectorizer_final, num_neighbors_final, y_pred, test_classes, test_nouns, best, want_compression, compression):
    vectorizer_param = "TF" if want_tf else "TFIDF"
    filename = f'{"dualClass" if class_type else "singleClass"}{"Compressed" if want_compression else ""}{vectorizer_param}ngram-{vectorizer_final.range[0]}-{vectorizer_final.range[1]}NN-{num_neighbors_final}.xlsx'
    
    if want_compression:
        test_nouns = DataLoader.decompress_array(test_nouns, compression)
    
    report_writer = ClassificationReportWriter(y_pred, test_classes, test_nouns, best, filename)
    report_writer.write_report()

if __name__ == "__main__":
    main()