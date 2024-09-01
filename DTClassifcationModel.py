from inputGathering import DecisionTreeInputGatherer
import random
from Classifiers import DTreeClassifier, TFIDFVectorizer, TFVectorizer
from DataLoader import DataLoader
from ReportGenerators import ClassificationReportWriter
from DecisionTreeVisualizer import visualizeTree, visualizeTreeCompressed
import pandas as pd
import re
import time

'''
Decision tree classifcation model
'''

def main():
    
    #Gather inputs for model
    (
        class_type,
        language_choice,
        single_run,
        ngram_start,
        ngram_end,
        max_depth_values,
        random_DT_seed,
        want_tf,
        want_prune,
        want_visualization,
        want_report,
        want_compression,
        compression_type,
        want_random_time
    ) = DecisionTreeInputGatherer()

    random_mtime = None

    compression = compression_type

    if want_compression and want_random_time:
        random_mtime = get_random_time()
    elif want_compression and not want_random_time:
        random_mtime = 0

    # Create an instance of the DataLoader class
    data_loader = get_data_loader(class_type, language_choice, random_mtime)

    # Load the training and testing data
    train_nouns, train_classes, test_nouns, test_classes = load_data(data_loader, want_compression, compression=compression)

    # Create instances of the TextVectorizer
    vectorizers = create_vectorizers(want_tf, single_run, ngram_start, ngram_end)

    # Create an instance of the DecisionTreeClassifier
    decision_classifier = DTreeClassifier(train_nouns, train_classes, test_nouns, test_classes, random_state=random_DT_seed, prune_tree=want_prune)

    # Test multiple settings for vectorizers and max_depth
    max_depth_list = [None] + max_depth_values
    results, best, y_pred, vectorizer_final, max_depth_final, dt = decision_classifier.test_multiple_settings_return_best(vectorizers, max_depth_list=max_depth_list)

    # Print results
    print_results(results)

    #if you need a tree visualization
    if want_visualization:
        visualize_tree(dt, vectorizer_final, train_classes, want_compression, class_type, want_tf, max_depth_final)

    #if you want to report
    if want_report:
        write_report(class_type, want_tf, vectorizer_final, max_depth_final, y_pred, test_classes, test_nouns, best, want_compression,compression)

#generate a random m_time for compression if required
def get_random_time():
    current_time = time.time()
    random_offset = random.randrange(0, int(current_time))
    return current_time - random_offset

#Function to get correct dataloader based on input
def get_data_loader(class_type, language_choice, random_mtime):
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

#Function to load data using chosen dataloader and parameters
def load_data(data_loader, want_compression,compression):
    if want_compression:
        train_nouns, train_classes = data_loader.load_train_data_compressed(compression)
        test_nouns, test_classes = data_loader.load_test_data_compressed(compression)
    else:
        train_nouns, train_classes = data_loader.load_train_data()
        test_nouns, test_classes = data_loader.load_test_data()
    return train_nouns, train_classes, test_nouns, test_classes

#Function to generate vectorizers to be used
def create_vectorizers(want_tf, single_run, ngram_start, ngram_end):
    VectorizerClass = TFVectorizer if want_tf else TFIDFVectorizer
    if single_run:
        return [VectorizerClass(ngram_range=(ngram_start, ngram_end))]
    else:
        return [VectorizerClass(ngram_range=(ngram_start, x)) for x in range(ngram_start, ngram_end+1)]

#function to print the results in a formatted manner
def print_results(results):
    dict = {
        "ngram-range": [],
        "max_depth": [],
        "accuracy": []
    }
    for vectorizer, max_depth, accuracy in results:
        dict["ngram-range"].append(re.sub(r'\s+', '', str(vectorizer.vectorizer.ngram_range)))
        dict["max_depth"].append(max_depth)
        dict["accuracy"].append("{:.4f}".format(accuracy))
    df = pd.DataFrame(dict)
    print(df.to_string(index=False))

#Fucntion to visualze the tree and save it with correct filename
def visualize_tree(dt, vectorizer_final, train_classes, want_compression, class_type, want_tf, max_depth_final):
    vectorizer_param = "TF" if want_tf else "TFIDF"
    tree_name = f'{"dualClass" if class_type == 1 else "singleClass"}{"Compressed" if want_compression else ""}{vectorizer_param}ngram-{vectorizer_final.range[0]}-{vectorizer_final.range[1]}maxdepth-{max_depth_final}'
    
    if want_compression:
        visualizeTreeCompressed(dt=dt, vectorizer=vectorizer_final, classNames=list(set(train_classes)), tree_name=tree_name)
    else:
        visualizeTree(dt=dt, vectorizer=vectorizer_final, classNames=list(set(train_classes)), tree_name=tree_name)

#fucntion to generate the report and save it with the correct filename
def write_report(class_type, want_tf, vectorizer_final, max_depth_final, y_pred, test_classes, test_nouns, best, want_compression, compression):
    vectorizer_param = "TF" if want_tf else "TFIDF"
    filename = f'{"dualClass" if class_type == 1 else "singleClass"}{"Compressed" if want_compression else ""}{vectorizer_param}ngram-{vectorizer_final.range[0]}-{vectorizer_final.range[1]}maxdepth-{max_depth_final}.xlsx'
    
    if want_compression:
        test_nouns = DataLoader.decompress_array(test_nouns,compression)
    
    report_writer = ClassificationReportWriter(y_pred, test_classes, test_nouns, best, filename)
    report_writer.write_report()


if __name__ == "__main__":
    main()