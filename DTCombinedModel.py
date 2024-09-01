import time
import random
from sklearn.metrics import accuracy_score
from Classifiers import DTreeClassifier, TFIDFVectorizer, TFVectorizer
from sklearn.tree import DecisionTreeClassifier

from DataLoader import DataLoader
from ReportGenerators import ClassificationReportWriter
from inputGathering import DTDualToSingleInputGatherer

'''
Decision tree classifcation model using Dual Class first then single class models
'''

def main():
    #Gather inputs for model
    (
        language_choice, 
        ngram_start_dual, ngram_end_dual, max_depth_dual,
        ngram_start_single, ngram_end_single, max_depth_single,
        random_DT_seed, want_compression, compression_type, want_random_time, 
        want_tf, want_tf2, want_prune, want_report
    ) = DTDualToSingleInputGatherer()

    compression = compression_type

    random_mtime = get_random_time() if want_compression and want_random_time else 0

    #Load the DataLoader based on language choice 
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
    #Load the Data
    train_nouns_dual, train_classes_dual = load_data(data_loader_dual, want_compression, is_train=True, compression=compression)
    test_nouns_dual, test_classes_dual = load_data(data_loader_dual, want_compression, is_train=False, compression=compression)

    #Create The Vectorizer
    vectorizer_dual = create_vectorizer(want_tf, ngram_start_dual, ngram_end_dual)

    #Create the Classifier
    dt_classifier_dual = DTreeClassifier(train_nouns_dual, train_classes_dual, test_nouns_dual, test_classes_dual, random_state=random_DT_seed, prune_tree=want_prune)
    results_dual, best_dual, y_pred_dual, vectorizer_final_dual, max_depth_final_dual, dt_dual = dt_classifier_dual.test_multiple_settings_return_best([vectorizer_dual], max_depth_list=[max_depth_dual])

    # --- Single Model ----
    #Load the Data
    train_nouns_single, train_classes_single = load_data(data_loader_single, want_compression, is_train=True,compression=compression)
    test_nouns_single, test_classes_single = load_data(data_loader_single, want_compression, is_train=False,compression=compression)

    #Create The Vectorizer
    vectorizer_single = create_vectorizer(want_tf2, ngram_start_single, ngram_end_single)

    X_train = vectorizer_single.fit_transform(train_nouns_single)
    y_train = train_classes_single

    X_test = vectorizer_single.transform(test_nouns_single)
    y_test = test_classes_single

    #Create Classifier
    dt = DecisionTreeClassifier(random_state=random_DT_seed, max_depth=max_depth_single)
    dt.fit(X_train, y_train)

    #If pruning of tree required
    if want_prune:
            path = dt.cost_complexity_pruning_path(X_train, y_train)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities

            # Train a series of decision trees with different alpha values
            pruned_models = []
            for ccp_alpha in ccp_alphas:
                pruned_model = DecisionTreeClassifier(criterion="gini", ccp_alpha=ccp_alpha, random_state=random_DT_seed)
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
            accuracy_after_pruning = best_pruned_model.score(X_test, y_test)
            print("Accuracy after pruning:", accuracy_after_pruning)

            y_pred = best_pruned_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            dt = best_pruned_model

    #Process predictions from dual to single
    final_y_pred = process_predictions(y_pred_dual,X_test,dt)

    accuracy = accuracy_score(y_test, final_y_pred)
    print(f"Final Accuracy: {accuracy}")

    #if need a report
    if want_report:
        write_report(want_tf, ngram_start_dual, ngram_end_dual, max_depth_final_dual,
                     ngram_start_single, ngram_end_single, max_depth_single,
                     final_y_pred, test_classes_single, test_nouns_single, accuracy, want_compression, want_prune)
        
#Get a random time for compression
def get_random_time():
    current_time = time.time()
    random_offset = random.randrange(0, int(current_time))
    return current_time - random_offset

#Load the data function
def load_data(data_loader, want_compression, is_train=True, compression='gzip'):
    if want_compression:
        return data_loader.load_train_data_compressed(compression) if is_train else data_loader.load_test_data_compressed(compression)
    else:
        return data_loader.load_train_data() if is_train else data_loader.load_test_data()

#create vecotrizer function
def create_vectorizer(want_tf, ngram_start, ngram_end):
    return TFVectorizer(ngram_range=(ngram_start, ngram_end)) if want_tf else TFIDFVectorizer(ngram_range=(ngram_start, ngram_end))

#process predictions from dual to single
def process_predictions(y_pred, X_test, dt):
    final_y_pred = []
    for pred, noun in zip(y_pred, X_test):
        probs = dt.predict_proba(noun)
        classes = list(dt.classes_)
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

#Write a report and assign correct filename
def write_report(want_tf, ngram_start_dual, ngram_end_dual, max_depth_dual,
                 ngram_start_single, ngram_end_single, max_depth_single,
                 final_y_pred, test_classes, test_nouns, accuracy, want_compression, want_prune):
    vectorizer_param = "TF" if want_tf else "TFIDF"
    prune_param = "Pruned" if want_prune else "Unpruned"
    filename = f'DTDualToSingle{"Compressed" if want_compression else ""}{vectorizer_param}{prune_param}ng-{ngram_start_dual}-{ngram_end_dual}md-{max_depth_dual}ng-{ngram_start_single}-{ngram_end_single}md-{max_depth_single}.xlsx'
    
    if want_compression:
        test_nouns = DataLoader.decompress_array(test_nouns)
    
    report_writer = ClassificationReportWriter(final_y_pred, test_classes, test_nouns, accuracy, filename)
    report_writer.write_report()

if __name__ == "__main__":
    main()