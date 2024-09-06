# A Morphosyntactic Approach to Noun Classification
## Summary
Evaluating a Morphosyntactic Approach to Noun Classification for Sepedi and isiZulu.
Code from our cs honours project, investigating the use of an approximation of morphosyntax through the use of charachter n-grams in various variations to classify nouns into noun classes.

Models are to be trained on the noun data and associated class labels i.e abantu: 2, to then classify unseen nouns.
The datasets used in our paper have been redacted since it is not publically available.

Models are trained on the fly, to allow for the mass testing and continued testing of model variations.
Models created and tested included the following.
- K-nearest neighbors: KNNClassifcationModel.py, KNNCombinedModel.py   
- Support vector machine: SVMClassificationModel.py, SVMCombinedModel.py
- Decision Tree: DTClassifcationModel.py, DTCombinedModel.py

Where the normal classifcation model variant is trained on a single class noun dataset i.e abantu: 2, and the combined model varaint is trained on the singular and plural from together i.e abantu: 1/2 which then feeds said combined prediction to a model trained on a single class classifcation model to deteremine if the classification should be 1 or 2.

## How To Use 

> [!IMPORTANT]
> Before using these models, you need to add your own traning and testing .xlxs sets in the format of column 1 -> noun |  column 2 -> class.<br/>
> You can do this by adding your own noun data to the empty xlsx files located in the /data/[datatype]/data files.<br/>
> You can use a different language but please note this is setup for the testing of Sepedi and isiZulu and may cause naming issues.

Running **DTClassifcationModel.py** will result in the following prompts.

|Prompt |Input |Description |
|--------|---------------|-------------|
| Select Class format (1: Dual Class, 2: Single Class): | 2 | Train on "1" or "1/2". |
| Select language (1: northern sotho, 2: isizulu): | 2 | Language selection (can add own). |
| Do you want a single run? (y/n): | y | Test one config. If no, n-grams are iterated over from start to end. |
| Enter your n-gram range start: | 2 | Start range for n-grams. |
| Enter your n-gram range end: | 3 | End range for n-grams. |
| Enter max-value depths, split by spaces: | 15 | Depth values to test. |
| Do you want a random Decision Tree? (y/n): | n | Randomness for decision tree. |
| Do you want to use TF? (y/n): | n | If no, defaults to TF-IDF |
| Do you want to prune the tree? (y/n): | n | If yes, performs cost complexity pruning |
| Do you want a tree visualization? (y/n): | n | Creates a PDF tree visualization if yes, uses most accurate |
| Do you want an in-depth report? (y/n): | n | Generates .xlsx file with per-class performance and incorrect predictions|
| Do you want to use compression? (y/n): | n | D.t if nouns will be compressed. If yes, choose between gzip and zlib |

Results in terminal:
| N-gram Range | Max Depth | Accuracy |
|--------------|-----------|----------|
| (2,3)        | 15        | 0.7588   |

## Some additional tools

If you have an xlxs file of noun data and classes and need to create the 80% training and 20% testing sets then you can use /distributionTools/distrubuteDataset.py and distributionCheck.py to create and then ensure they are distributed correctly. These have to be edited manually and reference the datasets in /data/coredatasets
