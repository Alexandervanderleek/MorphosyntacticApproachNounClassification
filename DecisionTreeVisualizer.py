import graphviz
from sklearn.tree import export_graphviz

'''
Functions for visualizing the decision trees generated
'''

#Simple method to create a visualization for a decision tree
def visualizeTree(dt,vectorizer,classNames,filled=True,rounded=True,tree_name="decision_tree"):
        dot_data = export_graphviz(dt, out_file=None,
                                feature_names=vectorizer.vectorizer.get_feature_names_out(),
                                class_names=classNames,
                                filled=filled,
                                rounded=rounded)
        graph = graphviz.Source(dot_data)
        graph.render(tree_name)

#Simple method to create a visualization for a compressed decision tree
def visualizeTreeCompressed(dt,vectorizer,classNames,filled=True,rounded=True,tree_name="decision_tree"):       
        features = list(vectorizer.vectorizer.get_feature_names_out())
 
        for i in range(len(features)):
                features[i] = features[i].replace('"',"")

        dot_data = export_graphviz(dt, out_file=None,
                                  feature_names=features,
                                  class_names=classNames,
                                  filled=filled,
                                  rounded=rounded)
        graph = graphviz.Source(dot_data)
        graph.render(tree_name)

