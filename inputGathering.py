import random

'''
Functions for gathering input to be used in testing
'''

#function get float list input
def get_float_list_input(prompt):
    while True:
        try:
            values = input(prompt).split()
            return [float(x) for x in values] if values else []
        except ValueError:
            print("Invalid input. Please enter floats separated by spaces.")

#function get a valid input
def get_validated_input(prompt, input_type, valid_options=None):
    while True:
        try:
            user_input = input_type(input(prompt))
            if valid_options and user_input not in valid_options:
                raise ValueError
            return user_input
        except ValueError:
            print("Invalid input. Please try again.")

#function get yes or no input
def get_yes_no_input(prompt):
    return get_validated_input(f"{prompt} (y/n): ", str.lower, ['y', 'n']) == 'y'

#function get list of int
def get_int_list_input(prompt):
    while True:
        try:
            values = input(prompt).split()
            return [int(x) for x in values] if values else []
        except ValueError:
            print("Invalid input. Please enter integers separated by spaces.")

#Function to get KNN INPUT
def KNNInputGatherer():
    class_type = get_validated_input("Select Class format (1: Dual Class, 2: Single Class): ", int, [1, 2])
    
    language_choice = get_validated_input("Select language (1: northern sotho, 2: isizulu): ", int, [1, 2])
    single_run = get_yes_no_input("Do you want a single run?")
    
    ngram_start = get_validated_input("Enter your n-gram range start: ", int)
    ngram_end = get_validated_input("Enter your n-gram range end: ", int)
    
    nearest_neighbours = get_int_list_input("Enter your list of NN values, split by spaces: ")
    
    want_tf = get_yes_no_input("Do you want to use TF?")
    want_report = get_yes_no_input("Do you want an in-depth report?")
    
    want_compression = get_yes_no_input("Do you want to use compression?")
    
    if want_compression:

        compression_type = get_validated_input("Select compression (1: gzip, 2: zlib): ", int, [1,2])

        if compression_type == 1:
            compression_type = 'gzip'
            want_random_time = get_yes_no_input("Do you want random compression?")
        else:
            compression_type = 'zlib'
            want_random_time = False
    
    else:
        compression_type = None
        want_random_time = False

    return (
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
    )

#Function to get Decision tree INPUT
def DecisionTreeInputGatherer():
    class_type = get_validated_input("Select Class format (1: Dual Class, 2: Single Class): ", int, [1, 2])
    
    language_choice = get_validated_input("Select language (1: northern sotho, 2: isizulu): ", int, [1, 2])
    single_run = get_yes_no_input("Do you want a single run?")
    
    ngram_start = get_validated_input("Enter your n-gram range start: ", int)
    ngram_end = get_validated_input("Enter your n-gram range end: ", int)
    
    max_depth_values = get_int_list_input("Enter your list of max-value depths, split by space (can be empty): ")
    
    want_random_DT = get_yes_no_input("Do you want a random Decision Tree?")
    random_DT_seed = random.randint(0, 42) if want_random_DT else 42
    
    want_tf = get_yes_no_input("Do you want to use TF?")
    want_prune = get_yes_no_input("Do you want to prune the tree?")
    want_visualization = get_yes_no_input("Do you want a tree visualization?")
    want_report = get_yes_no_input("Do you want an in-depth report?")
    
    want_compression = get_yes_no_input("Do you want to use compression?")
    
    if want_compression:
        compression_type = get_validated_input("Select compression (1: gzip, 2: zlib): ", int, [1,2])

        if compression_type == 1:
            compression_type = 'gzip'
            want_random_time = get_yes_no_input("Do you want random compression?")
        else:
            compression_type = 'zlib'
            want_random_time = False
    
    else:
        compression_type = None
        want_random_time = False

    return (
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
    )

#Function to get Dual to single class INPUT
def DualToSingleInputGatherer():
    language_choice = get_validated_input("Select language (1: northern sotho, 2: isizulu): ", int, [1, 2])

    ngram_start_dual = get_validated_input("Enter your n-gram range start for dual model: ", int)
    ngram_end_dual = get_validated_input("Enter your n-gram range end for dual model: ", int)
    nearest_neighbours_dual = get_validated_input("Enter your NN value for dual model: ", int)
    want_tf = get_yes_no_input("Do you want to use TF?")

    ngram_start_single = get_validated_input("Enter your n-gram range start for single model: ", int)
    ngram_end_single = get_validated_input("Enter your n-gram range end for single model: ", int)
    nearest_neighbours_single = get_validated_input("Enter your NN value for single model: ", int)
    want_tf2 = get_yes_no_input("Do you want to use TF?")

    want_compression = get_yes_no_input("Do you want to use compression?")
    
    if want_compression:
        compression_type = get_validated_input("Select compression (1: gzip, 2: zlib): ", int, [1,2])

        if compression_type == 1:
            compression_type = 'gzip'
            want_random_time = get_yes_no_input("Do you want random compression?")
        else:
            compression_type = 'zlib'
            want_random_time = False
    
    else:
        compression_type = None
        want_random_time = False

    want_report = get_yes_no_input("Do you want an in-depth report?")

    return (
        language_choice, ngram_start_dual, ngram_end_dual, nearest_neighbours_dual,
        ngram_start_single, ngram_end_single, nearest_neighbours_single,
        want_compression, compression_type, want_random_time, want_tf, want_tf2, want_report
    )

#Function to get Dual to single decision tree INPUT
def DTDualToSingleInputGatherer():
    language_choice = get_validated_input("Select language (1: northern sotho, 2: isizulu): ", int, [1, 2])

    ngram_start_dual = get_validated_input("Enter your n-gram range start for dual model: ", int)
    ngram_end_dual = get_validated_input("Enter your n-gram range end for dual model: ", int)
    max_depth_dual = get_validated_input("Enter your max depth for dual model (or press Enter for None): ", lambda x: None if x == '' else int(x))
    want_tf = get_yes_no_input("Do you want to use TF?")

    ngram_start_single = get_validated_input("Enter your n-gram range start for single model: ", int)
    ngram_end_single = get_validated_input("Enter your n-gram range end for single model: ", int)
    max_depth_single = get_validated_input("Enter your max depth for single model (or press Enter for None): ", lambda x: None if x == '' else int(x))
    want_tf2 = get_yes_no_input("Do you want to use TF?")

    want_random_DT = get_yes_no_input("Do you want random Decision Trees?")
    random_DT_seed = random.randint(0, 42) if want_random_DT else 42
    
    want_compression = get_yes_no_input("Do you want to use compression?")
    
    if want_compression:
        compression_type = get_validated_input("Select compression (1: gzip, 2: zlib): ", int, [1,2])

        if compression_type == 1:
            compression_type = 'gzip'
            want_random_time = get_yes_no_input("Do you want random compression?")
        else:
            compression_type = 'zlib'
            want_random_time = False
    
    else:
        compression_type = None
        want_random_time = False
    
    want_prune = get_yes_no_input("Do you want to prune the trees?")
    want_report = get_yes_no_input("Do you want an in-depth report?")

    return (
        language_choice, 
        ngram_start_dual, ngram_end_dual, max_depth_dual,
        ngram_start_single, ngram_end_single, max_depth_single,
        random_DT_seed, want_compression, compression_type, want_random_time, 
        want_tf,want_tf2, want_prune, want_report
    )

#Function to get simple prefix INPUT
def SimplePrefixModelInput():
    language_choice = get_validated_input("Select language (1: northern sotho, 2: isizulu): ", int, [1, 2])

    number_runs =  get_validated_input("How many runs do you want ? ", int)

    return ( language_choice, number_runs)

#Function to get simple prefix knowledge infused INPUT
def SimplePrefixKnowledgeInfused():

    language_choice = get_validated_input("Select language (1: northern sotho, 2: isizulu): ", int, [1, 2])

    return language_choice

#Function to get SVM INPUT
def SVMInputGatherer():
    class_type = get_validated_input("Select Class format (1: Dual Class, 2: Single Class): ", int, [1, 2])
    language_choice = get_validated_input("Select language (1: northern sotho, 2: isizulu): ", int, [1, 2])
    single_run = get_yes_no_input("Do you want a single run?")
    
    ngram_start = get_validated_input("Enter your n-gram range start: ", int)
    ngram_end = get_validated_input("Enter your n-gram range end: ", int)
    
    kernel_list = input("Enter your list of kernels, split by spaces (e.g., linear rbf poly sigmoid): ").split()
    C_list = get_float_list_input("Enter your list of C values, split by spaces: ")
    gamma_list = input("Enter your list of gamma values, split by spaces (or leave empty for 'scale'): ").split()
    gamma_list = gamma_list if gamma_list else ['scale']
    
    want_tf = get_yes_no_input("Do you want to use TF?")
    want_report = get_yes_no_input("Do you want an in-depth report?")
    
    want_compression = get_yes_no_input("Do you want to use compression?")
    
    if want_compression:
        compression_type = get_validated_input("Select compression (1: gzip, 2: zlib): ", int, [1,2])
        compression_type = 'gzip' if compression_type == 1 else 'zlib'
        want_random_time = get_yes_no_input("Do you want random compression?") if compression_type == 'gzip' else False
    else:
        compression_type = None
        want_random_time = False

    return (
        class_type, language_choice, single_run, ngram_start, ngram_end,
        kernel_list, C_list, gamma_list, want_tf, want_report,
        want_compression, compression_type, want_random_time
    )

#Function to get dual to single SVM INPUT
def SVMDualToSingleInputGatherer():
    language_choice = get_validated_input("Select language (1: northern sotho, 2: isizulu): ", int, [1, 2])

    ngram_start_dual = get_validated_input("Enter your n-gram range start for dual model: ", int)
    ngram_end_dual = get_validated_input("Enter your n-gram range end for dual model: ", int)
    kernel_dual = input("Enter kernel for dual model (linear, rbf, poly, sigmoid): ")
    C_dual = get_validated_input("Enter C value for dual model: ", float)
    gamma_dual = input("Enter gamma value for dual model (scale, auto, or float): ")
    want_tf = get_yes_no_input("Do you want to use TF?")

    
    ngram_start_single = get_validated_input("Enter your n-gram range start for single model: ", int)
    ngram_end_single = get_validated_input("Enter your n-gram range end for single model: ", int)
    kernel_single = input("Enter kernel for single model (linear, rbf, poly, sigmoid): ")
    C_single = get_validated_input("Enter C value for single model: ", float)
    gamma_single = input("Enter gamma value for single model (scale, auto, or float): ")
    want_tf2 = get_yes_no_input("Do you want to use TF?")


    want_compression = get_yes_no_input("Do you want to use compression?")
    
    if want_compression:
        compression_type = get_validated_input("Select compression (1: gzip, 2: zlib): ", int, [1,2])

        if compression_type == 1:
            compression_type = 'gzip'
            want_random_time = get_yes_no_input("Do you want random compression?")
        else:
            compression_type = 'zlib'
            want_random_time = False
    
    else:
        compression_type = None
        want_random_time = False

    
    want_report = get_yes_no_input("Do you want an in-depth report?")

    return (
        language_choice, 
        ngram_start_dual, ngram_end_dual, kernel_dual, C_dual, gamma_dual,
        ngram_start_single, ngram_end_single, kernel_single, C_single, gamma_single,
        want_compression, compression_type, want_random_time, want_tf, want_tf2, want_report
    )