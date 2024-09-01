import openpyxl
from collections import Counter

'''
Simple functionality to check the distrubution of a dataset
'''

def main():

    # Load the Excel file
    workbook = openpyxl.load_workbook(fr'..\data\80_20_Split_DualNorthSotho\NorthernSothoDual20%TestSet.xlsx')
    worksheet = workbook.active

    # Get all the class labels from the second column
    class_labels = [row[1] for row in worksheet.iter_rows(min_row=1, values_only=True)]

    # Count the occurrences of each class label
    class_counts = Counter(class_labels)

    # Calculate the total number of instances
    total_instances = sum(class_counts.values())

    # Print the class label, percentage occurrences, and count
    print("Class Distribution:")
    print("Class Label\tPercentage\tCount")
    for class_label, count in class_counts.items():
        percentage = (count / total_instances) * 100
        print(f"{class_label}\t{percentage:.2f}%\t\t{count}")

if __name__ == "__main__":
    main()