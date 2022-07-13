from random import shuffle
import multiprocessing
from multiprocessing import Pool
import csv
import nltk

nltk.download("punkt")

label_to_index = {}
label_to_index["Agent"] = 0
label_to_index["Device"] = 1
label_to_index["Event"] = 2
label_to_index["Place"] = 3
label_to_index["Species"] = 4
label_to_index["SportsSeason"] = 5
label_to_index["TopicalConcept"] = 6
label_to_index["UnitOfWork"] = 7
label_to_index["Work"] = 8


def transform_instance(row):
    """
    The transform_instance will be applied to each data instance in parallel using python’s multiprocessing module
    """
    cur_row = []
    print(row[0])
    print(row[1])
    print("-------------------------------------------------------------------------")
    label = "__label__" + str(label_to_index[row[1]])  # Prefix the index-ed label with __label__
    # print(label)
    cur_row.append(label)
    cur_row.extend(nltk.word_tokenize(row[0].lower()))
    # cur_row.extend(nltk.word_tokenize(row[2].lower()))
    return cur_row


def preprocess(input_file, output_file, keep=1):
    """
    A Map-reduce approach applied on the all_rows list
    Initially shuffle is called, typically needed by NLP classification algorithms
    """
    all_rows = []
    with open(input_file, "r", encoding="utf-8") as csvinfile:
        csv_reader = csv.reader(csvinfile, delimiter=",")
        for row in csv_reader:
            all_rows.append(row)
    shuffle(all_rows)
    all_rows = all_rows[: int(keep * len(all_rows))]
    print(f"Number of processes: {multiprocessing.cpu_count()}")
    pool = Pool(processes=multiprocessing.cpu_count()//4)
    transformed_rows = pool.map(transform_instance, all_rows)
    pool.close()
    pool.join()

    with open(output_file, "w") as csvoutfile:
        csv_writer = csv.writer(csvoutfile, delimiter=" ", lineterminator="\n")
        csv_writer.writerows(transformed_rows)


if __name__ == '__main__':

    preprocess("data/dbpedia_train.csv", "data/dbpedia.train", keep=0.2)

    # Preparing the validation dataset
    preprocess("data/dbpedia_test.csv", "data/dbpedia.validation")

