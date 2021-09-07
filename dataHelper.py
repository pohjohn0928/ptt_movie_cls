import csv
import cchardet
from sklearn.utils import shuffle


def read_dataset(file_name):
    file = open(file_name, 'rb')
    encoding = cchardet.detect(file.read())['encoding']
    csvfile = open(file_name, newline='', encoding=encoding)
    reader = csv.DictReader(x.replace('\0', '') for x in csvfile)

    pos = []
    pos_comments = []
    neg = []
    neg_comments = []
    neu = []
    neu_comments = []

    for row in reader:
        label = row['sentiment']
        main_content = row['main_content']
        comment = row['comment']
        if label:
            label = int(label)
            if label == 0:
                neg.append(main_content)
                neg_comments.append(comment)
            elif label == 1:
                pos.append(main_content)
                pos_comments.append(comment)
            elif label == 2:
                neu.append(main_content)
                neu_comments.append(comment)

    use_num = min(len(neg), len(pos), len(neu))

    neg, neg_comments = shuffle(neg, neg_comments)
    pos, pos_comments = shuffle(pos, pos_comments)
    neu, neu_comments = shuffle(neu, neu_comments)

    neg, neg_comments = neg[:use_num], neg_comments[:use_num]
    pos, pos_comments = pos[:use_num], pos_comments[:use_num]
    neu, neu_comments = neu[:use_num], neu_comments[:use_num]

    main_contents = neg + pos + neu
    comments = neg_comments + pos_comments + neu_comments
    labels = [0] * use_num + [1] * use_num + [2] * use_num

    main_contents, comments, labels = shuffle(main_contents, comments, labels)
    return main_contents, comments, labels
