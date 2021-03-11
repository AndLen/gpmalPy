import lzma
from pathlib import Path

import pandas as pd
from sklearn import preprocessing


def delim_map(delim):
    switch = {
        "comma": ",",
        "space": " ",
        "tab": "\t"
    }
    return switch.get(delim)


class_features = ['class', 'target', 'label','Class','Target','Label']


def read_data(filename):
    filename1 = filename.with_suffix('').with_suffix('.data')
    if filename1.exists():
        print('.data file.')
        with open(filename1, mode='rt') as f:
            first_line = f.readline()
            print('Header: {}'.format(first_line))
            config = first_line.strip().split(",")
            filename = filename1
    else:
        filename2 = filename.with_suffix('.data.xz')
        if filename2.exists():
            print('compressed .data file.')
            with lzma.open(filename2, mode='rt') as f:
                first_line = f.readline()
                print('Header: {}'.format(first_line))
                config = first_line.strip().split(",")
            filename = filename2

        else:
            filename3 = filename.with_suffix('').with_suffix('.csv')
            if filename3.exists():
                print('raw .csv file.')
                with open(filename3, mode='rt') as f:
                    first_line = f.readline()
                    header = first_line.strip().split(",")
                    print('Header: {}'.format(header))
                    if header[0] in class_features:
                        config = ['classFirst', first_line.count(','), None, 'comma']
                    elif header[-1] in class_features:
                        config = ['classLast', first_line.count(','), None, 'comma']
                    else:
                        raise ValueError('Class should be either the first or last feature, and called \'class\','
                                         '\'target\' or \'label\'')
                filename = filename3
            else:
                # there has to be a better way to do this
                filename4 = filename.with_suffix('').with_suffix('.tsv')
                if filename4.exists():
                    print('raw .tsv file.')
                    with open(filename4, mode='rt') as f:
                        first_line = f.readline()
                        header = first_line.strip().split("\t")
                        print('Header: {}'.format(header))
                        if header[0] in class_features:
                            config = ['classFirst', first_line.count('\t'), None, 'tab']
                        elif header[-1] in class_features:
                            config = ['classLast', first_line.count('\t'), None, 'tab']
                        else:
                            raise ValueError('Class should be either the first or last feature, and called \'class\','
                                             '\'target\' or \'label\'')
                    filename = filename4
                else:
                    raise ValueError(filename)

    classPos = config[0]
    num_feat = int(config[1])

    feat_labels = ['f' + str(x) for x in range(num_feat)]
    if classPos == "classFirst":
        feat_labels.insert(0, "class")
    elif classPos == "classLast":
        feat_labels.append("class")
    else:
        raise ValueError(classPos)

    delim = delim_map(config[3])
    print('Reading {}'.format(filename))
    rawData = pd.read_csv(filename, delimiter=delim, skiprows=1, header=None, names=feat_labels)
    labels = rawData['class']
    data = rawData.drop('class', axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    return {"data": data, "labels": labels}


if __name__ == '__main__':
    filename = "/home/lensenandr/datasetsPy/wine"
    read_data(Path(filename))
