import csv
import os

from app import rootdir


header = ['a', 'b', 'c', 'd']
csv_rows = [{'a': 1}, {'b': 2}, {'c': 3, 'b': 5}]

if __name__ == '__main__':
    with open(os.path.join(rootdir, 'data', 'test.csv'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for data in csv_rows:
            writer.writerow(data)
