import os

from tqdm import tqdm

from app import rootdir


schools = ['cornell', 'misc', 'texas', 'washington', 'wisconsin']
training = ['no_cornell', 'no_misc', 'no_texas', 'no_washington', 'no_wisconsin']

with open(os.path.join(rootdir, 'data', 'tokens_no_other_weighted.csv'), 'r') as input_file:
    rows = input_file.readlines()

r = rows[0].split(',')
r.pop(1)
header = ','.join(r)
rows.pop(0)

folder = os.path.join(rootdir, 'data', 'schools_no_other_weighted')

fd = []
for school in schools:
    f = open(os.path.join(folder, f'{school}.csv'), 'w')
    fd.append(f)
    f.write(header)

fd_training = []
for school in training:
    f = open(os.path.join(folder, 'training', f'{school}.csv'), 'w')
    fd_training.append(f)
    f.write(header)

for row in tqdm(rows):
    tokens = row.split(',')
    d_school = tokens.pop(1)
    text = ','.join(tokens)

    fd[schools.index(d_school)].write(text)
    no = f'no_{d_school}'
    m = fd_training.copy()
    m.pop(training.index(no))
    for file in m:
        file.write(text)
