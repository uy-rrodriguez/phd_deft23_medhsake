import sys

instances = {}
with open(sys.argv[1]) as fp:
    for line in fp:
        id, labels = line.strip().split(';')
        instances[id] = ''.join(sorted(labels.split('|'))).lower()

instances2 = {}
with open(sys.argv[2]) as fp:
    for line in fp:
        id, labels = line.strip().split(';')
        instances2[id] = ''.join(sorted(labels.split('|'))).lower()

assert set(instances.keys()) == set(instances2.keys())

identical = 0
for id in instances.keys():
    if instances[id] == instances2[id]:
        identical += 1

print(identical /  len(list(instances.keys())))
