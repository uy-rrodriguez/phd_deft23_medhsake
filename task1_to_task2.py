import sys
for line in sys.stdin:
    id, letters = line.strip().split(';')
    letters = letters.strip()
    print(id + ';' + str(len(letters.split('|'))))
