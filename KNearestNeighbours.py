import csv
import glob

datasets = {"0":[], "1": [], "3": [], "4": []}

for file in glob.glob("processed.*.data.csv"):
    with open (file) as fileIn:
        reader = csv.reader(fileIn)

        for line in reader:
            data = line
            missing = False
            for i in data:
                if i == "?":
                    missing = True
                    break
            if missing:
                continue
            datasets[data[13]].append(data)



