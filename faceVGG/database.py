import csv

with open('/Users/syedshakeeb/Desktop/faceVGG/attendance.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        print(row)

csvFile.close()