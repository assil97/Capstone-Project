import csv

#opening the file and reading the contents in the file row by row
with open('/Users/syedshakeeb/Desktop/faceVGG/attendance.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    #iterating to print the content row by row
    for row in reader:
        print(row)

csvFile.close()