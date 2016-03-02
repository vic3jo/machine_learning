import csv

lines = []    
def convertToCSV(inputFile,outputFile):
    global lines
    lines = []
    
    while True:
        row = []    
        line = inputFile.readline()
        if line == "":
            break
        line = line.split('\n')       
        nline = line[0]    
        row = nline.split(",")
        writer = csv.writer(outputFile,delimiter = ",")
        writer.writerow(row)

def main():
    global lines    
    f = open("Data\poker-hand-testing.data","r")
    f2 = open("Data\poker-hand-testing.csv","wb")
    writer = csv.writer(f2,delimiter = ",")
    row = ['C1','R1','C2','R2','C3','R3','C4','R4','C5','R5','Hand']
    writer.writerow(row)
    convertToCSV(f,f2)
    f.close()
    f2.close()


if __name__ == "__main__":# -*- coding: utf-8 -*-
    main()

