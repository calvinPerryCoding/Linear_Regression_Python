#USER INPUT
dataFileToOpen = "employeeWageData.csv"
yearsListToOpen = "yearsList_1900_To_2200.csv"
#DO NOT CHANGE ANYTHING PAST THIS POINT UNLESS YOU ARE THE DEVELOPER

import csv
import numpy as np
from sklearn.linear_model import LinearRegression

#Open CSV file and store data into a list (Maciej Gol & AMC, 2014)
with open(dataFileToOpen, newline='') as csvfile:
    valueData = list(csv.reader(csvfile))

with open(yearsListToOpen, newline='') as csvfile:
    yearsList = list(csv.reader(csvfile))

#Seperate data (Ranjith, 2019)
year = np.array(valueData)[:,0].reshape(-1,1)
value = np.array(valueData)[:,1].reshape(-1,1)

yearsToPredict = yearsList

#This is the actual linear regression (Ranjith, 2019)
yearsToPredict = np.array(yearsToPredict).reshape(-1,1)
regsr = LinearRegression()
regsr.fit(year,value)
valuePrediction = regsr.predict(yearsToPredict)
regsr = LinearRegression()
regsr.fit(year,value)

#When rounding with numpy, use np.round() isntead of round() (The SciPy community, 2021)
valuePredictionRounded = np.round(valuePrediction, 2)

print("\n\nYear     ", " Value")
print("====================")
#Combines both yearsToPredict and valuePredictionRounded (SCB, 2018)
res = "\n".join("{} {}".format(x, y) for x, y in zip(yearsToPredict, valuePredictionRounded))
print(res)

fields = ["Year", "Value"]

#Don't forget to add , newline = "" to get rid of spaces in csv file (Fabre, 2017)
with open("Results.csv", "w", newline = "") as f:
    write = csv.writer(f)
    write.writerow(fields)
    #Use zip(list1, list2, ect...) when writing multiple lists to a csv file (Trammer, 2013)
    write.writerows(zip(yearsToPredict, valuePredictionRounded))



# References APA 7th Edition


# Fabre, J. F. (2017, September 5). Python skips line when printing to CSV. Stack Overflow. https://stackoverflow.com/questions/46057470/python-skips-line-when-printing-to-csv

# Maciej Gol, & AMC. (2014, July 9). Python import csv to list. Stack Overflow. https://stackoverflow.com/questions/24662571/python-import-csv-to-list

# Ranjith, S. (2019, December 2). Predict next number in a sequence in Python scikit-learn. CodeSpeedy. https://www.codespeedy.com/predict-next-number-in-a-sequence-with-scikit-learn/

# SCB. (2018, January 1). Print 2 lists side by side. Stack Overflow. https://stackoverflow.com/questions/48053979/print-2-lists-side-by-side/48054025

# Trammer, L. (2013, October 10). How to write data from two lists into columns in a csv? Stack Overflow. https://stackoverflow.com/questions/19302612/how-to-write-data-from-two-lists-into-columns-in-a-csv

# The SciPy community. (2021, January 31). Numpy.around — NumPy v1.20 manual. NumPy. https://numpy.org/doc/stable/reference/generated/numpy.around.html

