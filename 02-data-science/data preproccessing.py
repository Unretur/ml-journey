# creating a file in .txt format 

from tkinter import W


with open('example_text.txt',  'w') as file:
    file.write("This is sample file for text.\n")
#print("file created succesfully")


# write multiply line 
with open('example_text.txt',  'w') as file:
     for i in range(1,6):
         file.write(f"This is line {i} text.\n")
#print("file created succesfully")


# reading the text file with open as its in built in mode  
with open ('example_text.txt','r') as file:
    data = file.read()
    #print(data)


## tsv (tab sepreated values) \\t
##wruttinga nd making tsv file  in dataframe 

# creating json
import json

data = {
    "employees": [
        {"name": "jhon","age": 30,"department":"sales"},
        {"name": "preet","age": 20,"department": "marketing"}
    ]
}

#Writtting the dictonary to a json file 
with open('json_example.json','w') as file :
     json.dump(data,file,indent=4)
