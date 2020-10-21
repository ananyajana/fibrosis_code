import os
import numpy as np
path='./'

file_names = []
for dirpath, dirs, files in os.walk(path): # recurse through the current directory
    for dirname in dirs:
        dname = os.path.join(dirpath, dirname)
        #print(dirname)
        if 'data' not in dname and 'fibrosis' not in dname:  # get the full direcotry name/path:
            onlyfiles = [f for f in os.listdir(dname) if os.path.isfile(os.path.join(dname, f))]    # check files in a particular directory
            for i in range(len(onlyfiles)):
                file = onlyfiles[i]
                if file.endswith('xlsx'):
                    print(file)
                    file_names.append(os.path.join(dname, file))

import pandas as pd
import xlsxwriter
## Method 2 gets all sheets of a given file
df_total = pd.DataFrame()
writer = pd.ExcelWriter('final_output.xlsx', engine='xlsxwriter')
for file in file_names:                         # loop through Excel files
    if file.endswith('.xlsx'):
        excel_file = pd.ExcelFile(file)
        sheets = excel_file.sheet_names
        print(sheets)
        for sheet in sheets:               # loop through sheets inside an Excel file
            df = excel_file.parse(sheet_name = sheet)
            df.to_excel(writer, index=False, sheet_name=sheet)
writer.save()


