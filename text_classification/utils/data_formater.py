import openpyxl as yxl
from openpyxl import load_workbook
import os
import re

source_path = r'C:\Users\king\Desktop\数据.xlsx'
target = ['../data/neg.txt']

wb = load_workbook(source_path)
ws1 = wb['Sheet1']
for j, row in enumerate(ws1.iter_rows()):
    if j == 0:
        continue
    row_txt = ''
    for i, cell in enumerate(row):
        if type(cell.value) == str:
            row_txt += re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a-\uFF00-\uFFEF])", "", cell.value)
        elif type(cell.value) == int:
            row_txt += str(cell.value)
        row_txt += ' '
    print(row_txt)
