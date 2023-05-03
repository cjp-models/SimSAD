
from docx import Document
import pandas as pd


    

def to_word(data: dict, report_name:str) -> docx.Document:
    assert type(data) == dict, 'data has to be dict'
    assert '.docx' in report_name, 'report_name has to be a .docx file'
    df = pd.DataFrame(data)
    doc = docx.Document()
    table = doc.add_table(df.shape[0]+1, df.shape[1])
    for j in range(df.shape[-1]):
        table.cell(0,j).text = df.columns[j]
    for i in range(df.shape[0]):
        for j in range(df.shape[-1]):
            table.cell(i+1,j).text = str(df.values[i,j])
    doc.save(f'./{report_name}')
