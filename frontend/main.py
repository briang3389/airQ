from datetime import date
from random import randint

from dash import Dash, dcc, Output, Input, html, dash_table, State
import plotly.express as px
import pandas as pd
import pyodbc
import dash_daq as daq

app = Dash(__name__) #also loads the css files in assets folder

colsList = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
numRows = 10
df = pd.DataFrame()
for col in colsList:
    df[col] = [None] * numRows
#print(df)
mytable = dash_table.DataTable(
    data=df.to_dict('records'),
    columns=[{"name": i, "id": i, "hideable": True} for i in df.columns],
    editable = True,
    style_cell={
        'height': 'auto',
        'minWidth': '80px', 'width': '80px', 'maxWidth': '80px',
        'whiteSpace': 'normal'
    }
)
rowsinputbox = daq.NumericInput(
        id='my-numeric-input-1',
        value=10,min=1,max=20
    )

app.layout = html.Div(children=[
    html.Div(children=["# of rows: ",rowsinputbox], style={'display': 'flex', 'justify-content': 'left', 'gap': '5px', 'align-items': 'center', 'margin': '10px'}),
    html.Div(children=[mytable])
])

@app.callback(
    Output(mytable, component_property='data'),
    inputs=[Input(rowsinputbox, 'value')]
)
def update_table_size(numOfRows):
    df = pd.DataFrame()
    for col in colsList:
        df[col] = [None] * numOfRows
    return df.to_dict('records')


# Run app
if __name__=='__main__':
    #app.run_server(host="0.0.0.0", debug=False, port=8054)
    app.run_server(debug=False, port=8054)
