from datetime import date
from random import randint

from dash import Dash, dcc, Output, Input, html, dash_table, State
import plotly.express as px
import pandas as pd
import pyodbc
import dash_daq as daq

app = Dash(__name__) #also loads the css files in assets folder

colsList = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']

mytableInput = dash_table.DataTable(
    columns=[{"name": i, "id": i, "hideable": True} for i in colsList],
    editable = True,
    style_cell={
        'height': 'auto',
        'minWidth': '80px',
        'whiteSpace': 'normal'
    }
)
rowsinputboxInput = daq.NumericInput(
        id='my-numeric-input-1',
        value=10,min=1,max=20
    )

tmpthing = html.P();
app.layout = html.Div(children=[
    html.Div(children=[html.H3('Input', style={'textAlign': 'center', 'margin-bottom': '-50px'})]),
    html.Div(children=["# of rows: ",rowsinputboxInput], style={'display': 'flex', 'justify-content': 'left', 'gap': '5px', 'align-items': 'center', 'margin': '10px'}),
    html.Div(children=[mytableInput]),
    tmpthing
])

@app.callback(
    Output(mytableInput, component_property='data'),
    inputs=[Input(rowsinputboxInput, 'value')]
)
def update_table_size(numOfRows):
    df = pd.DataFrame()
    for col in colsList:
        df[col] = [None] * numOfRows
    return df.to_dict('records')

@app.callback(
    Output(tmpthing, component_property='value'),
    inputs=[],
    state=[Input(mytableInput, 'data'), Input(mytableInput, 'hidden_columns')]
)
def dothings(tabledata, hiddencolumns):
    df = pd.DataFrame(tabledata)
    if (not (hiddencolumns is None)) and len(hiddencolumns) > 0:
        df.drop(hiddencolumns, axis=1, inplace=True)
    #for col in columnsdata:
    #    print(col)
        #df.drop([col['name']], axis=1, inplace=True)
    #print(tabledata)
    print(df)


# Run app
if __name__=='__main__':
    #app.run_server(host="0.0.0.0", debug=False, port=8054)
    app.run_server(debug=False, port=8054)
