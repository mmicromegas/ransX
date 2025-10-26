#import dash_core_components as dcc
#import dash_html_components as html

from dash import dcc
from dash import html

import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output
import dash
import os

###### important for latex ######
import dash_defer_js_import as dji  # for some reasons had to install it manually with pip install dash_defer_js_import

import pandas as pd

#import dash_table
from dash import dash_table

from UTILS.RANSX.CompareReadParamsRansX import CompareReadParamsRansX
from UTILS.RANSX.Properties import Properties
from UTILS.RANSX.ReadParamsRansXi import ReadParamsRansXi
from UTILS.RANSX.MasterPlot import MasterPlot

from UTILS.REACLIB.ReadParamsReaclib import ReadParamsReaclib


import numpy as np

app = dash.Dash(name='ransX')
server = app.server

###### important for latex #########
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            <script type="text/x-mathjax-config">
            MathJax.Hub.Config({
                tex2jax: {
                inlineMath: [ ['$','$'],],
                processEscapes: true
                }
            });
            </script>
            {%renderer%}
        </footer>
    </body>
</html>
"""

filepath = os.path.split(os.path.realpath(__file__))[0]
md_text_head = open(os.path.join(filepath, "ransX.md"), "r").read()
md_text_empty_line = open(os.path.join(filepath, "ransX-table-empty-line.md"), "r").read()

###### important for latex ######
#mathjax_script = dji.Import(src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG")
#mathjax_script = dji.Import(src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-MML-AM_CHTML")

listOfCodes = ['3d-oburn-prompi']
listOfTimescales = ['3d-oburn-prompi-timescales']

codes = [{'label': 'Oxygen Burning Shell Composition Equations', 'value': '3d-oburn-prompi'},
         {'label': 'Oxygen Burning Shell Timescales', 'value': '3d-oburn-prompi-timescales'}]
# {'label': 'MODELS (based on PROMPI)', 'value': '3d-oburn-prompi-models'}]

equations = [{'label': 'Temperature, Density, Pressure, Internal Energy', 'value': 'tdc'},
             {'label': 'Source, Mean and Turbulent Velocities', 'value': 'srcvel'},
             {'label': 'Transport Equation for prot', 'value': 'xtrseq_prot'},
             {'label': 'Transport Equation for neut', 'value': 'xtrseq_neut'},
             {'label': 'Transport Equation for he4', 'value': 'xtrseq_he4'},
             {'label': 'Transport Equation for c12', 'value': 'xtrseq_c12'},
             {'label': 'Transport Equation for o16', 'value': 'xtrseq_o16'},
             {'label': 'Transport Equation for ne20', 'value': 'xtrseq_ne20'},
             {'label': 'Transport Equation for na23', 'value': 'xtrseq_na23'},
             {'label': 'Transport Equation for mg24', 'value': 'xtrseq_mg24'},
             {'label': 'Transport Equation for si28', 'value': 'xtrseq_si28'},
             {'label': 'Transport Equation for p31', 'value': 'xtrseq_p31'},
             {'label': 'Transport Equation for s32', 'value': 'xtrseq_s32'},
             {'label': 'Transport Equation for s34', 'value': 'xtrseq_s34'},
             {'label': 'Transport Equation for cl35', 'value': 'xtrseq_cl35'},
             {'label': 'Transport Equation for ar36', 'value': 'xtrseq_ar36'}]


timescales = [{'label': 'Timescales for prot', 'value': 'xtimescales_prot'},
              {'label': 'Timescales for neut', 'value': 'xtimescales_neut'},
              {'label': 'Timescales for he4', 'value': 'xtimescales_he4'},
              {'label': 'Timescales for c12', 'value': 'xtimescales_c12'},
              {'label': 'Timescales for o16', 'value': 'xtimescales_o16'},
              {'label': 'Timescales for ne20', 'value': 'xtimescales_ne20'},
              {'label': 'Timescales for na23', 'value': 'xtimescales_na23'},
              {'label': 'Timescales for mg24', 'value': 'xtimescales_mg24'},
              {'label': 'Timescales for si28', 'value': 'xtimescales_si28'},
              {'label': 'Timescales for p31', 'value': 'xtimescales_p31'},
              {'label': 'Timescales for s32', 'value': 'xtimescales_s32'},
              {'label': 'Timescales for s34', 'value': 'xtimescales_s34'},
              {'label': 'Timescales for cl35', 'value': 'xtimescales_cl35'},
              {'label': 'Timescales for ar36', 'value': 'xtimescales_ar36'}]

dictOptions = {'3d-oburn-prompi': equations,
               '3d-oburn-prompi-timescales': timescales}

# initialize properties for table-properties
data = {'Name of Property': ['Resolution', 'Depth of the Convection Zone (in ccp units)', 'Effective Reynolds Number'],
        'Value': ['', '', ''],
        'Name of Property ': ['Time-Averaging Window (in turnover timescales)', 'Central Time (in s)',
                              'Averaging Time-Range (From, To in s)'], 'Value ': ['', '', ''],
        'Name of Property  ': ['Convective Turnover Timescale (in s)',
                               'Turbulent Kinetic Energy Dissipation Timescale (in s)',
                               'Root-Mean-Square Turbulence Velocity (in cm/s)'], 'Value  ': ['', '', '']}
df = pd.DataFrame(data)

app.layout = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dcc.Markdown(md_text_head, dangerously_allow_html=True)
                ], width=3),
            ], align='center'),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Container(children=[
                            # your application content goes here
                            html.Div([
                                html.Div([
                                    html.Label('Code:'),
                                    dcc.Dropdown(
                                        id='code',
                                        options=codes,
                                        value='3d-oburn-prompi',
                                        multi=False
                                    ),
                                ], style=dict(width='40%')),
                                html.Div([
                                    html.Label(
                                        'RANS Transport Equations or Timescales'),
                                    dcc.Dropdown(
                                        id='equation',
                                        multi=False,
                                        options=equations,
                                        value='xtrseq_neut'
                                    ),
                                ], style=dict(width='40%')),
                            ], style=dict(display='flex')),
                            dcc.Markdown(md_text_empty_line, dangerously_allow_html=True),
                            html.Div([
                                # html.Label("Properties of the Simulation"),
                                dash_table.DataTable(
                                    id='table-properties',
                                    columns=[{"name": i, "id": i} for i in df.columns],
                                    data=df.to_dict('records'),
                                )]),
                            dcc.Graph(id='figRANS',mathjax=True),
                        ])
                    ])
                ], width=3)], align='center'),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Div(html.Footer("2024 Created by mmicromegas"))
                ], width=9)], align='center'),
        ])
    )
])


def getParams(codeSelect):
    global params

    if codeSelect in listOfCodes:
        if codeSelect == '3d-oburn-prompi':
            paramFile = os.path.join('PARAMS', 'PROMPI', 'param.ransxi')
            params = ReadParamsRansXi(paramFile)
        else:
            print('ERROR (app.py): code not supported (update_table)')
    elif codeSelect in listOfTimescales:
            if codeSelect == '3d-oburn-prompi-timescales':
                paramFile = os.path.join('PARAMS', 'PROMPI', 'param.reaclib')
                params = ReadParamsReaclib(paramFile)
            else:
                print('ERROR (app.py): code not supported (update_table)')
    else:
        print('ERROR (app.py): code not supported (update_table)')

    return params


@app.callback(
    Output('table-properties', 'data'),
    [Input('code', 'value')])
def update_table(codeSelect):
    # print('from update table',codeSelect)

    global df

    if codeSelect in listOfCodes:
        # calculate properties
        params = getParams(codeSelect)
        ransP = Properties(params)
        prp = ransP.properties()

        data = {'Name of Property': ['Resolution', 'Depth of the Convection Zone (in 10e8 cm)',
                                     'Effective Reynolds Number'],
                'Value': [str(prp['nx']) + 'x' + str(prp['ny']) + 'x' + str(prp['nz']), np.round(prp['lc']/1.e8,1), prp['Re']],
                'Name of Property ': ['Time-Averaging Window (in turnover timescales)', 'Central Time (in seconds)',
                                      'Averaging Time-Range (From-To in seconds)'],
                'Value ': [prp['tavg_to'], prp['timec'], str(prp['timerange_beg']) + '-' + str(prp['timerange_end'])],
                'Name of Property  ': ['Convective Turnover Timescale (in seconds)',
                                       'Turbulent Kinetic Energy Dissipation Timescale (in seconds)',
                                       'Root-Mean-Square Turbulence Velocity (in 10e6 cm/s)'],
                'Value  ': [prp['tc'], prp['tD'], np.round(prp['urms']/1.e6,1)]}

        #    [prp['tc'], prp['tD'], '%.2e' % prp['urms']]

        df = pd.DataFrame(data)
    elif codeSelect in listOfTimescales:
        # calculate properties
        params = getParams(codeSelect)
        ransP = Properties(params)
        prp = ransP.properties()

        data = {'Name of Property': ['Resolution', 'Depth of the Convection Zone (in 10e8 cm)',
                                     'Effective Reynolds Number'],
                'Value': [str(prp['nx']) + 'x' + str(prp['ny']) + 'x' + str(prp['nz']), np.round(prp['lc']/1.e8,1), prp['Re']],
                'Name of Property ': ['Time-Averaging Window (in turnover timescales)', 'Central Time (in seconds)',
                                      'Averaging Time-Range (From-To in seconds)'],
                'Value ': [prp['tavg_to'], prp['timec'], str(prp['timerange_beg']) + '-' + str(prp['timerange_end'])],
                'Name of Property  ': ['Convective Turnover Timescale (in seconds)',
                                       'Turbulent Kinetic Energy Dissipation Timescale (in seconds)',
                                       'Root-Mean-Square Turbulence Velocity (in 10e6 cm/s)'],
                'Value  ': [prp['tc'], prp['tD'], np.round(prp['urms']/1.e6,1)]}

        #    [prp['tc'], prp['tD'], '%.2e' % prp['urms']]

        df = pd.DataFrame(data)
    else:
        print('ERROR (app.py): code not supported (update_table)')

    return df.to_dict('records')  # records is a parameter for to_dict ‘records’ : list like [{column -> value}, … ,
    # {column -> value}]


@app.callback(
    Output('equation', 'options'),
    [Input('code', 'value')]
)
def update_dropdown(name):
    return dictOptions[name]


# update_figRANS
@app.callback(
    Output('figRANS', 'figure'),
    [Input('code', 'value'),
     Input('equation', 'value')])
def update_figRANS(codeSelect, equationSelect):
    global paramFile, fig, params

    params = getParams(codeSelect)

    if codeSelect in listOfCodes:

        if equationSelect not in ['tdc','srcvel', 'xtrseq_prot',
                                  'xtrseq_neut', 'xtrseq_he4', 'xtrseq_c12', 'xtrseq_o16', 'xtrseq_ne20',
                                  'xtrseq_na23', 'xtrseq_mg24', 'xtrseq_si28', 'xtrseq_p31', 'xtrseq_s32',
                                  'xtrseq_s34', 'xtrseq_cl35', 'xtrseq_ar36']:

            equationSelect = 'xtrseq_neut'  # fallback option when coming from comparison

        # calculate properties
        ransP = Properties(params)
        prp = ransP.properties()

        # extract some properties
        bconv = prp['xzn0inc']
        tconv = prp['xzn0outc']
        tke_diss = prp['tke_diss']

        # instantiate master plot
        plt = MasterPlot(params)

        # VELOCITY
        if equationSelect == 'srcvel':
            fig = plt.execSrcvel(bconv, tconv)

        # TEMPERATURE, DENSITY, COMPOSITION
        if equationSelect == 'tdc':
            fig = plt.execTDC(bconv, tconv)

        # load network
        network = params.getNetwork()

        # COMPOSITION TRANSPORT
        for elem in network[1:]:  # skip network identifier in the list
            inuc = params.getInuc(network, elem)

            # COMPOSITION TRANSPORT EQUATION
            if equationSelect == 'xtrseq_' + elem:
                fig = plt.execXtrsEq(inuc, elem, equationSelect, bconv, tconv, prp['tc'])

            # COMPOSITION VARIANCE EQUATION
            #if equationSelect == 'xvareq_' + elem:
            #    fig = plt.execXvarEq(inuc, elem, equationSelect, bconv, tconv)

    elif codeSelect in listOfTimescales:

        if equationSelect not in ['xTimescales_prot',
                                  'xTimescales_neut', 'xTimescales_he4', 'xTimescales_c12', 'xTimescales_o16', 'xTimescales_ne20',
                                  'xTimescales_na23', 'xTimescales_mg24', 'xTimescales_si28', 'xTimescales_p31', 'xTimescales_s32',
                                  'xTimescales_s34', 'xTimescales_cl35', 'xTimescales_ar36']:

            equationSelect = 'xTimescales_neut'  # fallback option when coming from comparison

        # calculate properties
        ransP = Properties(params)
        prp = ransP.properties()

        # extract some properties
        bconv = prp['xzn0inc']
        tconv = prp['xzn0outc']
        tke_diss = prp['tke_diss']


        # instantiate master plot
        plt = MasterPlot(params)

        # load network
        network = params.getNetwork()

        # COMPOSITION TRANSPORT
        for elem in network[1:]:  # skip network identifier in the list
            inuc = params.getInuc(network, elem)

            print('xTimescales_',elem)


            # TIMESCALES
            if equationSelect == 'xTimescales_' + elem:
                fig = plt.execXtransportVSnuclearTimescales(inuc, elem, 'xTimescales_' + elem, prp['xzn0inc'], prp['xzn0outc'],
                                                  prp['tc'])

    else:
        print('ERROR (app.py): code not supported (update_figRANS)')

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
