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
mathjax_script = dji.Import(src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-MML-AM_CHTML")

listOfCodes = ['3d-ccptwo-prompi', '3d-ccptwo-flash', '3d-ccptwo-music', '3d-ccptwo-slh', '3d-ccptwo-slh2']
listOfComparison = ['3d-ccptwo-comparison']
listOfModels = ['3d-ccptwo-prompi-models']

codes = [{'label': 'PROMPI', 'value': '3d-ccptwo-prompi'},
         {'label': 'FLASH', 'value': '3d-ccptwo-flash'},
         {'label': 'SLH', 'value': '3d-ccptwo-slh'},
         {'label': 'SLH (latest April/2021)', 'value': '3d-ccptwo-slh2'},
         {'label': 'MUSIC', 'value': '3d-ccptwo-music'},
         {'label': 'COMPARISON', 'value': '3d-ccptwo-comparison'}]
# {'label': 'MODELS (based on PROMPI)', 'value': '3d-ccptwo-prompi-models'}]

equations = [{'label': 'Continuity Equation with Favrian Dilatation', 'value': 'conteq'},
             {'label': 'Continuity Equation with Turbulent Mass Flux', 'value': 'conteqfdd'},
             {'label': 'Momentum Equation X', 'value': 'momex'},
             {'label': 'Turbulent Kinetic Energy Equation', 'value': 'tkeeq'},
             {'label': 'Transport Equation for Fluid1ronds', 'value': 'xtrseq_fluid1'},
             {'label': 'Transport Equation for Fluid2ronds', 'value': 'xtrseq_fluid2'},
             {'label': 'Variance Equation for Fluid1ronds', 'value': 'xvareq_fluid1'},
             {'label': 'Variance Equation for Fluid2ronds', 'value': 'xvareq_fluid2'},
             {'label': 'Source, Mean and Turbulent Velocities', 'value': 'srcvel'}]

comparison = [{'label': 'Velocities', 'value': 'urmstke'},
              {'label': 'Composition Flux', 'value': 'xflux'}]

dictOptions = {'3d-ccptwo-prompi': equations,
               '3d-ccptwo-flash': equations,
               '3d-ccptwo-slh': equations,
               '3d-ccptwo-slh2': equations,
               '3d-ccptwo-music': equations,
               '3d-ccptwo-comparison': comparison}

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
                                        value='3d-ccptwo-prompi',
                                        multi=False
                                    ),
                                ], style=dict(width='40%')),
                                html.Div([
                                    html.Label(
                                        'RANS Equation: (or mean field for COMPARISON option in Code: dropdown)'),
                                    dcc.Dropdown(
                                        id='equation',
                                        multi=False,
                                        options=equations,
                                        value='conteq'
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
                            dcc.Graph(id='figRANS'),
                        ]),
                        mathjax_script
                    ])
                ], width=3)], align='center'),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Div(html.Footer("2021 Created by mmicromegas"))
                ], width=9)], align='center'),
        ])
    )
])


def getParams(codeSelect):
    global params

    if codeSelect in listOfCodes:
        if codeSelect == '3d-ccptwo-prompi':
            paramFile = os.path.join('PARAMS', 'PROMPI', 'param.ransxi')
            params = ReadParamsRansXi(paramFile)
        elif codeSelect == '3d-ccptwo-flash':
            paramFile = os.path.join('PARAMS', 'FLASH', 'param.ransxi')
            params = ReadParamsRansXi(paramFile)
        elif codeSelect == '3d-ccptwo-music':
            paramFile = os.path.join('PARAMS', 'MUSIC', 'param.ransxi')
            params = ReadParamsRansXi(paramFile)
        elif codeSelect == '3d-ccptwo-slh':
            paramFile = os.path.join('PARAMS', 'SLH', 'param.ransxi')
            params = ReadParamsRansXi(paramFile)
        elif codeSelect == '3d-ccptwo-slh2':
            paramFile = os.path.join('PARAMS', 'SLH', 'param.ransxi2')
            params = ReadParamsRansXi(paramFile)
        else:
            print('ERROR (app.py): code not supported (update_table)')
    elif codeSelect in listOfComparison:
        # create os independent path and read parameter file
        paramFile = os.path.join('PARAMS', 'COMPARISON', 'param.compare')
        params = CompareReadParamsRansX(paramFile)
    elif codeSelect in listOfModels:
        pass
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

        data = {'Name of Property': ['Resolution', 'Depth of the Convection Zone (in ccp units)',
                                     'Effective Reynolds Number'],
                'Value': [str(prp['nx']) + 'x' + str(prp['ny']) + 'x' + str(prp['nz']), prp['lc'], prp['Re']],
                'Name of Property ': ['Time-Averaging Window (in turnover timescales)', 'Central Time (in ccp units)',
                                      'Averaging Time-Range (From-To in ccp units)'],
                'Value ': [prp['tavg_to'], prp['timec'], str(prp['timerange_beg']) + '-' + str(prp['timerange_end'])],
                'Name of Property  ': ['Convective Turnover Timescale (in ccp units)',
                                       'Turbulent Kinetic Energy Dissipation Timescale (in ccp units)',
                                       'Root-Mean-Square Turbulence Velocity (in ccp units)'],
                'Value  ': [prp['tc'], prp['tD'], prp['urms']]}

        #    [prp['tc'], prp['tD'], '%.2e' % prp['urms']]

        df = pd.DataFrame(data)
    elif codeSelect in listOfComparison:
        # calculate properties
        params = getParams('3d-ccptwo-prompi')  # hardcoded - assuming all loaded data have the same resolution,
        # averaging window, and approx. central time
        ransP = Properties(params)
        prp = ransP.properties()

        data = {'Name of Property': ['Resolution', 'Averaging Time-Range (From-To in ccp units)'],
                'Value': [str(prp['nx']) + 'x' + str(prp['ny']) + 'x' + str(prp['nz']),
                          str(prp['timerange_beg']) + '-' + str(prp['timerange_end'])],
                'Name of Property ': ['Time-Averaging Window (in turnover timescales)',
                                      'Approximate Central Time (in ccp units)'],
                'Value ': [prp['tavg_to'], prp['timec']]}

        #    [prp['tc'], prp['tD'], '%.2e' % prp['urms']]

        df = pd.DataFrame(data)
    elif codeSelect in listOfModels:
        # calculate properties
        params = getParams('3d-ccptwo-prompi')  # hardcoded as the models are based on PROMPI only
        ransP = Properties(params)
        prp = ransP.properties()

        data = {'Name of Property': ['Resolution', 'Depth of the Convection Zone (in ccp units)',
                                     'Effective Reynolds Number'],
                'Value': [str(prp['nx']) + 'x' + str(prp['ny']) + 'x' + str(prp['nz']), prp['lc'], prp['Re']],
                'Name of Property ': ['Time-Averaging Window (in turnover timescales)', 'Central Time (in ccp units)',
                                      'Averaging Time-Range (From-To in ccp units)'],
                'Value ': [prp['tavg_to'], prp['timec'], str(prp['timerange_beg']) + '-' + str(prp['timerange_end'])],
                'Name of Property  ': ['Convective Turnover Timescale (in ccp units)',
                                       'Turbulent Kinetic Energy Dissipation Timescale (in ccp units)',
                                       'Root-Mean-Square Turbulence Velocity (in ccp units)'],
                'Value  ': [prp['tc'], prp['tD'], prp['urms']]}

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

        if equationSelect not in ['conteq', 'conteqfdd', 'momex', 'tkeeq', 'eieq', 'srcvel', 'xtrseq_fluid1',
                                  'xtrseq_fluid2', 'xvareq_fluid1', 'xvareq_fluid2']:
            equationSelect = 'conteq'  # fallback option when coming from comparison

        # calculate properties
        ransP = Properties(params)
        prp = ransP.properties()

        # extract some properties
        bconv = prp['xzn0inc']
        tconv = prp['xzn0outc']
        tke_diss = prp['tke_diss']

        # instantiate master plot
        plt = MasterPlot(params)

        # CONTINUITY EQUATION WITH FAVRIAN DILATATION
        if equationSelect == 'conteq':
            fig = plt.execContEq(bconv, tconv)

        # CONTINUITY EQUATION WITH TURBULENT MASS FLUX
        if equationSelect == 'conteqfdd':
            fig = plt.execContEqFdd(bconv, tconv)

        # MOMENTUM EQUATION IN X DIRECTION
        if equationSelect == 'momex':
            fig = plt.execMomex(bconv, tconv)

        # TURBULENT KINETIC ENERGY EQUATION
        if equationSelect == 'tkeeq':
            fig = plt.execTkeEq(bconv, tconv)

        # INTERNAL ENERGY EQUATION
        if equationSelect == 'eieq':
            fig = plt.execEiEq(bconv, tconv, tke_diss)

        # VELOCITY
        if equationSelect == 'srcvel':
            fig = plt.execSrcvel(bconv, tconv)

        # load network
        network = params.getNetwork()

        # COMPOSITION TRANSPORT
        for elem in network[1:]:  # skip network identifier in the list
            inuc = params.getInuc(network, elem)

            # COMPOSITION TRANSPORT EQUATION
            if equationSelect == 'xtrseq_' + elem:
                fig = plt.execXtrsEq(inuc, elem, equationSelect, bconv, tconv)

            # COMPOSITION VARIANCE EQUATION
            if equationSelect == 'xvareq_' + elem:
                fig = plt.execXvarEq(inuc, elem, equationSelect, bconv, tconv)

    elif codeSelect in listOfComparison:

        if equationSelect not in ['urmstke', 'xflux']:
            equationSelect = 'urmstke'

        # instantiate master plot
        plt = MasterPlot(params)

        # VELOCITY AND TKE
        if equationSelect == 'urmstke':
            fig = plt.execUxComparison()

        # COMPOSITION FLUX
        if equationSelect == 'xflux':
            fig = plt.execXfluxComparison()


    elif codeSelect in listOfModels:
        pass
    else:
        print('ERROR (app.py): code not supported (update_figRANS)')

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
