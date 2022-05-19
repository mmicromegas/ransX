# main.py
from flask import Flask, Request
from flask import url_for, jsonify, render_template, request

import ast
import os
import sys
import uuid

from UTILS.RANSX.Properties import Properties
from UTILS.RANSX.ReadParamsRansX import ReadParamsRansX
from UTILS.RANSX.MasterPlot import MasterPlot

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/readme')
def readme():
    return render_template('readme.html')

@app.route('/ransx', methods=['POST'])  # /ransx is your endpoint
def ransx():
    global params
    selectModelEquation = request.get_json(force=True)

    modelSelect = selectModelEquation.get('modelSelect')
    equationSelect = selectModelEquation.get('equationSelect')
    resSelect = selectModelEquation.get('resSelect')

    #print(resSelect, modelSelect, selectModelEquation)

    if modelSelect == '2d-ccptwo' and resSelect == 'res128x128':
        # create os independent path and read parameter file
        paramFile = os.path.join('PARAMS', 'param.ransx.2d.ccptwo.128x128')
        params = ReadParamsRansX(paramFile)
    elif modelSelect == '2d-ccptwo' and resSelect == 'res512x512':
        # create os independent path and read parameter file
        paramFile = os.path.join('PARAMS', 'param.ransx.2d.ccptwo.512x512')
        params = ReadParamsRansX(paramFile)
    elif modelSelect == '3d-ccptwo' and resSelect == 'res128x128x128':
        paramFile = os.path.join('PARAMS', 'param.ransx.3d.ccptwo.128x128x128')
        params = ReadParamsRansX(paramFile)
    elif modelSelect == '3d-ccptwo' and resSelect == 'res256x256x256':
        paramFile = os.path.join('PARAMS', 'param.ransx.3d.ccptwo.256x256x256')
        params = ReadParamsRansX(paramFile)
    elif modelSelect == '3d-ccptwo' and resSelect == 'res512x512x512':
        paramFile = os.path.join('PARAMS', 'param.ransx.3d.ccptwo.512x512x512')
        params = ReadParamsRansX(paramFile)
    elif modelSelect == '3d-oburn-14ele' and resSelect == 'res128x64x64':
        paramFile = os.path.join('PARAMS', 'param.ransx.3d14.oburn.128x64x64')
        params = ReadParamsRansX(paramFile)
    elif modelSelect == '3d-oburn-14ele' and resSelect == 'res512x128x128':
        paramFile = os.path.join('PARAMS', 'param.ransx.3d14.oburn.512x128x128')
        params = ReadParamsRansX(paramFile)
    elif modelSelect == '3d-oburn-25ele':
        paramFile = os.path.join('PARAMS', 'param.ransx.3d25.oburn')
        params = ReadParamsRansX(paramFile)
    elif modelSelect == '3d-neshellBoost10x-25ele':
        paramFile = os.path.join('PARAMS', 'param.ransx.3d25.neshellNucBoost10x')
        params = ReadParamsRansX(paramFile)
    elif modelSelect == '3d-heflashBoost100x-6ele':
        paramFile = os.path.join('PARAMS', 'param.ransx.3d6.heflashNucBoost100x')
        params = ReadParamsRansX(paramFile)
    elif modelSelect == '3d-thpulse-15ele':
        paramFile = os.path.join('PARAMS', 'param.ransx.3d15.thpulse')
        params = ReadParamsRansX(paramFile)
    else:
        print('ERROR(app.py): no model selected')

    # get input parameters
    filename = params.getForProp('prop')['eht_data']
    plabel = params.getForProp('prop')['plabel']
    ig = params.getForProp('prop')['ig']
    nsdim = params.getForProp('prop')['nsdim']
    ieos = params.getForProp('prop')['ieos']
    intc = params.getForProp('prop')['intc']
    laxis = params.getForProp('prop')['laxis']
    xbl = params.getForProp('prop')['xbl']
    xbr = params.getForProp('prop')['xbr']

    # calculate properties
    ransP = Properties(filename, plabel, ig, nsdim, ieos, intc, laxis, xbl, xbr)
    prp = ransP.properties()

    # instantiate master plot
    plt = MasterPlot(params)

    # obtain publication quality figures
    plt.SetMatplotlibParams()

    # remove all previous html plots
    # part of hack preventing iframe/html browser caching
    outputDir = 'static'
    outputDirFiles = os.listdir(outputDir)

    for item in outputDirFiles:
        if item.endswith(".html"):
            os.remove(os.path.join(outputDir, item))

    # give the output python html plot from mpld3 a unique name using UUID
    # this is part of hack preventing iframe/html browser caching
    uniqueid = str(uuid.uuid1().int)
    outputFile = 'pythonPlot' + uniqueid + '.html'
    outputFilePath = os.path.join(outputDir, outputFile)

    # SOURCE TERM, TEMPERATURE AND RMS OF FLUCTUATIONS
    if equationSelect == 'srctempflct':
        plt.execSrcTempFlct(outputFilePath, prp['xzn0inc'], prp['xzn0outc'], plabel, nsdim)

    # VELOCITIES AND TEMPERATURE GRADIENTS
    if equationSelect == 'velnablas':
        plt.execVelNablas(outputFilePath, prp['xzn0inc'], prp['xzn0outc'], prp['uconv'], prp['super_ad_i'], prp['super_ad_o'], plabel, nsdim)

    # CONTINUITY EQUATION WITH FAVRIAN DILATATION
    if equationSelect == 'conteq':
        plt.execContEq(outputFilePath, prp['xzn0inc'], prp['xzn0outc'], plabel, nsdim)

    # CONTINUITY EQUATION WITH TURBULENT MASS FLUX
    if equationSelect == 'conteqfdd':
        plt.execContFddEq(outputFilePath, prp['xzn0inc'], prp['xzn0outc'], plabel, nsdim)

    # MOMENTUM EQUATION X
    if equationSelect == 'momxeq':
        plt.execMomx(outputFilePath, prp['xzn0inc'], prp['xzn0outc'], plabel, nsdim)

    # ENTROPY EQUATION
    if equationSelect == 'sseq':
        plt.execSSeq(outputFilePath, prp['xzn0inc'], prp['xzn0outc'], prp['tke_diss'], plabel, nsdim)

    # TURBULENT KINETIC ENERGY EQUATION
    if equationSelect == 'tkeeq':
        plt.execTkeEq(outputFilePath, prp['kolm_tke_diss_rate'], prp['xzn0inc'],
                      prp['xzn0outc'], prp['super_ad_i'], prp['super_ad_o'], plabel, nsdim)

    # INTERNAL ENERGY EQUATION
    if equationSelect == 'eieq':
        plt.execEiEq(outputFilePath, prp['xzn0inc'], prp['xzn0outc'], prp['tke_diss'], plabel, nsdim)

    # load network
    network = params.getNetwork()

    # COMPOSITION TRANSPORT, FLUX, VARIANCE EQUATIONS and EULERIAN DIFFUSIVITY
    for elem in network[1:]:  # skip network identifier in the list
        inuc = params.getInuc(network, elem)

        # COMPOSITION TRANSPORT EQUATION
        if equationSelect == 'xtrseq_' + elem:
            plt.execXtrsEq(outputFilePath, inuc, elem, equationSelect,
                           prp['xzn0inc'],
                           prp['xzn0outc'], prp['super_ad_i'], prp['super_ad_o'], plabel)


    youtubeEmbed = ''
    if modelSelect == '2d-ccptwo' or modelSelect == '3d-ccptwo' :
        youtubeEmbed = 'https://www.youtube.com/embed/1CDoR3JAR3w'
    elif resSelect == 'res128x64x64' and modelSelect == '3d-oburn-14ele':
        youtubeEmbed = 'https://www.youtube.com/embed/9W6f0CjsmuI'
    elif resSelect == 'res512x128x128' and modelSelect == '3d-oburn-14ele':
        youtubeEmbed = 'https://www.youtube.com/embed/rLvrtau-zog'

    # GET EMBED LINKS FOR YOUTUBE VIDEO
    for elem in network[1:]:  # skip network identifier in the list
        inuc = params.getInuc(network, elem)

        if resSelect == 'res128x64x64' and modelSelect == '3d-oburn-14ele':
            if equationSelect == 'xtrseq_neut':
                youtubeEmbed = 'https://www.youtube.com/embed/ALRKyeFVaxg'
            if equationSelect == 'xtrseq_prot':
                youtubeEmbed = 'https://www.youtube.com/embed/GEL8KXfwo34'
            if equationSelect == 'xtrseq_he4':
                youtubeEmbed = 'https://www.youtube.com/embed/_rAa0_K54rE'
            if equationSelect == 'xtrseq_c12':
                youtubeEmbed = 'https://www.youtube.com/embed/3HjJNZ7fMsY'
            if equationSelect == 'xtrseq_o16':
                youtubeEmbed = 'https://www.youtube.com/embed/ch5gMmh3dJA'
            if equationSelect == 'xtrseq_ne20':
                youtubeEmbed = 'https://www.youtube.com/embed/xd-avhamnpI'
            if equationSelect == 'xtrseq_na23':
                youtubeEmbed = 'https://www.youtube.com/embed/P3uqKwDbckw'
            if equationSelect == 'xtrseq_mg24':
                youtubeEmbed = 'https://www.youtube.com/embed/r8ehfJkKrWs'
            if equationSelect == 'xtrseq_si28':
                youtubeEmbed = 'https://www.youtube.com/embed/pSzloNaAy0U'
            if equationSelect == 'xtrseq_p31':
                youtubeEmbed = 'https://www.youtube.com/embed/E5vzHswYWsk'
            if equationSelect == 'xtrseq_s32':
                youtubeEmbed = 'https://www.youtube.com/embed/aFUcacgEX7k'
            if equationSelect == 'xtrseq_s34':
                youtubeEmbed = 'https://www.youtube.com/embed/46j4srdK5Hw'
            if equationSelect == 'xtrseq_cl35':
                youtubeEmbed = 'https://www.youtube.com/embed/SumN7GtpRYE'
            if equationSelect == 'xtrseq_ar36':
                youtubeEmbed = 'https://www.youtube.com/embed/R1TXs9t9grU'
        elif resSelect == 'res512x128x128' and modelSelect == '3d-oburn-14ele':
            if equationSelect == 'xtrseq_neut':
                youtubeEmbed = 'https://www.youtube.com/embed/hzYtL4DWCqY'
            if equationSelect == 'xtrseq_prot':
                youtubeEmbed = 'https://www.youtube.com/embed/Odcq0RM2E6I'
            if equationSelect == 'xtrseq_he4':
                youtubeEmbed = 'https://www.youtube.com/embed/Z5ACzQj6R9M'
            if equationSelect == 'xtrseq_c12':
                youtubeEmbed = 'https://www.youtube.com/embed/5jPu-RFfdW4'
            if equationSelect == 'xtrseq_o16':
                youtubeEmbed = 'https://www.youtube.com/embed/vPXG22FCPOU'
            if equationSelect == 'xtrseq_ne20':
                youtubeEmbed = 'https://www.youtube.com/embed/Bhqc5Udzluc'
            if equationSelect == 'xtrseq_na23':
                youtubeEmbed = 'https://www.youtube.com/embed/S4CGQuRqMWQ'
            if equationSelect == 'xtrseq_mg24':
                youtubeEmbed = 'https://www.youtube.com/embed/2J2Vy_EXyT0'
            if equationSelect == 'xtrseq_si28':
                youtubeEmbed = 'https://www.youtube.com/embed/RPlEe-Hi-JI'
            if equationSelect == 'xtrseq_p31':
                youtubeEmbed = 'https://www.youtube.com/embed/tdt7MZ7U7kg'
            if equationSelect == 'xtrseq_s32':
                youtubeEmbed = 'https://www.youtube.com/embed/X0_EiNBr3Oo'
            if equationSelect == 'xtrseq_s34':
                youtubeEmbed = 'https://www.youtube.com/embed/AuHhk1Etc5I'
            if equationSelect == 'xtrseq_cl35':
                youtubeEmbed = 'https://www.youtube.com/embed/L9XlOfU-0LU'
            if equationSelect == 'xtrseq_ar36':
                youtubeEmbed = 'https://www.youtube.com/embed/rtGsPElX_80'

    # print('THIS IS TEST VALUE PASSED FROM HTML:',test)

    # , 'filename': prp['filename'], 'timec': prp['timec'], 'tavg': prp['tavg'],
    # 'nx': prp['nx'], 'ny': prp['ny'], 'nz': prp['nz'], 'xzn0in': prp['xzn0in'],
    # 'xzn0out': prp['xzn0out'], 'xzn0inc': prp['xzn0inc'], 'xzn0outc': prp['xzn0outc']

    # print(prp['timec'], prp['tavg'],type(prp['nx']),prp['nx'],prp['machMean_2'])
    data = [{'outputFile': outputFile, 'filename': prp['filename'], 'timec': prp['timec'],
             'tavg': prp['tavg'], 'tavg_to': prp['tavg_to'], 'nx': prp['nx'], 'ny': prp['ny'], 'nz': prp['nz'],
             'xzn0in': prp['xzn0in'], 'xzn0out': prp['xzn0out'], 'xzn0inc': prp['xzn0inc'], 'xzn0outc': prp['xzn0outc'],
             'Re': int(prp['Re']), 'mach': prp['machMean_2'], 'youtubeEmbed': youtubeEmbed, 'network': str(network[1:]),
             'tc': prp['tc'], 'tD': prp['tD'], 'urms': prp['urms'], 'timec': prp['timec'], 'trange': prp['trange']}]

    return jsonify(data)


# True/False strings to proper boolean
def str2bool(param):
    return ast.literal_eval(param)


if __name__ == "__main__":
    app.run(debug=True)
