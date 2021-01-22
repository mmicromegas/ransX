# wxPython RANSx Studio #

# File: wxRANStudio.py
# Author: Miroslav Mocak
# Email: miroslav.mocak@gmail.com
# Date: January/2021
# Desc: simple GUI to ransX framework
# Usage: python wxRANStudio.py

import wx

import ast
import os
import sys
import errno
import pandas as pd

from UTILS.RANSX.Properties import Properties
from UTILS.RANSX.ReadParamsRansX import ReadParamsRansX
from UTILS.RANSX.MasterPlot import MasterPlot
# import matplotlib.pyplot as plt


# wxPython basics
# 1. insert a widget (button, list, dropdown etc.) :: wx.Button etc.
# 2. bind an event and event handler to the widget :: self.Bind
# 3. code the event handler i.e. a method doing something on event :: def on_test(self, event)

class myFrame(wx.Frame):  # define application with initialization routine
    def __init__(self, *args, **kwds):  # define constructor referring to itself with arguments
        wx.Frame.__init__(self, *args, **kwds, size=(800, 430))  # initialize frame again with self and args and keywords
        self.panel = wx.Panel(self, wx.ID_ANY)  # panel where our widgets will be placed

        pic=wx.StaticBitmap(self.panel, pos=(20,250))
        pic.SetBitmap(wx.Bitmap(os.path.join('UTILS', 'ransx.png')))

        self.box = wx.BoxSizer(wx.VERTICAL)

        self.datadir = os.path.join('DATA', 'TSERIES')
        tseries = [filee for filee in sorted(os.listdir(self.datadir)) if "tseries" in filee]

        # create simple menu
        menubar = wx.MenuBar()
        menu_1 = wx.Menu()
        m_quit = menu_1.Append(wx.ID_ANY, "Quit\tCtrl+Q")
        menubar.Append(menu_1,"File")
        menu_2 = wx.Menu()
        m_about = menu_2.Append(wx.ID_ANY, "About")
        menubar.Append(menu_2,"Help")
        self.SetMenuBar(menubar)
        self.Bind(wx.EVT_MENU, self.on_quit, m_quit)
        self.Bind(wx.EVT_MENU, self.on_about, m_about)

        chlbl1 = wx.StaticText(self.panel, label="Select Input Data:", pos=(10,10))
        self.box.Add(chlbl1)
        self.choice1 = wx.Choice(self.panel, choices=tseries, size=(355,50), pos=(10,30))
        self.choice1.SetSelection(0)
        self.box.Add(self.choice1)
        self.choice1.Bind(wx.EVT_CHOICE, self.OnChoice1)


        chlbl2 = wx.StaticText(self.panel, label="Select RANS Equation:", pos=(10,70))

        self.ransEquations = pd.Series(['Continuity Equation with Favrian Dilatation',
                     'Continuity Equation with Turbulent Mass Flux',
                     'Momentum Equation (in X-direction)',
                     'Momentum Equation (in Y-direction)',
                     'Momentum Equation (in Z-direction)',
                     'Turbulent Kinetic Energy Equation',
                     'Temperature Equation'],
                            index=['conteq', 'conteqfdd',
                                   'momxeq', 'momyeq', 'momzeq','tkeeq','tteq'])

        self.box.Add(chlbl2)
        self.choice2 = wx.Choice(self.panel, choices=self.ransEquations.values.tolist(), size=(250,50), pos=(10,90))
        self.choice2.SetSelection(0)
        self.box.Add(self.choice2)
        self.choice2.Bind(wx.EVT_CHOICE, self.OnChoice2)

        self.ransInfo = wx.TextCtrl(self.panel, wx.ID_ANY,"Welcome to wxPython RANSx Studio, the GUI for ransX framework",pos=(390,30),size=(370,330), style= wx.TE_MULTILINE|wx.TE_READONLY)

        # add button, on our panel, with default object id
        self.b_test = wx.Button(self.panel, wx.ID_ANY, "Show", pos=(265,75),size=(100,50))


        # we need to bind it to event handler (clicking on button is the EVT_BUTTON event)
        self.Bind(wx.EVT_BUTTON, self.onClick, self.b_test)

        filename = os.path.join(self.datadir,self.choice1.GetString(self.choice1.GetSelection()))

        eqSelection = self.choice2.GetString(self.choice2.GetSelection())
        xbl = str('%.2e' % self.getPropSelection(filename, eqSelection, 'xbl'))
        xbr = str('%.2e' % self.getPropSelection(filename, eqSelection, 'xbr'))
        ybu = str('%.2e' % self.getPropSelection(filename, eqSelection, 'ybu'))
        ybd = str('%.2e' % self.getPropSelection(filename, eqSelection, 'ybd'))

        chlbl3 = wx.StaticText(self.panel, label="Plot Properties:", pos=(10,130))
        self.box.Add(chlbl3)

        chlbl4 = wx.StaticText(self.panel, label="X-Range:", pos=(10,160))
        self.box.Add(chlbl4)
        self.XrangeL = wx.TextCtrl(self.panel, wx.ID_ANY,xbl,pos=(60,160),size=(70,20),style=wx.TE_PROCESS_ENTER)
        self.XrangeR = wx.TextCtrl(self.panel, wx.ID_ANY,xbr,pos=(140,160),size=(70,20),style=wx.TE_PROCESS_ENTER)

        chlbl5 = wx.StaticText(self.panel, label="Y-Range:", pos=(10,180))
        self.box.Add(chlbl5)
        self.YrangeU = wx.TextCtrl(self.panel, wx.ID_ANY,ybu,pos=(60,180),size=(70,20),style=wx.TE_PROCESS_ENTER)
        self.YrangeD = wx.TextCtrl(self.panel, wx.ID_ANY,ybd,pos=(140,180),size=(70,20),style=wx.TE_PROCESS_ENTER)


        #panel.SetSizer(box)
        #self.Centre()
        self.Show()

    def OnChoice1(self, event):
        filename = os.path.join(self.datadir, self.choice1.GetString(self.choice1.GetSelection()))
        eqSelection = self.choice2.GetString(self.choice2.GetSelection())
        if eqSelection != "":
            self.XrangeL.ChangeValue(str('%.2e' % self.getPropSelection(filename,eqSelection,'xbl')))
            self.XrangeR.ChangeValue(str('%.2e' % self.getPropSelection(filename,eqSelection,'xbr')))
            self.YrangeD.ChangeValue(str('%.2e' % self.getPropSelection(filename,eqSelection,'ybd')))
            self.YrangeU.ChangeValue(str('%.2e' % self.getPropSelection(filename,eqSelection,'ybu')))
        return

    def OnChoice2(self, event):
        filename = os.path.join(self.datadir, self.choice1.GetString(self.choice1.GetSelection()))
        eqSelection = self.choice2.GetString(self.choice2.GetSelection())
        xbl = str('%.2e' % self.getPropSelection(filename, eqSelection, 'xbl'))
        xbr = str('%.2e' % self.getPropSelection(filename, eqSelection, 'xbr'))
        ybu = str('%.2e' % self.getPropSelection(filename, eqSelection, 'ybu'))
        ybd = str('%.2e' % self.getPropSelection(filename, eqSelection, 'ybd'))

        chlbl3 = wx.StaticText(self.panel, label="Plot Properties:", pos=(10,130))
        self.box.Add(chlbl3)

        chlbl4 = wx.StaticText(self.panel, label="X-Range:", pos=(10,160))
        self.box.Add(chlbl4)
        self.XrangeL = wx.TextCtrl(self.panel, wx.ID_ANY,xbl,pos=(60,160),size=(70,20),style=wx.TE_PROCESS_ENTER)
        self.XrangeR = wx.TextCtrl(self.panel, wx.ID_ANY,xbr,pos=(140,160),size=(70,20),style=wx.TE_PROCESS_ENTER)

        chlbl5 = wx.StaticText(self.panel, label="Y-Range:", pos=(10,180))
        self.box.Add(chlbl5)
        self.YrangeU = wx.TextCtrl(self.panel, wx.ID_ANY,ybu,pos=(60,180),size=(70,20),style=wx.TE_PROCESS_ENTER)
        self.YrangeD = wx.TextCtrl(self.panel, wx.ID_ANY,ybd,pos=(140,180),size=(70,20),style=wx.TE_PROCESS_ENTER)

    def onClick(self, event):

        filename = os.path.join(self.datadir, self.choice1.GetString(self.choice1.GetSelection()))
        eqSelection = self.choice2.GetString(self.choice2.GetSelection())
        plabel = self.getPropSelection(filename, eqSelection, 'plabel')
        ig = self.getPropSelection(filename, eqSelection, 'ig')
        nsdim = self.getPropSelection(filename, eqSelection, 'nsdim')
        ieos = self.getPropSelection(filename, eqSelection, 'ieos')
        intc = self.getPropSelection(filename, eqSelection, 'intc')
        laxis = self.getPropSelection(filename, eqSelection, 'laxis')
        xbl = self.getPropSelection(filename, eqSelection, 'xbl')
        xbr = self.getPropSelection(filename, eqSelection, 'xbr')

        # calculate properties
        ransP = Properties(filename, plabel, ig, nsdim, ieos, intc, laxis, xbl, xbr)
        prp = ransP.properties()

        # clear ransInfo text area first
        self.ransInfo.Clear()
        # self.ransInfo.AppendText('Datafile with space-time averages: ' + self.filename + "\n")

        self.ransInfo.AppendText('Central time (in s): ' + str(prp['timec']) + "\n")
        self.ransInfo.AppendText('Averaging windows (in s): ' + str(prp['tavg'])+ "\n")
        self.ransInfo.AppendText('Time range (in s from-to): ' + str(prp['tfrom']) + ' ' + str(prp['tto']) + "\n")
        self.ransInfo.AppendText('---------------'+ "\n")
        self.ransInfo.AppendText('Resolution: ' + str(prp['nx']) + ' ' + str(prp['ny']) + ' ' + str(prp['nz']) + "\n")
        self.ransInfo.AppendText('Radial size of computational domain (in cm): %.2e %.2e' % (prp['xzn0in'], prp['xzn0out']) + "\n")
        self.ransInfo.AppendText('Radial size of convection zone (in cm):  %.2e %.2e' % (prp['xzn0inc'], prp['xzn0outc'])+ "\n")
        self.ransInfo.AppendText('Total nuclear luminosity (in erg/s): %.2e' % prp['tenuc'] + "\n")
        self.ransInfo.AppendText('RMS velocities in convection zone (in cm/s):  %.2e' % prp['urms'] + "\n")
        self.ransInfo.AppendText('Convective turnover timescale (in s)  %.2e' % prp['tc'] + "\n")
        self.ransInfo.AppendText('P_turb o P_gas %.2e' % prp['pturb_o_pgas'] + "\n")
        self.ransInfo.AppendText('Mach number Max (using uu) %.2e' % prp['machMax_2'] + "\n")
        self.ransInfo.AppendText('Mach number Mean (using uu) %.2e' % prp['machMean_2'] + "\n")
        self.ransInfo.AppendText('Dissipation length scale (in cm): %.2e' % prp['ld'] + "\n")
        self.ransInfo.AppendText('Total nuclear luminosity (in erg/s): %.2e' % prp['tenuc'] + "\n")
        self.ransInfo.AppendText('Rate of TKE dissipation (in erg/s): %.2e' % prp['epsD'] + "\n")
        self.ransInfo.AppendText('Dissipation timescale for TKE (in s): %f' % prp['tD'] + "\n")
        self.ransInfo.AppendText('Dissipation timescale for TKE vertical (in s): %f' % prp['tDver'] + "\n")
        self.ransInfo.AppendText('Dissipation timescale for TKE horizontal (in s): %f' % prp['tDhor'] + "\n")

        # instantiate master plot
        paramFile = self.getParamFile(filename)
        params = ReadParamsRansX(paramFile)
        plt = MasterPlot(params)

        # obtain publication quality figures
        plt.SetMatplotlibParams()

        XrangeL = float(self.XrangeL.GetValue())
        XrangeR = float(self.XrangeR.GetValue())

        YrangeU = float(self.YrangeU.GetValue())
        YrangeD = float(self.YrangeD.GetValue())

        # set wxStudio parameter flags to allow GUI work properly
        wxStudio = [True,XrangeL,XrangeR,YrangeU,YrangeD]

        # print(self.XrangeL,type(self.XrangeL))

        # CONTINUITY EQUATION
        if self.choice2.GetString(self.choice2.GetSelection()) == 'Continuity Equation with Favrian Dilatation':
            plt.execContEq(wxStudio, prp['xzn0inc'], prp['xzn0outc'])

        if self.choice2.GetString(self.choice2.GetSelection()) == 'Continuity Equation with Turbulent Mass Flux':
            plt.execContFddEq(wxStudio, prp['xzn0inc'], prp['xzn0outc'])

        # MOMENTUM X EQUATION
        if self.choice2.GetString(self.choice2.GetSelection()) == 'Momentum Equation (in X-direction)':
            plt.execMomxEq(wxStudio, prp['xzn0inc'], prp['xzn0outc'])

        # MOMENTUM Y EQUATION
        if self.choice2.GetString(self.choice2.GetSelection()) == 'Momentum Equation (in Y-direction)':
            plt.execMomyEq(wxStudio, prp['xzn0inc'], prp['xzn0outc'])

        # MOMENTUM Z EQUATION
        if self.choice2.GetString(self.choice2.GetSelection()) == 'Momentum Equation (in Z-direction)':
            plt.execMomzEq(wxStudio, prp['xzn0inc'], prp['xzn0outc'])

        # TURBULENT KINETIC ENERGY
        if self.choice2.GetString(self.choice2.GetSelection()) == 'Turbulent Kinetic Energy Equation':
             plt.execTkeEq(wxStudio, prp['kolm_tke_diss_rate'], prp['xzn0inc'],
                         prp['xzn0outc'], prp['super_ad_i'], prp['super_ad_o'])

        # TEMPERATURE EQUATION
        if self.choice2.GetString(self.choice2.GetSelection()) == 'Temperature Equation':
            plt.execTTeq(wxStudio, prp['tke_diss'], prp['xzn0inc'], prp['xzn0outc'])

    def getPropSelection(self, filename, eqSelection, param):

        paramFile = self.getParamFile(filename)
        params = ReadParamsRansX(paramFile)
        if param not in ['eht_data','plabel','prefix','ig','nsdim','ieos','intc','laxis','fext']:
            eqIndex = self.ransEquations[self.ransEquations == eqSelection].index[0]
            parameter = params.getForEqs(eqIndex)[param]
        else:
            parameter = params.getForProp('prop')[param]

        return parameter

    def getParamFile(self, filename):
        global paramFile
        if "ccptwo" in filename:
            if "2D" in filename:
                if "128" in filename:
                    paramFile = os.path.join('PARAMS/CCPTWO/2D/128', 'param.ransx')
                if "256" in filename:
                    paramFile = os.path.join('PARAMS/CCPTWO/2D/256', 'param.ransx')
                if "512" in filename:
                    paramFile = os.path.join('PARAMS/CCPTWO/2D/512', 'param.ransx')
            if "3D" in filename:
                if "128" in filename:
                    paramFile = os.path.join('PARAMS/CCPTWO/3D/128', 'param.ransx')
                if "256" in filename:
                    paramFile = os.path.join('PARAMS/CCPTWO/3D/256', 'param.ransx')
                if "512" in filename:
                    paramFile = os.path.join('PARAMS/CCPTWO/3D/512', 'param.ransx')
        elif "oburn" in filename:
            if "14" in filename:
                paramFile = os.path.join('PARAMS/OBURN_14elems', 'param.ransx')
            if "25" in filename:
                paramFile = os.path.join('PARAMS/OBURN_25elems', 'param.ransx')

        return paramFile

    def on_quit(self,event):
        sys.exit(0)

    def on_about(self,event):
        wx.MessageBox("wxPython RANSx Studio v1.0")

class myApp(wx.App):  # application class
    def OnInit(self):  # define OnInit
        self.frame = myFrame(None, wx.ID_ANY,
                             title="wxPython RANSx Studio v1.0")  # create frame, None = no parent, wx.ID_ANY is object id
        self.SetTopWindow(self.frame)  # set it as top window on our frame
        self.frame.Show()  # show the frame
        return True


if __name__ == '__main__':  # main line code
    app = myApp(0)  # create instance of our application
    app.MainLoop()  # which goes into event loop and wait for an event
