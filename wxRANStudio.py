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
        wx.Frame.__init__(self, *args, **kwds, size=(700, 600))  # initialize frame again with self and args and keywords
        self.panel = wx.Panel(self, wx.ID_ANY)  # panel where our widgets will be placed

        # add button, on our panel, with default object id
        # self.b_test = wx.Button(self.panel, wx.ID_ANY, "Hi",pos=(10, 10))

        # we need to bind it to event handler (clicking on button is the EVT_BUTTON event)
        # self.Bind(wx.EVT_BUTTON, self.on_test,self.b_test)

        self.box = wx.BoxSizer(wx.VERTICAL)

        self.datadir = "DATA/TSERIES/"
        tseries = [filee for filee in sorted(os.listdir(self.datadir)) if "tseries" in filee]

        chlbl1 = wx.StaticText(self.panel, label="Select Input Data:", pos=(10,10))
        self.box.Add(chlbl1)
        self.choice1 = wx.Choice(self.panel, choices=tseries, size=(355,50), pos=(10,30))
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
        self.box.Add(self.choice2)
        self.choice2.Bind(wx.EVT_CHOICE, self.OnChoice2)

        self.ransInfo = wx.TextCtrl(self.panel, wx.ID_ANY,"",pos=(390,30),size=(270,400), style= wx.TE_MULTILINE|wx.TE_READONLY)

        # add button, on our panel, with default object id
        self.b_test = wx.Button(self.panel, wx.ID_ANY, "Show", pos=(265,75),size=(100,50))


        # we need to bind it to event handler (clicking on button is the EVT_BUTTON event)
        self.Bind(wx.EVT_BUTTON, self.onClick, self.b_test)


        #panel.SetSizer(box)
        #self.Centre()
        self.Show()

    def OnChoice1(self, event):
        self.filename = self.datadir + self.choice1.GetString(self.choice1.GetSelection())

        eqSelection = self.choice2.GetString(self.choice2.GetSelection())
        if eqSelection != "":
            self.XrangeL.ChangeValue(str('%.2e' % self.getPropSelection(self.filename,eqSelection,'xbl')))
            self.XrangeR.ChangeValue(str('%.2e' % self.getPropSelection(self.filename,eqSelection,'xbr')))
            self.YrangeD.ChangeValue(str('%.2e' % self.getPropSelection(self.filename,eqSelection,'ybd')))
            self.YrangeU.ChangeValue(str('%.2e' % self.getPropSelection(self.filename,eqSelection,'ybu')))
        return

    def OnChoice2(self, event):
        print(self.choice2.GetString(self.choice2.GetSelection()))

        eqSelection = self.choice2.GetString(self.choice2.GetSelection())
        xbl = str('%.2e' % self.getPropSelection(self.filename, eqSelection, 'xbl'))
        xbr = str('%.2e' % self.getPropSelection(self.filename, eqSelection, 'xbr'))
        ybu = str('%.2e' % self.getPropSelection(self.filename, eqSelection, 'ybu'))
        ybd = str('%.2e' % self.getPropSelection(self.filename, eqSelection, 'ybd'))

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

        eqSelection = self.choice2.GetString(self.choice2.GetSelection())
        plabel = self.getPropSelection(self.filename, eqSelection, 'plabel')
        ig = self.getPropSelection(self.filename, eqSelection, 'ig')
        nsdim = self.getPropSelection(self.filename, eqSelection, 'nsdim')
        ieos = self.getPropSelection(self.filename, eqSelection, 'ieos')
        intc = self.getPropSelection(self.filename, eqSelection, 'intc')
        laxis = self.getPropSelection(self.filename, eqSelection, 'laxis')
        xbl = self.getPropSelection(self.filename, eqSelection, 'xbl')
        xbr = self.getPropSelection(self.filename, eqSelection, 'xbr')

        # calculate properties
        ransP = Properties(self.filename, plabel, ig, nsdim, ieos, intc, laxis, xbl, xbr)
        prp = ransP.properties()

        self.ransInfo.AppendText('Total nuclear luminosity (in erg/s): %.2e' % prp['tenuc'])

        # instantiate master plot
        paramFile = self.getParamFile(self.filename)
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

        print(self.XrangeL,type(self.XrangeL))

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
        if "ccptwo" in self.filename:
            if "2D" in self.filename:
                if "128" in self.filename:
                    paramFile = os.path.join('PARAMS/CCPTWO/2D/128', 'param.ransx')
                if "256" in self.filename:
                    paramFile = os.path.join('PARAMS/CCPTWO/2D/256', 'param.ransx')
                if "512" in self.filename:
                    paramFile = os.path.join('PARAMS/CCPTWO/2D/512', 'param.ransx')
            if "3D" in self.filename:
                if "128" in self.filename:
                    paramFile = os.path.join('PARAMS/CCPTWO/3D/128', 'param.ransx')
                if "256" in self.filename:
                    paramFile = os.path.join('PARAMS/CCPTWO/3D/256', 'param.ransx')
                if "512" in self.filename:
                    paramFile = os.path.join('PARAMS/CCPTWO/3D/512', 'param.ransx')
        elif "oburn" in self.filename:
            if "14" in self.filename:
                paramFile = os.path.join('PARAMS/OBURN_14elems', 'param.ransx')
            if "25" in self.filename:
                paramFile = os.path.join('PARAMS/OBURN_25elems', 'param.ransx')

        return paramFile


class myApp(wx.App):  # application class
    def OnInit(self):  # define OnInit
        self.frame = myFrame(None, wx.ID_ANY,
                             title="wxPython RANSx Studio")  # create frame, None = no parent, wx.ID_ANY is object id
        self.SetTopWindow(self.frame)  # set it as top window on our frame
        self.frame.Show()  # show the frame
        return True


if __name__ == '__main__':  # main line code
    app = myApp(0)  # create instance of our application
    app.MainLoop()  # which goes into event loop and wait for an event
