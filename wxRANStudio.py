import wx

import ast
import os
import sys
import errno

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
        ransEquations = ['Continuity Equation with Favrian Dilatation',
                     'Continuity Equation with Turbulent Mass Flux',
                     'Momentum Equation (in X-direction)',
                     'Momentum Equation (in Y-direction)',
                     'Momentum Equation (in Z-direction)',
                     'Turbulent Kinetic Energy Equation',
                     'Temperature Equation']
        self.box.Add(chlbl2)
        self.choice2 = wx.Choice(self.panel, choices=ransEquations, size=(250,50), pos=(10,90))
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
        return

    def OnChoice2(self, event):
        print(self.choice2.GetString(self.choice2.GetSelection()))

        paramFile = os.path.join('PARAMS', 'param.ransx')
        params = ReadParamsRansX(paramFile)

        # get input parameters
        filename = params.getForProp('prop')['eht_data']
        plabel = params.getForProp('prop')['plabel']
        ig = params.getForProp('prop')['ig']
        nsdim = params.getForProp('prop')['nsdim']
        ieos = params.getForProp('prop')['ieos']
        intc = params.getForProp('prop')['intc']
        laxis = params.getForProp('prop')['laxis']
        xbl = str('%.2e' % params.getForProp('prop')['xbl'])
        xbr = str('%.2e' % params.getForProp('prop')['xbr'])

        ybu = str('%.2e' % params.getForEqs('tkie')['ybu'])
        ybd = str('%.2e' % params.getForEqs('tkie')['ybd'])

        chlbl3 = wx.StaticText(self.panel, label="Plot Properties:", pos=(10,130))
        self.box.Add(chlbl3)

        chlbl4 = wx.StaticText(self.panel, label="X-Range:", pos=(10,160))
        self.box.Add(chlbl4)
        self.XrangeL = wx.TextCtrl(self.panel, wx.ID_ANY,xbl,pos=(60,160),size=(70,20))
        self.XrangeR = wx.TextCtrl(self.panel, wx.ID_ANY,xbr,pos=(140,160),size=(70,20))

        chlbl5 = wx.StaticText(self.panel, label="Y-Range:", pos=(10,180))
        self.box.Add(chlbl5)
        self.YrangeU = wx.TextCtrl(self.panel, wx.ID_ANY,ybu,pos=(60,180),size=(70,20))
        self.YrangeD = wx.TextCtrl(self.panel, wx.ID_ANY,ybd,pos=(140,180),size=(70,20))



    def onClick(self, event):
        paramFile = os.path.join('PARAMS', 'param.ransx')
        params = ReadParamsRansX(paramFile)

        # get input parameters
        # filename = params.getForProp('prop')['eht_data']
        plabel = params.getForProp('prop')['plabel']
        ig = params.getForProp('prop')['ig']
        nsdim = params.getForProp('prop')['nsdim']
        ieos = params.getForProp('prop')['ieos']
        intc = params.getForProp('prop')['intc']
        laxis = params.getForProp('prop')['laxis']
        xbl = params.getForProp('prop')['xbl']
        xbr = params.getForProp('prop')['xbr']

        # calculate properties
        ransP = Properties(self.filename, plabel, ig, nsdim, ieos, intc, laxis, xbl, xbr)
        prp = ransP.properties()

        self.ransInfo.AppendText('Total nuclear luminosity (in erg/s): %.2e' % prp['tenuc'])

        # instantiate master plot
        plt = MasterPlot(params)

        # obtain publication quality figures
        plt.SetMatplotlibParams()

        # set wxStudio flag to True to allow GUI work properly
        wxStudio = True

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

    def OnChoice3(self, event):
        print(self.choice3.GetString(self.choice3.GetSelection()))

    def on_test(self, event):  # define event handler for the b_test button
        print("Button pressed")
        event.Skip()  # all it for other event handler registered for it


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
