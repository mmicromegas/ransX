import wx


# wxPython basics
# 1. insert a widget (button, list, dropdown etc.) :: wx.Button etc.
# 2. bind an event and event handler to the widget :: self.Bind
# 3. code the event handler i.e. a method doing something on event :: def on_test(self, event)

class myFrame(wx.Frame):  # define application with initialization routine
    def __init__(self, *args, **kwds):  # define constructor referring to itself with arguments
        wx.Frame.__init__(self, *args, **kwds)  # initialize frame again with self and args and keywords
        self.panel = wx.Panel(self, wx.ID_ANY)  # panel where our widgets will be placed

        # add button, on our panel, with default object id
        self.b_test = wx.Button(self.panel, wx.ID_ANY, "Hi",
                                pos=(60, 60))

        # we need to bind it to event handler (clicking on button is the EVT_BUTTON event)
        self.Bind(wx.EVT_BUTTON, self.on_test,
                  self.b_test)

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
