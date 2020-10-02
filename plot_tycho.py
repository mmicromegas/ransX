import TYCHO as tch

ftycho = 'DATA_D/INIMODEL/imodel.tycho'

star = tch.TYCHO(ftycho)

star.plot_dd_tt_tycho_ini()
star.plot_x_tycho_ini()
