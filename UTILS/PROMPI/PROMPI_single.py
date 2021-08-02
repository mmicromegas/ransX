import numpy as np
import matplotlib.pyplot as plt

from UTILS.Calculus import Calculus
from UTILS.PROMPI.PROMPI_data import PROMPI_ransdat
from pylab import *

import sys

# class for plotting background stratification of PROMPI models from ransdat

class PROMPI_single(PROMPI_ransdat, Calculus, object):

    def __init__(self, filename, endianness, precision):
        super(PROMPI_single, self).__init__(filename, endianness, precision)
        self.data = self.rans()

    def SetMatplotlibParams(self):
        """ This routine sets some standard values for matplotlib """
        """ to obtain publication-quality figures """

        # plt.rc('text',usetex=True)
        # plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        plt.rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
        plt.rc('font', size=14.)
        plt.rc('lines', linewidth=2, markeredgewidth=2., markersize=10)
        plt.rc('axes', linewidth=1.5)
        plt.rcParams['xtick.major.size'] = 8.
        plt.rcParams['xtick.minor.size'] = 4.
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['figure.subplot.left'] = 0.17
        plt.rcParams['figure.subplot.right'] = 0.85

    def plot_log_q1q2(self, xbl, xbr, f1, f2, xlabel_1, ylabel_1, ylabel_2, plabel_1, plabel_2):
        rr = np.asarray(self.data['xzn0'])
        f_1 = np.abs(self.data[f1]) + 1.
        f_2 = self.data[f2]

        idxl, idxr = self.idx_bndry(xbl, xbr)

        to_plt1 = np.log10(f_1)
        to_plt2 = np.log10(f_2)

        fig, ax1 = plt.subplots(figsize=(7, 6))

        ax1.axis([xbl, xbr, np.min(to_plt1[idxl:idxr]), np.max(to_plt1[idxl:idxr])])
        ax1.plot(rr, to_plt1, color='b', label=plabel_1)

        ax1.set_xlabel(xlabel_1)
        ax1.set_ylabel(ylabel_1)
        ax1.legend(loc=7, prop={'size': 18})

        ax2 = ax1.twinx()
        ax2.axis([xbl, xbr, np.min(to_plt2[idxl:idxr]), np.max(to_plt2[idxl:idxr])])
        ax2.plot(rr, to_plt2, color='r', label=plabel_2)
        ax2.set_ylabel(ylabel_2)
        ax2.tick_params('y')
        ax2.legend(loc=1, prop={'size': 18})

        plt.show(block=False)

    def plot_log_q1(self, xbl, xbr, f1, xlabel_1, ylabel_1, plabel_1):
        rr = np.asarray(self.data['xzn0'])
        f_1 = self.data[f1]

        idxl, idxr = self.idx_bndry(xbl, xbr)

        to_plt1 = f_1
        #to_plt1 = f_1*self.data['dd']
        #to_plt1 = f_1 + self.data['enuc2']
        print('Time:', self.data['rans_tstart'])

        fig, ax1 = plt.subplots(figsize=(7, 6))

        #ax1.axis([xbl,xbr,np.min(to_plt1[idxl:idxr]),np.max(to_plt1[idxl:idxr])])
        ax1.semilogy(rr, to_plt1, color='b', label=plabel_1)

        # fmonstar = 'C:\\Users\\mmocak\\Desktop\\GITDEV\\ransX\\DATA_D\\INIMODEL\\imodel.monstar'
        fmonstar = 'DATA_D\\INIMODEL\\imodel.monstar'
        tdata = open(fmonstar, 'r')

        header_line1 = tdata.readline().split()
        header_line2 = tdata.readline().split()
        header_line3 = tdata.readline().split()

        t_line1 = tdata.readline().split()

        rrm = []
        epspp = []
        epscno = []
        epshe = []
        prot,he4,he3 = [],[],[]
        dd = []
        uconv = []
        for i in range(1996):
            rl = tdata.readline().split()
            rrm.append(rl[2])
            dd.append(rl[6])
            epspp.append(rl[7])
            epscno.append(rl[8])
            epshe.append(rl[9])
            prot.append(rl[11])
            he4.append(rl[12])
            he3.append(rl[13])
            uconv.append(rl[18])

            # rrm.append(tdata.readline().split()[2])
            # epspp.append(tdata.readline().split()[7])
            # epscno.append(tdata.readline().split()[8])
            # epshe.append(tdata.readline().split()[9])

        tdata.close()

        # print(epspp,epscno,epshe)

        rrm = np.asarray(rrm[::-1], dtype=float)
        dd = np.asarray(dd[::-1], dtype=float)
        epspp = np.asarray(epspp[::-1], dtype=float)
        epscno = np.asarray(epscno[::-1], dtype=float)
        epshe = np.asarray(epshe[::-1], dtype=float)
        prot = np.asarray(prot[::-1], dtype=float)
        he4 = np.asarray(he4[::-1], dtype=float)
        he3 = np.asarray(he3[::-1], dtype=float)

        dd_i = np.interp(rr, 10 ** rrm, dd)
        epspp_i = np.interp(rr, 10 ** rrm, epspp)
        epscno_i = np.interp(rr, 10 ** rrm, epscno)
        epshe_i = np.interp(rr, 10 ** rrm, epshe)

        prot_i = np.interp(rr, 10 ** rrm, prot)
        he4_i = np.interp(rr, 10 ** rrm, he4)
        he3_i = np.interp(rr, 10 ** rrm, he3)

        #print(he3)

        ax1.semilogy(rr, (epspp_i + epscno_i + epshe_i), color='k', linestyle='--', label='monstar ini')
        #ax1.semilogy(rr, prot_i, color='k', linestyle='--', label='monstar ini')
        #ax1.semilogy(rr, he4_i, color='k', linestyle='--', label='monstar ini')
        #ax1.semilogy(rr, he3_i, color='k', linestyle='--', label='monstar ini')

        # print(to_plt1)

        ax1.set_xlabel(xlabel_1)
        ax1.set_ylabel(ylabel_1)
        ax1.legend(loc=4, prop={'size': 18})

        plt.show(block=False)

    def plot_lin_q1(self, xbl, xbr, f1, xlabel_1, ylabel_1, plabel_1):
        rr = np.asarray(self.data['xzn0'])
        rrl = np.asarray(self.data['xznl'])
        rrr = np.asarray(self.data['xznr'])
        # dd = np.asarray(self.data['dd'])

        print('Time: ',np.asarray(self.data['rans_tstart']))

        f_1 = self.data[f1]

        idxl, idxr = self.idx_bndry(xbl, xbr)

        to_plt1 = f_1

        fig, ax1 = plt.subplots(figsize=(7, 6))

        ax1.axis([xbl, xbr, np.min(to_plt1[idxl:idxr]), np.max(to_plt1[idxl:idxr])])
        # ax1.axis([xbl,xbr,-2.5e6,2.5e6])
        ax1.plot(rr, to_plt1, color='b', label=plabel_1)

        print('Max: ',np.max(to_plt1))

        xlabel_1 = 'x'
        ylabel_1 = f1

        ax1.set_xlabel(xlabel_1)
        ax1.set_ylabel(ylabel_1)
        ax1.legend(loc=1, prop={'size': 18})

        savefig('RESULTS/' + f1 + '.png')

        plt.show(block=False)

    def plot_lin_q1q2(self, xbl, xbr, f1, f2, xlabel_1, ylabel_1, ylabel_2, plabel_1, plabel_2):
        rr = np.asarray(self.data['xzn0'])
        if (f1) == 'enuc':
            f_1 = self.data['enuc1'] + self.data['enuc2']
            f_2 = self.data[f2]
        elif (f2) == 'enuc':
            f_1 = self.data[f1]
            f_2 = self.data['enuc1'] + self.data['enuc2']
        else:
            f_1 = self.data[f1]
            f_2 = self.data[f2]

        idxl, idxr = self.idx_bndry(xbl, xbr)

        to_plt1 = f_1
        to_plt2 = f_2

        fig, ax1 = plt.subplots(figsize=(7, 6))

        ax1.axis([xbl, xbr, np.min(to_plt1[idxl:idxr]), np.max(to_plt1[idxl:idxr])])
        ax1.plot(rr, to_plt1, color='b', label=plabel_1)

        ax1.set_xlabel(xlabel_1)
        ax1.set_ylabel(ylabel_1)
        ax1.legend(loc=7, prop={'size': 18})

        ax2 = ax1.twinx()
        ax2.axis([xbl, xbr, np.min(to_plt2[idxl:idxr]), np.max(to_plt2[idxl:idxr])])
        ax2.plot(rr, to_plt2, color='r', label=plabel_2)
        ax2.set_ylabel(ylabel_2)
        ax2.tick_params('y')
        ax2.legend(loc=1, prop={'size': 18})

        plt.show(block=False)

    def GETRATEcoeff(self, reaction):

        cl = np.zeros(7)

        if (reaction == 'c12_plus_c12_to_p_na23_r'):
            cl[0] = +0.585029E+02
            cl[1] = +0.295080E-01
            cl[2] = -0.867002E+02
            cl[3] = +0.399457E+01
            cl[4] = -0.592835E+00
            cl[5] = -0.277242E-01
            cl[6] = -0.289561E+01

        if (reaction == 'c12_plus_c12_to_he4_ne20_r'):
            cl[0] = +0.804485E+02
            cl[1] = -0.120189E+00
            cl[2] = -0.723312E+02
            cl[3] = -0.352444E+02
            cl[4] = +0.298646E+01
            cl[5] = -0.309013E+00
            cl[6] = +0.115815E+02

        if (reaction == 'he4_plus_c12_to_o16_r'):
            cl[0] = +0.142191E+03
            cl[1] = -0.891608E+02
            cl[2] = +0.220435E+04
            cl[3] = -0.238031E+04
            cl[4] = +0.108931E+03
            cl[5] = -0.531472E+01
            cl[6] = +0.136118E+04

        if (reaction == 'he4_plus_c12_to_o16_nr'):
            cl[0] = +0.184977E+02
            cl[1] = +0.482093E-02
            cl[2] = -0.332522E+02
            cl[3] = +0.333517E+01
            cl[4] = -0.701714E+00
            cl[5] = +0.781972E-01
            cl[6] = -0.280751E+01

        if (reaction == 'o16_plus_o16_to_p_p31_r'):
            cl[0] = +0.852628E+02
            cl[1] = +0.223453E+00
            cl[2] = -0.145844E+03
            cl[3] = +0.872612E+01
            cl[4] = -0.554035E+00
            cl[5] = -0.137562E+00
            cl[6] = -0.688807E+01

        if (reaction == 'o16_plus_o16_to_he4_si28_r'):
            cl[0] = +0.972435E+02
            cl[1] = -0.268514E+00
            cl[2] = -0.119324E+03
            cl[3] = -0.322497E+02
            cl[4] = +0.146214E+01
            cl[5] = -0.200893E+00
            cl[6] = +0.132148E+02

        if (reaction == 'ne20_to_he4_o16_nv'):
            cl[0] = +0.637915E+02
            cl[1] = -0.549729E+02
            cl[2] = -0.343457E+02
            cl[3] = -0.251939E+02
            cl[4] = +0.479855E+01
            cl[5] = -0.146444E+01
            cl[6] = +0.784333E+01

        if (reaction == 'ne20_to_he4_o16_rv'):
            cl[0] = +0.109310E+03
            cl[1] = -0.727584E+02
            cl[2] = +0.293664E+03
            cl[3] = -0.384974E+03
            cl[4] = +0.202380E+02
            cl[5] = -0.100379E+01
            cl[6] = +0.201193E+03

        if (reaction == 'si28_to_he4_mg24_nv1'):
            cl[0] = +0.522024E+03
            cl[1] = -0.122258E+03
            cl[2] = +0.434667E+03
            cl[3] = -0.994288E+03
            cl[4] = +0.656308E+02
            cl[5] = -0.412503E+01
            cl[6] = +0.426946E+03

        if (reaction == 'si28_to_he4_mg24_nv2'):
            cl[0] = +0.157580E+02
            cl[1] = -0.129560E+03
            cl[2] = -0.516428E+02
            cl[3] = +0.684625E+02
            cl[4] = -0.386512E+01
            cl[5] = +0.208028E+00
            cl[6] = -0.320727E+02

        if (reaction == 'he4_plus_si28_to_p_p31_rv1'):
            cl[0] = -0.906347E+01
            cl[1] = -0.241182E+02
            cl[2] = +0.373526E+01
            cl[3] = -0.664843E+01
            cl[4] = +0.254122E+00
            cl[5] = -0.588282E-02
            cl[6] = +0.191121E+01

        if (reaction == 'he4_plus_si28_to_p_p31_rv2'):
            cl[0] = +0.552169E+01
            cl[1] = -0.265651E+02
            cl[2] = +0.456462E-08
            cl[3] = -0.105997E-07
            cl[4] = +0.863175E-09
            cl[5] = -0.640626E-10
            cl[6] = -0.150000E+01

        if (reaction == 'he4_plus_si28_to_p_p31_rv3'):
            cl[0] = -0.126553E+01
            cl[1] = -0.287435E+02
            cl[2] = -0.309775E+02
            cl[3] = +0.458298E+02
            cl[4] = -0.272557E+01
            cl[5] = +0.163910E+00
            cl[6] = -0.239582E+02

        if (reaction == 'he4_plus_si28_to_p_p31_rv4'):
            cl[0] = +0.296908E+02
            cl[1] = -0.330803E+02
            cl[2] = +0.553217E+02
            cl[3] = -0.737793E+02
            cl[4] = +0.325554E+01
            cl[5] = -0.144379E+00
            cl[6] = +0.388817E+02

        if (reaction == 'he4_plus_si28_to_p_p31_rv5'):
            cl[0] = +0.128202E+02
            cl[1] = -0.376275E+02
            cl[2] = -0.487688E+02
            cl[3] = +0.549854E+02
            cl[4] = -0.270916E+01
            cl[5] = +0.142733E+00
            cl[6] = -0.319614E+02

        if (reaction == 'he4_plus_si28_to_p_p31_rv6'):
            cl[0] = +0.381739E+02
            cl[1] = -0.406821E+02
            cl[2] = -0.546650E+02
            cl[3] = +0.331135E+02
            cl[4] = -0.644696E+00
            cl[5] = -0.155955E-02
            cl[6] = -0.271330E+02

        if (reaction == 'he4_plus_o16_to_ne20_n'):
            cl[0] = +0.390340E+02
            cl[1] = -0.358600E-01
            cl[2] = -0.343457E+02
            cl[3] = -0.251939E+02
            cl[4] = +0.479855E+01
            cl[5] = -0.146444E+01
            cl[6] = +0.634333E+01

        if (reaction == 'he4_plus_o16_to_ne20_r'):
            cl[0] = +0.845522E+02
            cl[1] = -0.178214E+02
            cl[2] = +0.293664E+03
            cl[3] = -0.384974E+03
            cl[4] = +0.202380E+02
            cl[5] = -0.100379E+01
            cl[6] = +0.199693E+03

        if (reaction == 'he4_plus_ne20_to_mg24_n'):
            cl[0] = +0.321588E+02
            cl[1] = -0.151494E-01
            cl[2] = -0.446410E+02
            cl[3] = -0.833867E+01
            cl[4] = +0.241631E+01
            cl[5] = -0.778056E+00
            cl[6] = +0.193576E+01

        if (reaction == 'he4_plus_ne20_to_mg24_r'):
            cl[0] = -0.291641E+03
            cl[1] = -0.120966E+02
            cl[2] = -0.633725E+02
            cl[3] = +0.394643E+03
            cl[4] = -0.362432E+02
            cl[5] = +0.264060E+01
            cl[6] = -0.121219E+03

        if (reaction == 'mg24_to_he4_ne20_nv'):
            cl[0] = +0.569781E+02
            cl[1] = -0.108074E+03
            cl[2] = -0.446410E+02
            cl[3] = -0.833867E+01
            cl[4] = +0.241631E+01
            cl[5] = -0.778056E+00
            cl[6] = +0.343576E+01

        if (reaction == 'mg24_to_he4_ne20_rv'):
            cl[0] = -0.266822E+03
            cl[1] = -0.120156E+03
            cl[2] = -0.633725E+02
            cl[3] = +0.394643E+03
            cl[4] = -0.362432E+02
            cl[5] = +0.264060E+01
            cl[6] = -0.119719E+03

        if (reaction == 'p_plus_na23_to_he4_ne20_n'):
            cl[0] = +0.334868E+03
            cl[1] = -0.247143E+00
            cl[2] = +0.371150E+02
            cl[3] = -0.478518E+03
            cl[4] = +0.190867E+03
            cl[5] = -0.136026E+03
            cl[6] = +0.979858E+02

        if (reaction == 'p_plus_na23_to_he4_ne20_r1'):
            cl[0] = +0.942806E+02
            cl[1] = -0.312034E+01
            cl[2] = +0.100052E+03
            cl[3] = -0.193413E+03
            cl[4] = +0.123467E+02
            cl[5] = -0.781799E+00
            cl[6] = +0.890392E+02

        if (reaction == 'p_plus_na23_to_he4_ne20_r2'):
            cl[0] = -0.288152E+02
            cl[1] = -0.447000E+00
            cl[2] = -0.184674E-09
            cl[3] = +0.614357E-09
            cl[4] = -0.658195E-10
            cl[5] = +0.593159E-11
            cl[6] = -0.150000E+01

        if (reaction == 'he4_plus_si28_to_c12_ne20_r'):
            cl[0] = -0.307762E+03
            cl[1] = -0.186722E+03
            cl[2] = +0.514197E+03
            cl[3] = -0.200896E+03
            cl[4] = -0.642713E+01
            cl[5] = +0.758256E+00
            cl[6] = +0.236359E+03

        if (reaction == 'p_plus_p31_to_c12_ne20_r'):
            cl[0] = -0.266452E+03
            cl[1] = -0.156019E+03
            cl[2] = +0.361154E+03
            cl[3] = -0.926430E+02
            cl[4] = -0.998738E+01
            cl[5] = +0.892737E+00
            cl[6] = +0.161042E+03

        if (reaction == 'c12_plus_ne20_to_p_p31_r'):
            cl[0] = -0.268136E+03
            cl[1] = -0.387624E+02
            cl[2] = +0.361154E+03
            cl[3] = -0.926430E+02
            cl[4] = -0.998738E+01
            cl[5] = +0.892737E+00
            cl[6] = +0.161042E+03

        if (reaction == 'c12_plus_ne20_to_he4_si28_r'):
            cl[0] = -0.308905E+03
            cl[1] = -0.472175E+02
            cl[2] = +0.514197E+03
            cl[3] = -0.200896E+03
            cl[4] = -0.642713E+01
            cl[5] = +0.758256E+00
            cl[6] = +0.236359E+03

        if (reaction == 'he4_plus_ne20_to_p_na23_n'):
            cl[0] = +0.335091E+03
            cl[1] = -0.278531E+02
            cl[2] = +0.371150E+02
            cl[3] = -0.478518E+03
            cl[4] = +0.190867E+03
            cl[5] = -0.136026E+03
            cl[6] = +0.979858E+02

        if (reaction == 'he4_plus_ne20_to_p_na23_r1'):
            cl[0] = +0.945037E+02
            cl[1] = -0.307263E+02
            cl[2] = +0.100052E+03
            cl[3] = -0.193413E+03
            cl[4] = +0.123467E+02
            cl[5] = -0.781799E+00
            cl[6] = +0.890392E+02

        if (reaction == 'he4_plus_ne20_to_p_na23_r2'):
            cl[0] = -0.285920E+02
            cl[1] = -0.280530E+02
            cl[2] = -0.184674E-09
            cl[3] = +0.614357E-09
            cl[4] = -0.658195E-10
            cl[5] = +0.593159E-11
            cl[6] = -0.150000E+01

        return cl

    def GET1NUCtimescale(self, c1l, c2l, c3l, c4l, c5l, c6l, c7l, tt):

        temp09 = self.eht_tt[:, tt] * 1.e-9
        rate = exp(c1l + c2l * (temp09 ** (-1.)) + c3l * (temp09 ** (-1. / 3.)) + c4l * (
                    temp09 ** (1. / 3.)) + c5l * temp09 + c6l * (temp09 ** (5. / 3.)) + c7l * np.log(temp09))
        timescale = 1. / rate

        return timescale

    def GET2NUCtimescale(self, c1l, c2l, c3l, c4l, c5l, c6l, c7l, yi, yj, yk, tt):

        temp09 = self.eht_tt[:, tt] * 1.e-9
        rate = exp(c1l + c2l * (temp09 ** (-1.)) + c3l * (temp09 ** (-1. / 3.)) + c4l * (
                    temp09 ** (1. / 3.)) + c5l * temp09 + c6l * (temp09 ** (5. / 3.)) + c7l * np.log(temp09))
        timescale = 1. / (self.eht_dd[:, tt] * yj * yk * rate / yi)

        return timescale

    def GET3NUCtimescale(self, c1l, c2l, c3l, c4l, c5l, c6l, c7l, yi1, yi2, tt):

        temp09 = self.eht_tt[:, tt] * 1.e-9
        rate = exp(c1l + c2l * (temp09 ** (-1.)) + c3l * (temp09 ** (-1. / 3.)) + c4l * (
                    temp09 ** (1. / 3.)) + c5l * temp09 + c6l * (temp09 ** (5. / 3.)) + c7l * np.log(temp09))
        timescale = 1. / (self.eht_dd[:, tt] * self.eht_dd[:, tt] * yi1 * yi2 * rate)

        return timescale

    def PlotNucEnergyGen(self, xbl, xbr):
        """Plot nuclear reaction timescales"""

        nx = np.asarray(self.data['nx'])
        rc = np.asarray(self.data['xzn0'])
        xznl = np.asarray(self.data['xznl'])
        xznr = np.asarray(self.data['xznr'])

        tt = self.data['tt']
        dd = self.data['dd']

        # for 25 element network
        #xhe4 = self.data['x0003']
        #xc12 = self.data['x0004']
        #xo16 = self.data['x0005']
        #xne20 = self.data['x0006']
        #xsi28 = self.data['x0009']

        # for 14 elements network
        xhe4 = self.data['x0003']
        xc12 = self.data['x0004']
        xo16 = self.data['x0005']
        xne20 = self.data['x0006']
        xsi28 = self.data['x0009']

        bconv = 4.2e8
        tconv = 9.5e8

        Vol = 4. / 3. * np.pi * (xznr ** 3 - xznl ** 3)
        ind = np.where((rc > bconv) & (rc < tconv))[0]
        M = (dd * Vol)[ind].sum()

        if 1==1:
            Mhe4 = (dd * xhe4 * Vol)[ind].sum()
            Mc12 = (dd * xc12 * Vol)[ind].sum()
            Mo16 = (dd * xo16 * Vol)[ind].sum()
            Mne20 = (dd * xne20 * Vol)[ind].sum()
            Msi28 = (dd * xsi28 * Vol)[ind].sum()

            xhe4inst = Mhe4/M
            xc12inst = Mc12/M
            xo16inst = Mo16/M
            xne20inst = Mne20/M
            xsi28inst = Msi28/M

            xhe4mean = xhe4[ind].mean()
            xc12mean = xc12[ind].mean()
            xo16mean = xo16[ind].mean()
            xne20mean = xne20[ind].mean()
            xsi28mean = xsi28[ind].mean()

            xc12rd = (xc12mean-xc12inst)/xc12inst
            xo16rd = (xo16mean-xo16inst)/xo16inst
            xne20rd = (xne20mean-xne20inst)/xne20inst
            xsi28rd = (xsi28mean-xsi28inst)/xsi28inst

            print('Xc12 mean:' + str(xc12mean) + '  X inst. mass conserved: ' + str(xc12inst) + ' rel.diff. ' + str(np.round(xc12rd,6)))
            print('Xo16 mean:' + str(xo16mean) + '  X inst. mass conserved: ' + str(xo16inst) + ' rel.diff. ' + str(np.round(xo16rd,6)))
            print('Xne20 mean:' + str(xne20mean) + '  X inst. mass conserved: ' + str(xne20inst) + ' rel.diff. ' + str(np.round(xne20rd,6)))
            print('Xsi28 mean:' + str(xsi28mean) + '  X inst. mass conserved: ' + str(xsi28inst) + ' rel.diff. ' + str(np.round(xsi28rd,6)))

            #xhe4 = np.zeros(nx)
            #xc12 = np.zeros(nx)
            #xo16 = np.zeros(nx)
            #xne20 = np.zeros(nx)
            #xsi28 = np.zeros(nx)

            xbl = rc[0]
            xbr = rc[-1]

            plt.figure(figsize=(7, 6))
            miny = 1.e-6
            maxy = 1.e-4
            plt.axis([xbl, xbr, miny, maxy])
            plt.semilogy(rc,xc12,color='r',label='3D non-instantaneous')
            xc12[ind] = xc12inst
            plt.title(r'X(C12)')
            plt.semilogy(rc,xc12,color='b',label='instantaneous')
            plt.legend(loc=2, prop={'size': 18}, ncol=1)
            plt.ylabel(r"X")
            plt.xlabel('r (cm)')
            plt.show(block=False)
            plt.savefig('RESULTS/xc12.png')

            plt.figure(figsize=(7, 6))
            miny = 4.e-1
            maxy = 5.e-1
            plt.axis([xbl, xbr, miny, maxy])
            plt.semilogy(rc,xo16,color='r',label='3D non-instantaneous')
            xo16[ind] = xo16inst
            plt.title(r'X(O16)')
            plt.semilogy(rc,xo16,color='b',label='instantaneous')
            plt.legend(loc=4, prop={'size': 18}, ncol=1)
            plt.ylabel(r"X")
            plt.xlabel('r (cm)')
            plt.show(block=False)
            plt.savefig('RESULTS/xo16.png')

            plt.figure(figsize=(7, 6))
            miny = 1.e-6
            maxy = 1.e-1
            plt.axis([xbl, xbr, miny, maxy])
            plt.semilogy(rc,xne20,color='r',label='3D non-instantaneous')
            xne20[ind] = xne20inst
            plt.title(r'X(Ne20)')
            plt.semilogy(rc,xne20,color='b',label='instantaneous')
            plt.legend(loc=4, prop={'size': 18}, ncol=1)
            plt.ylabel(r"X")
            plt.xlabel('r (cm)')
            plt.show(block=False)
            plt.savefig('RESULTS/xne20.png')
            plt.savefig('RESULTS/xne20.eps')

            plt.figure(figsize=(7, 6))
            miny = 3.e-1
            maxy = 4.e-1
            plt.axis([xbl, xbr, miny, maxy])
            plt.semilogy(rc,xsi28,color='r',label='3D non-instantaneous')
            xsi28[ind] = xsi28inst
            plt.title(r'X(Si28)')
            plt.semilogy(rc,xsi28,color='b',label='instantaneous')
            plt.legend(loc=3, prop={'size': 18}, ncol=1)
            plt.ylabel(r"X")
            plt.xlabel('r (cm)')
            plt.show(block=False)
            plt.savefig('RESULTS/xsi28.png')


            #print(xne20)
            #sys.exit()

            #  enuc = self.data['enuc1']+self.data['enuc2']


            #xo16 = self.data['x0003']
            #xne20 = self.data['x0004']
            #xc12 = np.zeros(xne20.shape[0])
            #xsi28 = np.zeros(xne20.shape[0])

        enuc1 = self.data['enuc1']
        enuc2 = np.abs(self.data['enuc2'])

        #       ne20 > he4 + o16 (photo-d: resonance)
        t9 = tt / 1.e9
        # + 4.e-2*self.eht_tt[:,tt]/1.e9

        # rate coefficients from netsu (source cf88)

        cl = self.GETRATEcoeff(reaction='ne20_to_he4_o16_rv')
        rate_ne20_alpha_gamma = np.exp(
            cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

        #       he4 + ne20 > mg24
        cl = self.GETRATEcoeff(reaction='he4_plus_ne20_to_mg24_r')
        rate_ne20_alpha_gamma_code = np.exp(
            cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

        #       o16 + o16 > p + p31 (resonance)
        #        xo16 = self.fht_xo16[:,tt]
        cl = self.GETRATEcoeff(reaction='o16_plus_o16_to_p_p31_r')
        rate_o16_o16_pchannel_r = np.exp(
            cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

        #       o16 + o16 > he4 + si28 (resonance)
        #        xo16 = self.fht_xo16[:,tt]
        cl = self.GETRATEcoeff(reaction='o16_plus_o16_to_he4_si28_r')
        rate_o16_o16_achannel_r = np.exp(
            cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

        #       c12 + c12 > p + na23 (resonance)
        cl = self.GETRATEcoeff(reaction='c12_plus_c12_to_p_na23_r')
        rate_c12_c12_pchannel_r = np.exp(
            cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

        #       c12 + c12 > he4 + ne20 (resonance)
        cl = self.GETRATEcoeff(reaction='c12_plus_c12_to_he4_ne20_r')
        rate_c12_c12_achannel_r = np.exp(
            cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

        # ANALYTIC EXPRESSIONS Caughlan & Fowler 1988

        t9a = t9 / (1. + 0.0396 * t9)
        c_tmp1 = (4.27e26) * (t9a ** (5. / 6.))
        c_tmp2 = t9 ** (3. / 2.)
        c_e_tmp1 = -84.165 / (t9a ** (1. / 3.))
        c_e_tmp2 = -(2.12e-3) * (t9 ** 3.)

        rate_c12_c12 = c_tmp1 / c_tmp2 * (np.exp(c_e_tmp1 + c_e_tmp2))

        o_tmp1 = 7.1e36 / (t9 ** (2. / 3.))
        o_c_tmp1 = -135.93 / (t9 ** (1. / 3.))
        o_c_tmp2 = -0.629 * (t9 ** (2. / 3.))
        o_c_tmp3 = -0.445 * (t9 ** (4. / 3.))
        o_c_tmp4 = +0.0103 * (t9 ** 2.)

        rate_o16_o16 = o_tmp1 * np.exp(o_c_tmp1 + o_c_tmp2 + o_c_tmp3 + o_c_tmp4)

        n_tmp1 = 4.11e11 / (t9 ** (2. / 3.))
        n_e_tmp1 = -46.766 / (t9 ** (1. / 3.)) - (t9 / 2.219) ** 2.
        n_tmp2 = 1. + 0.009 * (t9 ** (1. / 3.)) + 0.882 * (t9 ** (2. / 3.)) + 0.055 * t9 + 0.749 * (
                    t9 ** (4. / 3.)) + 0.119 * (t9 ** (5. / 3.))
        n_tmp3 = 5.27e3 / (t9 ** (3. / 2.))
        n_e_tmp3 = -15.869 / t9

        n_tmp4 = 6.51e3 * (t9 ** (1. / 2.))
        n_e_tmp4 = -16.223 / t9

        rate_alpha_gamma_cf88 = n_tmp1 * np.exp(n_e_tmp1) * n_tmp2 + n_tmp3 * np.exp(n_e_tmp3) + n_tmp4 * np.exp(
            n_e_tmp4)

        c1_c12 = 4.8e18
        c1_o16 = 8.e18
        c1_ne20 = 2.5e29
        c1_si28 = 1.8e28

        yc12sq = (xc12 / 12.) ** 2.
        yo16sq = (xo16 / 16.) ** 2.
        yne20sq = (xne20 / 20.) ** 2.

        yo16 = xo16 / 16.

        lag = (3.e-3) * (t9 ** (10.5))
        lox = (2.8e-12) * (t9 / 2.) ** 33.
        lca = (4.e-11) * (t9 ** 29.)
        lsi = 120. * (t9 / 3.5) ** 5.

        en_c12 = c1_c12 * yc12sq * dd * (rate_c12_c12_achannel_r + rate_c12_c12_pchannel_r)
        en_c12_acf88 = c1_c12 * yc12sq * dd * (rate_c12_c12)
        en_o16 = c1_o16 * yo16sq * dd * (rate_o16_o16_achannel_r + rate_o16_o16_pchannel_r)
        en_o16_acf88 = c1_o16 * yo16sq * dd * (rate_o16_o16)
        en_ne20 = c1_ne20 * (t9 ** (3. / 2.)) * (yne20sq / yo16) * rate_ne20_alpha_gamma_code * np.exp(-54.89 / t9)
        en_ne20_acf88 = c1_ne20 * (t9 ** (3. / 2.)) * (yne20sq / yo16) * rate_ne20_alpha_gamma * np.exp(-54.89 / t9)
        en_ne20_hw = c1_ne20 * (t9 ** (3. / 2.)) * (yne20sq / yo16) * lag * np.exp(-54.89 / t9)
        #        en_ne20_ini = c1_ne20*(t9**(3./2.))*(yne20sq_ini/yo16)*rate_ne20_alpha_gamma_code*np.exp(-54.89/t9)
        en_ne20_lag = c1_ne20 * (t9 ** (3. / 2.)) * (yne20sq / yo16) * lag * np.exp(-54.89 / t9)
        en_si28 = c1_si28 * (t9 ** 3. / 2.) * xsi28 * (np.exp(-142.07 / t9)) * rate_ne20_alpha_gamma_code
        en_si28_acf88 = c1_si28 * (t9 ** 3. / 2.) * xsi28 * (np.exp(-142.07 / t9)) * rate_ne20_alpha_gamma

        plt.figure(figsize=(7, 6))

        lb = 1.e-5
        ub = 1.e18

        #plt.yscale('symlog')

        plt.axis([xbl, xbr, lb, ub])

        #plt.title(r'instantaneous')
        plt.title(r'instantaneous')
        plt.semilogy(rc, en_c12, label=r"$\dot{\epsilon}_{\rm nuc}$ (C$^{12}$)")
        plt.semilogy(rc, en_o16, label=r"$\dot{\epsilon}_{\rm nuc}$ (O$^{16}$)")
        plt.semilogy(rc, en_ne20, label=r"$\dot{\epsilon}_{\rm nuc}$ (Ne$^{20}$)")
        plt.semilogy(rc, en_si28, label=r"$\dot{\epsilon}_{\rm nuc}$ (Si$^{28}$)")
        # plt.semilogy(rc, en_c12 + en_o16 + en_ne20 + en_si28,label='total', color='k',linestyle='--')
        plt.plot(rc,enuc1,color='m',linestyle='--',label='enuc1')
        #plt.plot(rc,enuc2,color='r',linestyle='--',label='-neut code')
        # plt.plot(rc,enuc1-enuc2,color='b',linestyle='--',label='enuc1-enuc2')

        en_c12tot = (en_c12 * dd * Vol)[ind].sum()
        en_o16tot = (en_o16 * dd * Vol)[ind].sum()
        en_ne20tot = (en_ne20 * dd * Vol)[ind].sum()
        en_si28tot = (en_si28 * dd * Vol)[ind].sum()

        print('Total Enuc c12 burn:' + str(en_c12tot))
        print('Total Enuc o16 burn:' + str(en_o16tot))
        print('Total Enuc ne20 burn:' + str(en_ne20tot))
        print('Total Enuc si28 burn:' + str(en_si28tot))


        # Clayton, Principles of Stellar Evolution and Nucleosynthesis, page 414, eq.5-105
        f = 1. # screening factor
        tt8 = tt/1.e8
        #epsilon3alpha = 4.4e-8*(dd**2.)*(xhe4**3.)*((tt/1.e8)**40.)*f  # this is around 1e8 K
        epsilon3alpha = 3.9e11*((dd**2.)*(xhe4**3.)/(tt8**3.))*np.exp(-42.94/tt8)
        #print(epsilon3alpha)

        #plt.plot(rc, en_c12, label=r"$\dot{\epsilon}_{\rm nuc}$ (C$^{12}$)")
        #plt.plot(rc, en_o16, label=r"$\dot{\epsilon}_{\rm nuc}$ (O$^{16}$)")
        #plt.plot(rc, en_ne20, label=r"$\dot{\epsilon}_{\rm nuc}$ (Ne$^{20}$)")
        #plt.plot(rc, en_si28, label=r"$\dot{\epsilon}_{\rm nuc}$ (Si$^{28}$)")
        #plt.plot(rc, en_c12 + en_o16 + en_ne20 + en_si28,label='total', color='k')
        #plt.plot(rc, epsilon3alpha, label=r"$\dot{\epsilon}_{\rm nuc}$ (He$^{4}$)")

        #print("en_ne20")
        #print(en_ne20)
        #print("*****")
        #print(en_si28)

        #plt.semilogy(rc,en_c12_acf88,label=r"$\dot{\epsilon}_{\rm nuc}$ (C$^{12}$)")
        #plt.semilogy(rc,en_o16_acf88,label=r"$\dot{\epsilon}_{\rm nuc}$ (O$^{16}$)")
        #plt.semilogy(rc,en_ne20_acf88,label=r"$\dot{\epsilon}_{\rm nuc}$ (Ne$^{20}$)")
        #plt.semilogy(rc,en_si28_acf88,label=r"$\dot{\epsilon}_{\rm nuc}$ (Si$^{28}$)")
        #        plt.semilogy(rc,en_c12_acf88+en_o16_acf88+en_ne20_acf88+en_si28_acf88,label='total',color='k')

        #plt.semilogy(rc,enuc1,color='m',linestyle='--',label='enuc1')
        # plt.semilogy(rc,enuc2,color='r',linestyle='--',label='-neut code')
        #plt.semilogy(rc,enuc1-enuc2,color='b',linestyle='--',label='enuc1-enuc2')
        #print(enuc1)

        # convective boundary markers
        #plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        #plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        plt.legend(loc=1, prop={'size': 14}, ncol=1)

        plt.ylabel(r"$\dot{\epsilon}_{\rm nuc}$ (erg g$^{-1}$ s$^{-1}$)")
        #plt.xlabel('r ($10^8$ cm)')
        plt.xlabel('r (cm)')

        #axvline(x=5.65, color='k', linewidth=1)
        plt.show(block=False)
        #        text(9.,1.e6,r"ob",fontsize=42,color='k')

        plt.savefig('RESULTS/oburn14_nuclear_energy_gen_inst.png')
        plt.savefig('RESULTS/oburn14_nuclear_energy_gen_inst.eps')

    def PlotNucEnergyGen2(self, xbl, xbr):
        """Plot nuclear reaction timescales"""

        nx = np.asarray(self.data['nx'])
        rc = np.asarray(self.data['xzn0'])
        xznl = np.asarray(self.data['xznl'])
        xznr = np.asarray(self.data['xznr'])

        tt = self.data['tt']
        dd = self.data['dd']

        # for 25 element network
        #xhe4 = self.data['x0003']
        #xc12 = self.data['x0004']
        #xo16 = self.data['x0005']
        #xne20 = self.data['x0006']
        #xsi28 = self.data['x0009']

        # for 14 elements network
        xhe4 = self.data['x0003']
        xc12 = self.data['x0004']
        xo16 = self.data['x0005']
        xne20 = self.data['x0006']
        xsi28 = self.data['x0009']

        bconv = 4.2e8
        #tconv_list = [7.e8,8.e8,9.e8,9.5e8]
        tconv_list = np.linspace(5.5e8,9.7e8,num=100)

        en_c12tot_list, en_o16tot_list, en_ne20tot_list, en_si28tot_list = [],[],[],[]

        for tconv in tconv_list:

            Vol = 4. / 3. * np.pi * (xznr ** 3 - xznl ** 3)
            ind = np.where((rc > bconv) & (rc < tconv))[0]
            M = (dd * Vol)[ind].sum()

            Mc12 = (dd * xc12 * Vol)[ind].sum()
            Mo16 = (dd * xo16 * Vol)[ind].sum()
            Mne20 = (dd * xne20 * Vol)[ind].sum()
            Msi28 = (dd * xsi28 * Vol)[ind].sum()

            xc12inst = Mc12/M
            xo16inst = Mo16/M
            xne20inst = Mne20/M
            xsi28inst = Msi28/M

            xc12[ind] = xc12inst
            xo16[ind] = xo16inst
            xne20[ind] = xne20inst
            xsi28[ind] = xsi28inst

            enuc1 = self.data['enuc1']
            enuc2 = np.abs(self.data['enuc2'])

            #       ne20 > he4 + o16 (photo-d: resonance)
            t9 = tt / 1.e9
            # + 4.e-2*self.eht_tt[:,tt]/1.e9

            # rate coefficients from netsu (source cf88)

            cl = self.GETRATEcoeff(reaction='ne20_to_he4_o16_rv')
            rate_ne20_alpha_gamma = np.exp(
                cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                    5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

            #       he4 + ne20 > mg24
            cl = self.GETRATEcoeff(reaction='he4_plus_ne20_to_mg24_r')
            rate_ne20_alpha_gamma_code = np.exp(
                cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                    5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

            #       o16 + o16 > p + p31 (resonance)
            #        xo16 = self.fht_xo16[:,tt]
            cl = self.GETRATEcoeff(reaction='o16_plus_o16_to_p_p31_r')
            rate_o16_o16_pchannel_r = np.exp(
                cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                    5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

            #       o16 + o16 > he4 + si28 (resonance)
            #        xo16 = self.fht_xo16[:,tt]
            cl = self.GETRATEcoeff(reaction='o16_plus_o16_to_he4_si28_r')
            rate_o16_o16_achannel_r = np.exp(
                cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                    5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

            #       c12 + c12 > p + na23 (resonance)
            cl = self.GETRATEcoeff(reaction='c12_plus_c12_to_p_na23_r')
            rate_c12_c12_pchannel_r = np.exp(
                cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                    5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

            #       c12 + c12 > he4 + ne20 (resonance)
            cl = self.GETRATEcoeff(reaction='c12_plus_c12_to_he4_ne20_r')
            rate_c12_c12_achannel_r = np.exp(
                cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                    5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

            # ANALYTIC EXPRESSIONS Caughlan & Fowler 1988

            t9a = t9 / (1. + 0.0396 * t9)
            c_tmp1 = (4.27e26) * (t9a ** (5. / 6.))
            c_tmp2 = t9 ** (3. / 2.)
            c_e_tmp1 = -84.165 / (t9a ** (1. / 3.))
            c_e_tmp2 = -(2.12e-3) * (t9 ** 3.)

            rate_c12_c12 = c_tmp1 / c_tmp2 * (np.exp(c_e_tmp1 + c_e_tmp2))

            o_tmp1 = 7.1e36 / (t9 ** (2. / 3.))
            o_c_tmp1 = -135.93 / (t9 ** (1. / 3.))
            o_c_tmp2 = -0.629 * (t9 ** (2. / 3.))
            o_c_tmp3 = -0.445 * (t9 ** (4. / 3.))
            o_c_tmp4 = +0.0103 * (t9 ** 2.)

            rate_o16_o16 = o_tmp1 * np.exp(o_c_tmp1 + o_c_tmp2 + o_c_tmp3 + o_c_tmp4)

            n_tmp1 = 4.11e11 / (t9 ** (2. / 3.))
            n_e_tmp1 = -46.766 / (t9 ** (1. / 3.)) - (t9 / 2.219) ** 2.
            n_tmp2 = 1. + 0.009 * (t9 ** (1. / 3.)) + 0.882 * (t9 ** (2. / 3.)) + 0.055 * t9 + 0.749 * (
                        t9 ** (4. / 3.)) + 0.119 * (t9 ** (5. / 3.))
            n_tmp3 = 5.27e3 / (t9 ** (3. / 2.))
            n_e_tmp3 = -15.869 / t9

            n_tmp4 = 6.51e3 * (t9 ** (1. / 2.))
            n_e_tmp4 = -16.223 / t9

            rate_alpha_gamma_cf88 = n_tmp1 * np.exp(n_e_tmp1) * n_tmp2 + n_tmp3 * np.exp(n_e_tmp3) + n_tmp4 * np.exp(
                n_e_tmp4)

            c1_c12 = 4.8e18
            c1_o16 = 8.e18
            c1_ne20 = 2.5e29
            c1_si28 = 1.8e28

            yc12sq = (xc12 / 12.) ** 2.
            yo16sq = (xo16 / 16.) ** 2.
            yne20sq = (xne20 / 20.) ** 2.

            yo16 = xo16 / 16.

            lag = (3.e-3) * (t9 ** (10.5))
            lox = (2.8e-12) * (t9 / 2.) ** 33.
            lca = (4.e-11) * (t9 ** 29.)
            lsi = 120. * (t9 / 3.5) ** 5.

            en_c12 = c1_c12 * yc12sq * dd * (rate_c12_c12_achannel_r + rate_c12_c12_pchannel_r)
            en_c12_acf88 = c1_c12 * yc12sq * dd * (rate_c12_c12)
            en_o16 = c1_o16 * yo16sq * dd * (rate_o16_o16_achannel_r + rate_o16_o16_pchannel_r)
            en_o16_acf88 = c1_o16 * yo16sq * dd * (rate_o16_o16)
            en_ne20 = c1_ne20 * (t9 ** (3. / 2.)) * (yne20sq / yo16) * rate_ne20_alpha_gamma_code * np.exp(-54.89 / t9)
            en_ne20_acf88 = c1_ne20 * (t9 ** (3. / 2.)) * (yne20sq / yo16) * rate_ne20_alpha_gamma * np.exp(-54.89 / t9)
            en_ne20_hw = c1_ne20 * (t9 ** (3. / 2.)) * (yne20sq / yo16) * lag * np.exp(-54.89 / t9)
            #        en_ne20_ini = c1_ne20*(t9**(3./2.))*(yne20sq_ini/yo16)*rate_ne20_alpha_gamma_code*np.exp(-54.89/t9)
            en_ne20_lag = c1_ne20 * (t9 ** (3. / 2.)) * (yne20sq / yo16) * lag * np.exp(-54.89 / t9)
            en_si28 = c1_si28 * (t9 ** 3. / 2.) * xsi28 * (np.exp(-142.07 / t9)) * rate_ne20_alpha_gamma_code
            en_si28_acf88 = c1_si28 * (t9 ** 3. / 2.) * xsi28 * (np.exp(-142.07 / t9)) * rate_ne20_alpha_gamma

            en_c12tot = (en_c12 * dd * Vol)[ind].sum()
            en_o16tot = (en_o16 * dd * Vol)[ind].sum()
            en_ne20tot = (en_ne20 * dd * Vol)[ind].sum()
            en_si28tot = (en_si28 * dd * Vol)[ind].sum()

            en_c12tot_list.append(en_c12tot)
            en_o16tot_list.append(en_o16tot)
            en_ne20tot_list.append(en_ne20tot)
            en_si28tot_list.append(en_si28tot)

            #print('Total Enuc c12 burn:' + str(en_c12tot))
            #print('Total Enuc o16 burn:' + str(en_o16tot))
            #print('Total Enuc ne20 burn:' + str(en_ne20tot))
            #print('Total Enuc si28 burn:' + str(en_si28tot))


        print(en_c12tot_list[-1]) # 7.59e42
        print(en_o16tot_list[-1]) # 2.25e46
        print(en_ne20tot_list[-1]) # 5.67e43
        print(en_si28tot_list[-1]) # 4.68e37

        plt.figure(figsize=(7, 6))

        xbl = tconv_list[0]
        xbr = tconv_list[-1]

        lb = 1.e37
        ub = 1.e50

        plt.axis([xbl, xbr, lb, ub])


        plt.title(r'total enuc in convection zone inst.mix')
        plt.semilogy(tconv_list, en_c12tot_list, label=r"$\dot{\epsilon}_{\rm nuc}$ (C$^{12}$)",color='r')
        plt.semilogy(tconv_list, en_o16tot_list, label=r"$\dot{\epsilon}_{\rm nuc}$ (O$^{16}$)",color='g')
        plt.semilogy(tconv_list, en_ne20tot_list, label=r"$\dot{\epsilon}_{\rm nuc}$ (Ne$^{20}$)",color='b')
        plt.semilogy(tconv_list, en_si28tot_list, label=r"$\dot{\epsilon}_{\rm nuc}$ (Si$^{28}$)",color='m')

        # convective boundary markers
        #plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        #plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # from 3D profiles
        #print(en_c12tot_list[-1]) # 7.59e42
        #print(en_o16tot_list[-1]) # 2.25e46
        #print(en_ne20tot_list[-1]) # 5.67e43
        #print(en_si28tot_list[-1]) # 4.68e37

        plt.axhline(7.59e42, linestyle='--', linewidth=0.7, color='r',label=r'c12 3D')
        plt.axhline(2.25e46, linestyle='--', linewidth=0.7, color='g',label=r'o16 3D')
        plt.axhline(5.67e43, linestyle='--', linewidth=0.7, color='b',label=r'ne20 3D')
        plt.axhline(4.68e37, linestyle='--', linewidth=0.7, color='m',label=r'si28 3D')

        plt.legend(loc=4, prop={'size': 14}, ncol=2)

        plt.ylabel(r"ergs s$^{-1}$")
        plt.xlabel('position of outer cnvz bndry (cm)')

        #axvline(x=5.65, color='k', linewidth=1)
        plt.show(block=False)


        plt.savefig('RESULTS/oburn14_nuclear_energy_gen3.png')
        plt.savefig('RESULTS/oburn14_nuclear_energy_gen3.eps')


    def plot_check_heq1(self):
        xzn0 = np.asarray(self.data['xzn0'])
        xznl = np.asarray(self.data['xznl'])
        xznr = np.asarray(self.data['xznr'])
        nx = np.asarray(self.data['nx'])

        press = np.asarray(self.data['pp'])
        dd = np.asarray(self.data['dd'])
        gg = np.asarray(self.data['gg'])

        fig, ax1 = plt.subplots(figsize=(7, 6))

        # ax1.axis([xbl,xbr,np.min(to_plt1[idxl:idxr]),np.max(to_plt1[idxl:idxr])])
        # ax1.plot(xzn0,f_1*f_2,color='b',label = 'rho g')
        # ax1.plot(xzn0,self.Grad(f_3,rr),color='r',label = 'grad pp')

        ax1.plot(xzn0, press, color='b', label='press')

        pp = np.zeros(nx)
        pp[nx - 1] = press[nx - 1]

        for i in range(nx - 2, -1, -1):
            pp[i] = pp[i + 1] - dd[i] * gg[i] * (xznr[i] - xznl[i])
            # print(i,pp[i])

        # for i in range(0,nx-1):
        #    pp[i] = - f_1[i]*f_2[i]*(xznr[i]-xznl[i])
        #    print(i,pp[i],f_3[i],f_2[i],f_1[i],xznr[i],xznl[i],xzn0[i])

        ax1.plot(xzn0, pp, color='r', label='pp hydrostatic')

        ax1.set_xlabel(r'x')
        ax1.set_ylabel(r'pp')
        ax1.legend(loc=1, prop={'size': 18})

        savefig('RESULTS/hse1.png')

        plt.show(block=False)

    def plot_check_heq2(self, xbl, xbr):
        xzn0 = np.asarray(self.data['xzn0'])
        xznl = np.asarray(self.data['xznl'])
        xznr = np.asarray(self.data['xznr'])
        nx = np.asarray(self.data['nx'])

        press = np.asarray(self.data['pp'])
        dd = np.asarray(self.data['dd'])
        ddgg = np.asarray(self.data['ddgg'])
        gg = np.asarray(self.data['gg'])
        mm = np.asarray(self.data['mm'])

        fig, ax1 = plt.subplots(figsize=(7, 6))

        # ax1.axis([xbl,xbr,-0.1,0.1])
        # ax1.plot(xzn0,f_1*f_2,color='b',label = 'rho g')
        # ax1.plot(xzn0,self.Grad(f_3,rr),color='r',label = 'grad pp')

        # ax1.semilogy(xzn0,press,color='b',label = 'press')

        pp = np.zeros(nx)
        ppgrad = np.zeros(nx)
        pp[nx - 1] = press[nx - 1]

        ggtest = np.zeros(nx)
        kappa = 6.673e-8
        for i in range(0, nx):
            ggtest[i] = -kappa * mm[i] / (xznl[i] ** 2.)

        for i in range(nx - 2, -1, -1):
            # pp[i] = pp[i+1] - dd[i]*ggtest[i]*(xznr[i]-xznl[i])
            pp[i] = pp[i + 1] - ddgg[i] * (xznr[i] - xznl[i])
            # print(i,pp[i],dd[i],gg[i],(xznr[i]-xznl[i]))
            # print(i,gg[i],dd[i],pp[i],(xznr[i]-xznl[i])))

        for i in range(nx - 1):
            ppgrad[i] = (press[i + 1] - press[i]) / (xzn0[i + 1] - xzn0[i])

        # pp[0] = pp[1]

        # print(nx)
        # for i in range(0,nx):
        #    pp[i] = - f_1[i]*f_2[i]*(xznr[i]-xznl[i])
        #    print(i,(press[i]-pp[i])/pp[i],press[i],pp[i])
        #    print(i,grav[i],dd[i],pp[i],(xznr[i]-xznl[i]),mm[i],xzn0[i])            

        # ax1.plot(xzn0,(press-pp)/pp,color='r',label = 'press-hse/hse')
        # ax1.semilogy(xzn0,pp,color='r',label = 'hse')

        #        ax1.plot(xzn0,self.Grad(pp,xzn0),color='r',label='grad pp')

        ax1.plot(xzn0, ppgrad, color='r', label='grad pp')
        ax1.plot(xzn0, ddgg, color='b', label='dd gg')

        rd = (press - pp) / pp
        # print(np.sum(rd))

        ax1.set_xlabel(r'r')
        ax1.set_ylabel(r'grad P')
        ax1.legend(loc=1, prop={'size': 18})

        savefig('RESULTS/hse2.png')

        plt.show(block=False)

    def plot_nablas(self, xbl, xbr):
        xzn0 = np.asarray(self.data['xzn0'])/1.e8
        nx = np.asarray(self.data['nx'])

        pp = np.asarray(self.data['pp'])
        tt = np.asarray(self.data['tt'])
        mu = np.asarray(self.data['abar'])
        chim = np.asarray(self.data['chim'])
        chit = np.asarray(self.data['chit'])
        gamma2 = np.asarray(self.data['gamma2'])

        lntt = np.log(tt)
        lnpp = np.log(pp)
        lnmu = np.log(mu)

        nabla = self.deriv(lntt, lnpp)
        nabla_ad = (gamma2 - 1.) / gamma2
        nabla_mu = (chim / chit) * self.deriv(lnmu, lnpp)

        fig, ax1 = plt.subplots(figsize=(7, 6))
        idxl, idxr = self.idx_bndry(xbl, xbr)

        # ax1.axis([xbl,xbr,np.min(to_plt1[idxl:idxr]),np.max(to_plt1[idxl:idxr])])
        ax1.axis([xbl, xbr, -1., 1.])

        ax1.plot(xzn0, nabla, color='r', label=r"$\nabla$")
        ax1.plot(xzn0, nabla_ad, color='g', label=r"$\nabla_{ad}$")
        ax1.plot(xzn0, nabla_mu, color='b', label=r"$\nabla_{\mu}$")

        ax1.set_xlabel('r ($10^8$ cm)')
        ax1.set_ylabel("nabla")
        ax1.legend(loc=1, prop={'size': 18})

        plt.show(block=False)

    def plot_mm(self, xbl, xbr):
        xzn0 = np.asarray(self.data['xzn0'])
        xznl = np.asarray(self.data['xznl'])
        xznr = np.asarray(self.data['xznr'])
        nx = np.asarray(self.data['nx'])

        mm = np.asarray(self.data['mm'])
        dd = np.asarray(self.data['dd'])

        fig, ax1 = plt.subplots(figsize=(7, 6))
        idxl, idxr = self.idx_bndry(xbl, xbr)

        # ax1.axis([xbl,xbr,np.min(to_plt1[idxl:idxr]),np.max(to_plt1[idxl:idxr])])
        # ax1.axis([xbl,xbr,-1.,1.])

        pmass = 2.106e33
        mmint = np.zeros(nx)
        mmint[0] = pmass

        for i in range(1, nx):
            mmint[i] = mmint[i - 1] + 4. * np.pi * (xzn0[i] ** 2) * dd[i] * (xznr[i] - xznl[i])

        ax1.plot(xzn0, mm, color='r', label=r"$mm$")
        # ax1.plot(xzn0,(4./3.)*np.pi*(xzn0**3)*dd,color='g',label = r"$V \rho$")
        ax1.plot(xzn0, mmint, color='g', linestyle='--', label=r"$mmint$")

        ax1.set_xlabel("r")
        ax1.set_ylabel("mass")
        ax1.legend(loc=1, prop={'size': 18})

        plt.show(block=False)

    def plot_dx(self, xbl, xbr):
        xzn0 = np.asarray(self.data['xzn0'])
        xznl = np.asarray(self.data['xznl'])
        xznr = np.asarray(self.data['xznr'])
        nx = np.asarray(self.data['nx'])

        fig, ax1 = plt.subplots(figsize=(7, 6))
        idxl, idxr = self.idx_bndry(xbl, xbr)
        ax1.axis([xbl, xbr, 0., 1.e8])

        dx = xznr - xznl
        ax1.plot(xzn0, dx, color='r', label=r"$dx$")

        ax1.set_xlabel("r")
        ax1.set_ylabel("dx")
        ax1.legend(loc=1, prop={'size': 18})

        plt.show(block=False)

    def plot_check_heq3(self):
        xzn0 = np.asarray(self.data['xzn0'])
        xznl = np.asarray(self.data['xznl'])
        xznr = np.asarray(self.data['xznr'])
        nx = np.asarray(self.data['nx'])

        press = np.asarray(self.data['pp'])
        dd = np.asarray(self.data['dd'])
        gg = np.asarray(self.data['gg'])

        fig, ax1 = plt.subplots(figsize=(7, 6))

        # ax1.axis([4.e8, 1.2e9, 1.e-4, 0.1])
        # ax1.axis([4.e8, 6.e9, 1.e-8, 0.9])
        ax1.axis([3.0e8, 1.e9, 1.e-8, 1.])
        #ax1.axis([3.4e8, 4.04e8, 1.e-8, 1.])

        plt.title(r'HSE deviation (evolved)')

        pp = np.zeros(nx)
        pp[nx - 1] = press[nx - 1]

        for i in range(nx - 2, -1, -1):
            pp[i] = pp[i + 1] - dd[i] * gg[i] * (xznr[i] - xznl[i])
            print(i,pp[i],dd[i],gg[i])

        ax1.semilogy(xzn0, np.abs((pp - press)) / press, color='r', label='(pp hydrostatic - press)/press')

        print(pp)
        print('*******************')
        print(press)

        ax1.set_xlabel(r'x')
        ax1.set_ylabel(r'(delta pp)/pp')
        ax1.legend(loc=1, prop={'size': 18})

        savefig('RESULTS/hse3_ccptwo_256cubed_evolved.png')

        plt.show(block=False)

    def plot_check_ux(self, xbl, xbr):
        ux = np.asarray(self.data['ux'])
        gg = np.asarray(self.data['gg'])
        xzn0 = np.asarray(self.data['xzn0'])

        fig, ax1 = plt.subplots(figsize=(7, 6))
        idxl, idxr = self.idx_bndry(xbl, xbr)

        # ax1.axis([xbl,xbr,np.min(to_plt1[idxl:idxr]),np.max(to_plt1[idxl:idxr])])
        ax1.axis([xbl, xbr, -1.e7, 1.e7])

        ax1.plot(xzn0, ux, color='r', label='ux')
        ax1.plot(xzn0, 0.5e7 + 1.e7 * gg / (max(np.abs(gg))), color='b', label='gg')

        ax1.set_xlabel(r'r')
        ax1.set_ylabel(r'ux')
        ax1.legend(loc=1, prop={'size': 18})

        savefig('RESULTS/uxcheck.png')

        plt.show(block=False)

    def PlotTrippleAlphaNucEnergyGen(self, xbl, xbr):
        """Plot nuclear reaction timescales"""

        rc = np.asarray(self.data['xzn0'])
        tt = self.data['tt']
        dd = self.data['dd']

        xhe4 = self.data['x0003']

        #  enuc = self.data['enuc1']+self.data['enuc2']
        enuc1 = self.data['enuc1']
        enuc2 = np.abs(self.data['enuc2'])


        # Clayton, Principles of Stellar Evolution and Nucleosynthesis, page 414, eq.5-105
        f = 1. # screening factor
        tt8 = tt/1.e8
        #epsilon3alpha = 4.4e-8*(dd**2.)*(xhe4**3.)*((tt/1.e8)**40.)*f  # this is around 1e8 K
        epsilon3alpha = 3.9e11*((dd**2.)*(xhe4**3.)/(tt8**3.))*np.exp(-42.94/tt8)
        print(epsilon3alpha)

        plt.figure(figsize=(7, 6))


        lb = 1.e-10
        ub = 1.e14
        plt.axis([xbl, xbr, lb, ub])

        plt.semilogy(rc,epsilon3alpha,color='r',label=r"$\epsilon(3\alpha) \sim 4.4 \times 10^{-8} \rho^2 X(He^4)^3 T_8^{40}$")
        plt.semilogy(rc,enuc1,color='m',linestyle='--',label='nuc code (boost 100x)')

        setxlabel = r'r (cm)'
        # setylabel = r'log $\overline{\varepsilon_{enuc}}$ (erg g$^{-1}$ s$^{-1}$)'
        setylabel = r'$\overline{\varepsilon_{enuc}}$ (erg g$^{-1}$ s$^{-1}$)'
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        plt.legend(loc=3, prop={'size': 14})

        plt.show(block=False)

    def plotMonstarIni(self):

        # fmonstar = 'C:\\Users\\mmocak\\Desktop\\GITDEV\\ransX\\DATA_D\\INIMODEL\\imodel.monstar'
        fmonstar = 'DATA_D\\INIMODEL\\imodel.monstar'
        tdata = open(fmonstar, 'r')

        header_line1 = tdata.readline().split()
        header_line2 = tdata.readline().split()
        header_line3 = tdata.readline().split()

        t_line1 = tdata.readline().split()

        rrm = []
        epspp = []
        epscno = []
        epshe = []
        prot,he4,he3 = [],[],[]
        dd = []
        uconv = []
        for i in range(1996):
            rl = tdata.readline().split()
            rrm.append(rl[2])
            dd.append(rl[6])
            epspp.append(rl[7])
            epscno.append(rl[8])
            epshe.append(rl[9])
            prot.append(rl[11])
            he4.append(rl[12])
            he3.append(rl[13])
            uconv.append(rl[18])

            # rrm.append(tdata.readline().split()[2])
            # epspp.append(tdata.readline().split()[7])
            # epscno.append(tdata.readline().split()[8])
            # epshe.append(tdata.readline().split()[9])

        tdata.close()

        # print(epspp,epscno,epshe)

        rrm = np.asarray(rrm[::-1], dtype=float)
        dd = np.asarray(dd[::-1], dtype=float)
        epspp = np.asarray(epspp[::-1], dtype=float)
        epscno = np.asarray(epscno[::-1], dtype=float)
        epshe = np.asarray(epshe[::-1], dtype=float)
        prot = np.asarray(prot[::-1], dtype=float)
        he4 = np.asarray(he4[::-1], dtype=float)
        he3 = np.asarray(he3[::-1], dtype=float)

        rrm = 10**rrm
        dd = 10**dd
        ldhe = 1.e9
        enuc = epspp + epscno + epshe  # this should be integral over convection zone
        ulim = (enuc*ldhe/dd)*(1./3.) # wrong

        plt.figure(figsize=(7, 6))

        # xbl = 5.e8
        # xbr = 6.e9
        xbl = 3.e9
        xbr = 6.e9
        lb = 1.e2
        ub = 1.e14

        rr = np.asarray(rrm)
        xlm = np.abs(rr - xbl)
        xrm = np.abs(rr - xbr)
        ia = int(np.where(xlm == xlm.min())[0])
        ib = int(np.where(xrm == xrm.min())[0])

        #print(ulim[ia:ib],enuc[ia:ib],dd[ia:ib])
        #print(epspp[ia:ib],epscno[ia:ib],epshe[ia:ib])
        print(ia,ib)

        plt.axis([xbl, xbr, lb, ub])

        plt.title='he-burning shell'
        plt.semilogy(rrm[ia:ib],ulim[ia:ib],color='m',linestyle='--',label='ulim')
        plt.semilogy(rrm[ia:ib],uconv[ia:ib],color='r',label='uconv')
        #plt.semilogy(epscno)

        setxlabel = r'r (cm)'
        # setylabel = r'log $\overline{\varepsilon_{enuc}}$ (erg g$^{-1}$ s$^{-1}$)'
        setylabel = r'$ulim$ (cm s$^{-1}$)'
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        plt.legend(loc=3, prop={'size': 14})
        plt.show(block=False)


    def idx_bndry(self, xbl, xbr):
        rr = np.asarray(self.data['xzn0'])
        xlm = np.abs(rr - xbl)
        xrm = np.abs(rr - xbr)
        idxl = int(np.where(xlm == xlm.min())[0])
        idxr = int(np.where(xrm == xrm.min())[0])
        return idxl, idxr

