from UTILS.SetAxisLimit import SetAxisLimit
import numpy as np
from scipy import integrate


# class for tools

class Tools(SetAxisLimit, object):

    def __init__(self):
        super(Tools, self).__init__()

    def customLoad(self, fn):
        return np.load(fn, allow_pickle=True, encoding='latin1')

    def getRAdata(self, ransdat, q):
        quantity = np.asarray(ransdat.item().get(q))
        return quantity

    def thirdOrder(self, eht, intc, a, b, c):

        ## DEV IN PROGRESS ##

        ab = a + b
        bc = b + c
        ac = a + c
        abc = a + b + c

        if ab in ['uyux']:
            ab = 'uxuy'
        elif ab in ['uzux']:
            ab = 'uxuz'

        if bc in ['uyux']:
            bc = 'uxuy'
        elif ab in ['uzux']:
            bc = 'uxuz'

        if ac in ['uyux']:
            ac = 'uxuy'
        elif ac in ['uzux']:
            ac = 'uxuz'

        eht_a = self.getRAdata(eht, a)[intc]
        eht_b = self.getRAdata(eht, b)[intc]
        eht_c = self.getRAdata(eht, c)[intc]

        print(ab, bc, ac, abc)

        eht_ab = self.getRAdata(eht, ab)[intc]
        eht_bc = self.getRAdata(eht, bc)[intc]
        eht_ac = self.getRAdata(eht, ac)[intc]

        eht_abc = self.getRAdata(eht, abc)[intc]

        thirdOrderMoment = eht_abc - eht_a * eht_bc - eht_b * eht_ac - eht_c * eht_ab + eht_a * eht_b * eht_c

        return thirdOrderMoment

    def calcIntegralBudget(self, terms, xbl, xbr, nx, xzn0, yzn0, zzn0, nsdim, plabel, laxis, ig):

        # hack for the ccp setup getting rid of bndry noise
        if plabel == 'ccptwo':
            fct1 = 2.e-1
            fct2 = 1.e-1
            xbl = xbl + fct1*xbl
            xbr = xbr - fct2*xbl

        #if plabel == 'ccptwo':
        #    xbl = 4.5e8
        #    xbr = 11.5e8

        # calculate INDICES for grid boundaries
        if laxis == 1 or laxis == 2:
            idxl, idxr = self.idx_bndry(xbl, xbr, xzn0)
        else:
            idxl = 0
            idxr = nx - 1

        ints = []
        for term in terms:
            term_sel = term[idxl:idxr]

            rc = xzn0[idxl:idxr]

            # handle geometry
            Sr = 0.
            if ig == 1 and nsdim == 3:
                Sr = (yzn0[-1] - yzn0[0]) * (zzn0[-1] - zzn0[0])
            elif ig == 1 and nsdim == 2:
                Sr = (yzn0[-1] - yzn0[0]) * (yzn0[-1] - yzn0[0])
            elif ig == 2:
                Sr = 4. * np.pi * rc ** 2

            int_term = integrate.simps(term_sel * Sr, rc)
            ints.append(int_term)

        return ints
