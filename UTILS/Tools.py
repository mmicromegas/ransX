import numpy as np


# class for tools

class Tools:

    def __init__(self):
        pass

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
