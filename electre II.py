from Ele import *
import numpy as np

header_dict, data_dict = readtxt('e2.csv')
pprint(header_dict)
pprint(data_dict)

concordance, discordance = concordance_discordance(data_dict, header_dict['pesos'])

print(concordance)
print(discordance)

swdm = strong_weak_dominance(concordance, discordance, .65, .75, .85, .25, .5)

swdm[2,1]='-'

print(swdm)
rd = destilacao_descendente(np.copy(swdm))
print(swdm)
ra = destilacao_ascendente(np.copy(swdm))
# print(swdm)
ordenacoes(rd, ra)

