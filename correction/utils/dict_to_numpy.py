# --*- coding:utf-8 -*--
import numpy as np
import pandas as pd
# 将字典转换为numpy



test_dict={'v': {}, 'z': {'h': 0.6994784876140808, 'u': 0.06584093872229466, "a'": 0.009126466753585397, 'a': 0.05997392438070404, "e'": 0.02346805736636245, 'e': 0.018252933507170794, "i'": 0.05410691003911343, 'o': 0.044980443285528034, "u'": 0.024771838331160364}, 'u': {'a': 0.31958365458750965, "n'": 0.18851195065535853, "o'": 0.19390902081727063, "i'": 0.18427139552814187, "e'": 0.07864302235929067, "a'": 0.03508095605242868}, '^': {'b': 0.04026375678356772, 'p': 0.031160646554239366, "m'": 0.00017505981210246834, 'm': 0.037462799789928225, 'f': 0.027776156853591645, 'd': 0.04358989321351462, 't': 0.043881659567018734, "n'": 0.00011670654140164556, 'n': 0.022641069031919238, 'l': 0.0735251210830367, 'g': 0.035945614751706836, 'k': 0.026142265273968606, 'h': 0.04819980159887962, 'j': 0.07317500145883177, 'q': 0.050825698780416644, 'x': 0.0673396743887495, 'z': 0.08951391725506215, 'c': 0.05601913987278987, 's': 0.0637801248759993, 'r': 0.014996790570111454, 'y': 0.09873373402579215, 'w': 0.03063546711793196, "a'": 0.00040847289490575947, 'a': 0.008052751356713544, "e'": 0.004960028009569936, 'e': 0.002976016805741962, "o'": 0.0002917663535041139, 'o': 0.001575538308922215, '$': 0.005835327070082278}, 'f': {"a'": 0.04831932773109244, 'a': 0.20168067226890757, 'e': 0.35714285714285715, 'i': 0.0021008403361344537, "o'": 0.0063025210084033615, 'o': 0.01050420168067227, "u'": 0.3739495798319328}, '$': {}, 'w': {"a'": 0.04380952380952381, 'a': 0.18666666666666668, 'e': 0.4438095238095238, "o'": 0.0838095238095238, "u'": 0.2419047619047619}, 'd': {'u': 0.1646586345381526, "a'": 0.041499330655957165, 'a': 0.250334672021419, "e'": 0.012048192771084338, 'e': 0.0321285140562249, "i'": 0.11780455153949129, 'i': 0.2101740294511379, 'o': 0.09906291834002677, "u'": 0.07228915662650602}, 'i': {'a': 0.4410029498525074, "e'": 0.12064896755162242, "n'": 0.13185840707964602, 'n': 0.17315634218289086, "u'": 0.06991150442477877, "a'": 0.044837758112094395, 'o': 0.018584070796460177}, 'n': {"g'": 0.8741030658838878, 'u': 0.007827788649706457, "v'": 0.0029354207436399216, "a'": 0.007175472928897586, 'a': 0.02837573385518591, "e'": 0.00228310502283105, 'e': 0.003913894324853229, "i'": 0.01989562948467058, 'i': 0.04337899543378995, 'o': 0.006523157208088715, "u'": 0.003587736464448793}, 'c': {'h': 0.6864583333333333, 'u': 0.08645833333333333, "a'": 0.005208333333333333, 'a': 0.08229166666666667, "e'": 0.014583333333333334, 'e': 0.0125, "i'": 0.04375, 'o': 0.046875, "u'": 0.021875}, 'o': {"u'": 0.4613379669852302, 'n': 0.5386620330147698}, 'l': {'u': 0.09841269841269841, "v'": 0.04523809523809524, "a'": 0.02142857142857143, 'a': 0.1373015873015873, "e'": 0.012698412698412698, 'e': 0.05476190476190476, "i'": 0.15079365079365079, 'i': 0.30952380952380953, "o'": 0.0007936507936507937, 'o': 0.06984126984126984, "u'": 0.0992063492063492}, 'b': {"a'": 0.06231884057971015, 'a': 0.20434782608695654, 'e': 0.12318840579710146, "i'": 0.18405797101449275, 'i': 0.26521739130434785, "o'": 0.11014492753623188, "u'": 0.050724637681159424}, 'j': {'u': 0.11881977671451356, "i'": 0.19856459330143542, 'i': 0.5829346092503987, "u'": 0.09968102073365231}, 'g': {'u': 0.36688311688311687, "a'": 0.017857142857142856, 'a': 0.21103896103896103, "e'": 0.09902597402597403, 'e': 0.056818181818181816, 'o': 0.11688311688311688, "u'": 0.1314935064935065}, 'm': {"a'": 0.059190031152647975, 'a': 0.21339563862928349, "e'": 0.006230529595015576, 'e': 0.18847352024922118, "i'": 0.10903426791277258, 'i': 0.21806853582554517, "o'": 0.10747663551401869, 'o': 0.021806853582554516, "u'": 0.0763239875389408}, 's': {'h': 0.6184812442817932, 'u': 0.10978956999085086, "a'": 0.01463860933211345, 'a': 0.056724611161939616, "e'": 0.022872827081427266, 'e': 0.0027447392497712718, "i'": 0.06312900274473925, 'o': 0.06221408966148216, "u'": 0.04940530649588289}, 'x': {'u': 0.1655112651646447, "i'": 0.17764298093587522, 'i': 0.5693240901213171, "u'": 0.08752166377816291}, 't': {'u': 0.15558510638297873, "a'": 0.06382978723404255, 'a': 0.2872340425531915, "e'": 0.010638297872340425, 'e': 0.02127659574468085, "i'": 0.09707446808510638, 'i': 0.2047872340425532, 'o': 0.09042553191489362, "u'": 0.06914893617021277}, 'a': {"i'": 0.09064775776159448, "n'": 0.48179379072441547, 'n': 0.1910693752395554, "o'": 0.23648907627443466}, 'q': {'u': 0.12055109070034443, "i'": 0.20780711825487944, 'i': 0.5648679678530425, "u'": 0.10677382319173363}, 'k': {'u': 0.453125, "a'": 0.024553571428571428, 'a': 0.19419642857142858, "e'": 0.15178571428571427, 'e': 0.05357142857142857, 'o': 0.06473214285714286, "u'": 0.05803571428571429}, 'r': {'u': 0.20233463035019456, 'a': 0.1828793774319066, "e'": 0.01556420233463035, 'e': 0.1867704280155642, "i'": 0.011673151750972763, 'o': 0.24124513618677043, "u'": 0.15953307392996108}, 'y': {'u': 0.13829787234042554, "a'": 0.041371158392434985, 'a': 0.22340425531914893, "e'": 0.03664302600472813, "i'": 0.18439716312056736, 'i': 0.13002364066193853, "o'": 0.0017730496453900709, 'o': 0.08983451536643026, "u'": 0.15425531914893617}, 'h': {'n': 0.00030921459492888067, 'u': 0.22943722943722944, "a'": 0.04452690166975881, 'a': 0.1982065553494125, "e'": 0.05442176870748299, 'e': 0.12894248608534323, 'o': 0.08843537414965986, "u'": 0.11750154607297464, "i'": 0.13790970933828076, "m'": 0.00030921459492888067}, 'e': {"n'": 0.3277661795407098, 'n': 0.3291579679888657, "i'": 0.313848295059151, "r'": 0.029227557411273485}, 'p': {"a'": 0.035580524344569285, 'a': 0.19662921348314608, 'e': 0.16666666666666666, "i'": 0.16104868913857678, 'i': 0.24719101123595505, "o'": 0.08239700374531835, 'o': 0.016853932584269662, "u'": 0.09363295880149813}, "a'": {'b': 0.04026375678356772, 'p': 0.031160646554239366, "m'": 0.00017505981210246834, 'm': 0.037462799789928225, 'f': 0.027776156853591645, 'd': 0.04358989321351462, 't': 0.043881659567018734, "n'": 0.00011670654140164556, 'n': 0.022641069031919238, 'l': 0.0735251210830367, 'g': 0.035945614751706836, 'k': 0.026142265273968606, 'h': 0.04819980159887962, 'j': 0.07317500145883177, 'q': 0.050825698780416644, 'x': 0.0673396743887495, 'z': 0.08951391725506215, 'c': 0.05601913987278987, 's': 0.0637801248759993, 'r': 0.014996790570111454, 'y': 0.09873373402579215, 'w': 0.03063546711793196, "a'": 0.00040847289490575947, 'a': 0.008052751356713544, "e'": 0.004960028009569936, 'e': 0.002976016805741962, "o'": 0.0002917663535041139, 'o': 0.001575538308922215, '$': 0.005835327070082278}, "i'": {'b': 0.04026375678356772, 'p': 0.031160646554239366, "m'": 0.00017505981210246834, 'm': 0.037462799789928225, 'f': 0.027776156853591645, 'd': 0.04358989321351462, 't': 0.043881659567018734, "n'": 0.00011670654140164556, 'n': 0.022641069031919238, 'l': 0.0735251210830367, 'g': 0.035945614751706836, 'k': 0.026142265273968606, 'h': 0.04819980159887962, 'j': 0.07317500145883177, 'q': 0.050825698780416644, 'x': 0.0673396743887495, 'z': 0.08951391725506215, 'c': 0.05601913987278987, 's': 0.0637801248759993, 'r': 0.014996790570111454, 'y': 0.09873373402579215, 'w': 0.03063546711793196, "a'": 0.00040847289490575947, 'a': 0.008052751356713544, "e'": 0.004960028009569936, 'e': 0.002976016805741962, "o'": 0.0002917663535041139, 'o': 0.001575538308922215, '$': 0.005835327070082278}, "n'": {'b': 0.04026375678356772, 'p': 0.031160646554239366, "m'": 0.00017505981210246834, 'm': 0.037462799789928225, 'f': 0.027776156853591645, 'd': 0.04358989321351462, 't': 0.043881659567018734, "n'": 0.00011670654140164556, 'n': 0.022641069031919238, 'l': 0.0735251210830367, 'g': 0.035945614751706836, 'k': 0.026142265273968606, 'h': 0.04819980159887962, 'j': 0.07317500145883177, 'q': 0.050825698780416644, 'x': 0.0673396743887495, 'z': 0.08951391725506215, 'c': 0.05601913987278987, 's': 0.0637801248759993, 'r': 0.014996790570111454, 'y': 0.09873373402579215, 'w': 0.03063546711793196, "a'": 0.00040847289490575947, 'a': 0.008052751356713544, "e'": 0.004960028009569936, 'e': 0.002976016805741962, "o'": 0.0002917663535041139, 'o': 0.001575538308922215, '$': 0.005835327070082278}, "g'": {'b': 0.04026375678356772, 'p': 0.031160646554239366, "m'": 0.00017505981210246834, 'm': 0.037462799789928225, 'f': 0.027776156853591645, 'd': 0.04358989321351462, 't': 0.043881659567018734, "n'": 0.00011670654140164556, 'n': 0.022641069031919238, 'l': 0.0735251210830367, 'g': 0.035945614751706836, 'k': 0.026142265273968606, 'h': 0.04819980159887962, 'j': 0.07317500145883177, 'q': 0.050825698780416644, 'x': 0.0673396743887495, 'z': 0.08951391725506215, 'c': 0.05601913987278987, 's': 0.0637801248759993, 'r': 0.014996790570111454, 'y': 0.09873373402579215, 'w': 0.03063546711793196, "a'": 0.00040847289490575947, 'a': 0.008052751356713544, "e'": 0.004960028009569936, 'e': 0.002976016805741962, "o'": 0.0002917663535041139, 'o': 0.001575538308922215, '$': 0.005835327070082278}, "o'": {'b': 0.04026375678356772, 'p': 0.031160646554239366, "m'": 0.00017505981210246834, 'm': 0.037462799789928225, 'f': 0.027776156853591645, 'd': 0.04358989321351462, 't': 0.043881659567018734, "n'": 0.00011670654140164556, 'n': 0.022641069031919238, 'l': 0.0735251210830367, 'g': 0.035945614751706836, 'k': 0.026142265273968606, 'h': 0.04819980159887962, 'j': 0.07317500145883177, 'q': 0.050825698780416644, 'x': 0.0673396743887495, 'z': 0.08951391725506215, 'c': 0.05601913987278987, 's': 0.0637801248759993, 'r': 0.014996790570111454, 'y': 0.09873373402579215, 'w': 0.03063546711793196, "a'": 0.00040847289490575947, 'a': 0.008052751356713544, "e'": 0.004960028009569936, 'e': 0.002976016805741962, "o'": 0.0002917663535041139, 'o': 0.001575538308922215, '$': 0.005835327070082278}, "e'": {'b': 0.04026375678356772, 'p': 0.031160646554239366, "m'": 0.00017505981210246834, 'm': 0.037462799789928225, 'f': 0.027776156853591645, 'd': 0.04358989321351462, 't': 0.043881659567018734, "n'": 0.00011670654140164556, 'n': 0.022641069031919238, 'l': 0.0735251210830367, 'g': 0.035945614751706836, 'k': 0.026142265273968606, 'h': 0.04819980159887962, 'j': 0.07317500145883177, 'q': 0.050825698780416644, 'x': 0.0673396743887495, 'z': 0.08951391725506215, 'c': 0.05601913987278987, 's': 0.0637801248759993, 'r': 0.014996790570111454, 'y': 0.09873373402579215, 'w': 0.03063546711793196, "a'": 0.00040847289490575947, 'a': 0.008052751356713544, "e'": 0.004960028009569936, 'e': 0.002976016805741962, "o'": 0.0002917663535041139, 'o': 0.001575538308922215, '$': 0.005835327070082278}, "u'": {'b': 0.04026375678356772, 'p': 0.031160646554239366, "m'": 0.00017505981210246834, 'm': 0.037462799789928225, 'f': 0.027776156853591645, 'd': 0.04358989321351462, 't': 0.043881659567018734, "n'": 0.00011670654140164556, 'n': 0.022641069031919238, 'l': 0.0735251210830367, 'g': 0.035945614751706836, 'k': 0.026142265273968606, 'h': 0.04819980159887962, 'j': 0.07317500145883177, 'q': 0.050825698780416644, 'x': 0.0673396743887495, 'z': 0.08951391725506215, 'c': 0.05601913987278987, 's': 0.0637801248759993, 'r': 0.014996790570111454, 'y': 0.09873373402579215, 'w': 0.03063546711793196, "a'": 0.00040847289490575947, 'a': 0.008052751356713544, "e'": 0.004960028009569936, 'e': 0.002976016805741962, "o'": 0.0002917663535041139, 'o': 0.001575538308922215, '$': 0.005835327070082278}, "m'": {'b': 0.04026375678356772, 'p': 0.031160646554239366, "m'": 0.00017505981210246834, 'm': 0.037462799789928225, 'f': 0.027776156853591645, 'd': 0.04358989321351462, 't': 0.043881659567018734, "n'": 0.00011670654140164556, 'n': 0.022641069031919238, 'l': 0.0735251210830367, 'g': 0.035945614751706836, 'k': 0.026142265273968606, 'h': 0.04819980159887962, 'j': 0.07317500145883177, 'q': 0.050825698780416644, 'x': 0.0673396743887495, 'z': 0.08951391725506215, 'c': 0.05601913987278987, 's': 0.0637801248759993, 'r': 0.014996790570111454, 'y': 0.09873373402579215, 'w': 0.03063546711793196, "a'": 0.00040847289490575947, 'a': 0.008052751356713544, "e'": 0.004960028009569936, 'e': 0.002976016805741962, "o'": 0.0002917663535041139, 'o': 0.001575538308922215, '$': 0.005835327070082278}, "v'": {'b': 0.04026375678356772, 'p': 0.031160646554239366, "m'": 0.00017505981210246834, 'm': 0.037462799789928225, 'f': 0.027776156853591645, 'd': 0.04358989321351462, 't': 0.043881659567018734, "n'": 0.00011670654140164556, 'n': 0.022641069031919238, 'l': 0.0735251210830367, 'g': 0.035945614751706836, 'k': 0.026142265273968606, 'h': 0.04819980159887962, 'j': 0.07317500145883177, 'q': 0.050825698780416644, 'x': 0.0673396743887495, 'z': 0.08951391725506215, 'c': 0.05601913987278987, 's': 0.0637801248759993, 'r': 0.014996790570111454, 'y': 0.09873373402579215, 'w': 0.03063546711793196, "a'": 0.00040847289490575947, 'a': 0.008052751356713544, "e'": 0.004960028009569936, 'e': 0.002976016805741962, "o'": 0.0002917663535041139, 'o': 0.001575538308922215, '$': 0.005835327070082278}, "r'": {'b': 0.04026375678356772, 'p': 0.031160646554239366, "m'": 0.00017505981210246834, 'm': 0.037462799789928225, 'f': 0.027776156853591645, 'd': 0.04358989321351462, 't': 0.043881659567018734, "n'": 0.00011670654140164556, 'n': 0.022641069031919238, 'l': 0.0735251210830367, 'g': 0.035945614751706836, 'k': 0.026142265273968606, 'h': 0.04819980159887962, 'j': 0.07317500145883177, 'q': 0.050825698780416644, 'x': 0.0673396743887495, 'z': 0.08951391725506215, 'c': 0.05601913987278987, 's': 0.0637801248759993, 'r': 0.014996790570111454, 'y': 0.09873373402579215, 'w': 0.03063546711793196, "a'": 0.00040847289490575947, 'a': 0.008052751356713544, "e'": 0.004960028009569936, 'e': 0.002976016805741962, "o'": 0.0002917663535041139, 'o': 0.001575538308922215, '$': 0.005835327070082278}}
print(test_dict)
a = pd.DataFrame(test_dict)
a = a.fillna(value=0)
print(a)









