from __future__ import annotations

"""
ENDF particle id and reaction type comments

this code is part of the RT2 project
"""


# reaction type #
# noinspection PyDictCreation
REACTION_TYPE = {
    1  : "(total)"     , 2  : "(elastic)"   , 3  : "(nonelastic)",
    4  : "(z,n')"      , 5  : "(any)"       , 10 : "(continuum)" ,
    11 : "(z,2nd)"     , 16 : "(z,2n)"      , 17 : "(z,3n)"      ,
    18 : "(fission)"   , 19 : "(z,f)"       , 20 : "(z,nf)"      ,
    21 : "(z,2nf)"     , 22 : "(z,na)"      , 23 : "(z,n3a)"     ,
    24 : "(z,2na)"     , 25 : "(z,3na)"     , 27 : "(absorp)"    ,
    28 : "(z,np)"      , 29 : "(z,n2a)"     , 30 : "(z,2n2a)"    ,
    32 : "(z,nd)"      , 33 : "(z,nt)"      , 34 : "(z,nHe-3)"   ,
    35 : "(z,nd2a)"    , 36 : "(z,nt2a)"    , 37 : "(z,4n)"      ,
    38 : "(z,3nf)"     , 41 : "(z,2np)"     , 42 : "(z,3np)"     ,
    44 : "(z,n2p)"     , 45 : "(z,npa)"     , 91 : "(z,nc)"      ,
    101: "(disapp)"    , 102: "(z,r)"       , 103: "(z,p)"       ,
    104: "(z,d)"       , 105: "(z,t)"       , 106: "(z,He-3)"    ,
    107: "(z,a)"       , 108: "(z,2a)"      , 109: "(z,3a)"      ,
    111: "(z,2p)"      , 112: "(z,pa)"      , 113: "(z,t2a)"     ,
    114: "(z,d2a)"     , 115: "(z,pd)"      , 116: "(z,pt)"      ,
    117: "(z,da)"      , 201: "(z,xn)"      , 202: "(z,xr)"      ,
    203: "(z,xp)"      , 204: "(z,xd)"      , 205: "(z,xt)"      ,
    206: "(z,xHe-3)"   , 207: "(z,xa)"      , 221: "(thermal)"   ,
    649: "(z,pc)"      , 699: "(z,dc)"      , 749: "(z,tc)"      ,
    799: "(z,3-Hec)"   , 849: "(z,ac)"      , 891: "(z,2nc)"
}

REACTION_TYPE[451] = "(z,...)"

for i in range(41):  # (z,n') reactions
    REACTION_TYPE[i+50] = "(z,n" + str(i) + ")"

for i in range(49):  # (z,p') reactions
    REACTION_TYPE[i+600] = "(z,p" + str(i) + ")"

for i in range(49):  # (z,d') reactions
    REACTION_TYPE[i+650] = "(z,d" + str(i) + ")"

for i in range(49):  # (z,t') reactions
    REACTION_TYPE[i+700] = "(z,t" + str(i) + ")"

for i in range(49):  # (z,He-3') reactions
    REACTION_TYPE[i+750] = "(z,3-He" + str(i) + ")"

for i in range(49):  # (z,a') reactions
    REACTION_TYPE[i+800] = "(z,a" + str(i) + ")"

for i in range(26):  # (z,2n') reactions
    REACTION_TYPE[i+875] = "(z,2n" + str(i) + ")"

SECONDARY_TYPE = {
    6  : "neutron"     , 16 : "gamma"       ,
    21 : "proton"      , 22 : "deuteron"    ,
    23 : "tritium"     , 24 : "helium-3"    ,
    25 : "alpha"
}

SECONDARY_TO_PID = {
    6 : 6,    16: 0,    21: 1001, 22: 1002,
    23: 1003, 24: 2003, 25: 2004
}
