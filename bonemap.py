"""Bone names (Bayonetta 3 style) for the Eternal Darkness skeleton.

Mapping supplied by Pearl23 for Alexandra's skeleton (importer bone index -> name).  It is stored by the game's bone
name hash, so it applies to every model that uses the same bones (cutscene actors share most of them).  Bones that
are not listed keep the default name  b<index>_<hash>.
"""
BONE_NAMES = {
    0xA1579367: 'bone1280',   # Alex b21 / Pious b20 (added from Pious's model, U_Level01 slot 0)
    0xB4877DD4: 'bone14',   # b01
    0x681A465F: 'bone1',   # b02
    0x1A679F56: 'bone2',   # b03
    0x1A66DF54: 'bone3',   # b04
    0x786A9B21: 'bone4',   # b05
    0xA5E8C9AD: 'bone6',   # b06
    0xC32C2F69: 'bone10',   # b07
    0x9A47880A: 'bone7',   # b08
    0xFC836ECE: 'bone11',   # b09
    0x8C7F9E32: 'bone12',   # b10
    0xF2192054: 'bone8',   # b11
    0x69E347A7: 'bone145',   # b12
    0x6F0D8149: 'bone209',   # b13
    0xAD886CCC: 'bone5',   # b14
    0x359392DB: 'bone15',   # b15
    0xDB0ADC42: 'bone19',   # b16
    0xA154F360: 'bone1345',   # b17
    0xA1567364: 'bone1313',   # b18
    0xA156D365: 'bone1329',   # b19
    0xA1573366: 'bone1297',   # b20
    0xB2999EDF: 'bone16',   # b22
    0xB4775831: 'bone20',   # b23
    0xDF306D03: 'bone1073',   # b24
    0xDF30CD02: 'bone1057',   # b25
    0xDF312D01: 'bone1024',   # b26
    0xDF318D00: 'bone1041',   # b27
    0xDF324D06: 'bone1089',   # b28
    0x30510360: 'bone1074',   # b31
    0x3082E2B3: 'bone1058',   # b32
    0x31258114: 'bone1025',   # b33
    0x31F660C7: 'bone1042',   # b34
    0x326B465A: 'bone1090',   # b35
    0x54AFA09E: 'bone1346',   # b36
    0x56460477: 'bone1314',   # b37
    0x5695E5A4: 'bone1330',   # b38
    0x57328603: 'bone1298',   # b39
    0x57E167D0: 'bone1281',   # b40
    0xF964B630: 'bone17',   # b41
    0xFF8A70DE: 'bone21',   # b42
    0x30504362: 'bone1075',   # b48
    0x3083A2B1: 'bone1059',   # b49
    0x3124C116: 'bone1026',   # b50
    0x31F720C5: 'bone1043',   # b51
    0x326A0658: 'bone1091',   # b52
    0x54AEE09C: 'bone1347',   # b54
    0x56474475: 'bone1315',   # b55
    0x5694A5A6: 'bone1331',   # b56
    0x5733C601: 'bone1299',   # b57
    0x57E027D2: 'bone1282',   # b58
    0xD3B5B685: 'bone18',   # b69
    0xD55B706B: 'bone22',   # b70
}
