;; spanners=140, nuts=70, locations=55, out_folder=testing_new/hard, instance_id=4, seed=2011

(define (problem spanner-04)
 (:domain spanner)
 (:objects 
    bob - man
    spanner1 spanner2 spanner3 spanner4 spanner5 spanner6 spanner7 spanner8 spanner9 spanner10 spanner11 spanner12 spanner13 spanner14 spanner15 spanner16 spanner17 spanner18 spanner19 spanner20 spanner21 spanner22 spanner23 spanner24 spanner25 spanner26 spanner27 spanner28 spanner29 spanner30 spanner31 spanner32 spanner33 spanner34 spanner35 spanner36 spanner37 spanner38 spanner39 spanner40 spanner41 spanner42 spanner43 spanner44 spanner45 spanner46 spanner47 spanner48 spanner49 spanner50 spanner51 spanner52 spanner53 spanner54 spanner55 spanner56 spanner57 spanner58 spanner59 spanner60 spanner61 spanner62 spanner63 spanner64 spanner65 spanner66 spanner67 spanner68 spanner69 spanner70 spanner71 spanner72 spanner73 spanner74 spanner75 spanner76 spanner77 spanner78 spanner79 spanner80 spanner81 spanner82 spanner83 spanner84 spanner85 spanner86 spanner87 spanner88 spanner89 spanner90 spanner91 spanner92 spanner93 spanner94 spanner95 spanner96 spanner97 spanner98 spanner99 spanner100 spanner101 spanner102 spanner103 spanner104 spanner105 spanner106 spanner107 spanner108 spanner109 spanner110 spanner111 spanner112 spanner113 spanner114 spanner115 spanner116 spanner117 spanner118 spanner119 spanner120 spanner121 spanner122 spanner123 spanner124 spanner125 spanner126 spanner127 spanner128 spanner129 spanner130 spanner131 spanner132 spanner133 spanner134 spanner135 spanner136 spanner137 spanner138 spanner139 spanner140 - spanner
    nut1 nut2 nut3 nut4 nut5 nut6 nut7 nut8 nut9 nut10 nut11 nut12 nut13 nut14 nut15 nut16 nut17 nut18 nut19 nut20 nut21 nut22 nut23 nut24 nut25 nut26 nut27 nut28 nut29 nut30 nut31 nut32 nut33 nut34 nut35 nut36 nut37 nut38 nut39 nut40 nut41 nut42 nut43 nut44 nut45 nut46 nut47 nut48 nut49 nut50 nut51 nut52 nut53 nut54 nut55 nut56 nut57 nut58 nut59 nut60 nut61 nut62 nut63 nut64 nut65 nut66 nut67 nut68 nut69 nut70 - nut
    shed location1 location2 location3 location4 location5 location6 location7 location8 location9 location10 location11 location12 location13 location14 location15 location16 location17 location18 location19 location20 location21 location22 location23 location24 location25 location26 location27 location28 location29 location30 location31 location32 location33 location34 location35 location36 location37 location38 location39 location40 location41 location42 location43 location44 location45 location46 location47 location48 location49 location50 location51 location52 location53 location54 location55 gate - location
 )
 (:init 
    (at bob shed)
    (at spanner1 location34)
    (usable spanner1)
    (at spanner2 location35)
    (usable spanner2)
    (at spanner3 location41)
    (usable spanner3)
    (at spanner4 location16)
    (usable spanner4)
    (at spanner5 location52)
    (usable spanner5)
    (at spanner6 location20)
    (usable spanner6)
    (at spanner7 location11)
    (usable spanner7)
    (at spanner8 location19)
    (usable spanner8)
    (at spanner9 location4)
    (usable spanner9)
    (at spanner10 location42)
    (usable spanner10)
    (at spanner11 location24)
    (usable spanner11)
    (at spanner12 location2)
    (usable spanner12)
    (at spanner13 location19)
    (usable spanner13)
    (at spanner14 location14)
    (usable spanner14)
    (at spanner15 location54)
    (usable spanner15)
    (at spanner16 location15)
    (usable spanner16)
    (at spanner17 location45)
    (usable spanner17)
    (at spanner18 location3)
    (usable spanner18)
    (at spanner19 location4)
    (usable spanner19)
    (at spanner20 location12)
    (usable spanner20)
    (at spanner21 location21)
    (usable spanner21)
    (at spanner22 location26)
    (usable spanner22)
    (at spanner23 location33)
    (usable spanner23)
    (at spanner24 location30)
    (usable spanner24)
    (at spanner25 location39)
    (usable spanner25)
    (at spanner26 location2)
    (usable spanner26)
    (at spanner27 location8)
    (usable spanner27)
    (at spanner28 location43)
    (usable spanner28)
    (at spanner29 location19)
    (usable spanner29)
    (at spanner30 location41)
    (usable spanner30)
    (at spanner31 location25)
    (usable spanner31)
    (at spanner32 location39)
    (usable spanner32)
    (at spanner33 location54)
    (usable spanner33)
    (at spanner34 location11)
    (usable spanner34)
    (at spanner35 location25)
    (usable spanner35)
    (at spanner36 location54)
    (usable spanner36)
    (at spanner37 location46)
    (usable spanner37)
    (at spanner38 location52)
    (usable spanner38)
    (at spanner39 location36)
    (usable spanner39)
    (at spanner40 location34)
    (usable spanner40)
    (at spanner41 location20)
    (usable spanner41)
    (at spanner42 location42)
    (usable spanner42)
    (at spanner43 location42)
    (usable spanner43)
    (at spanner44 location16)
    (usable spanner44)
    (at spanner45 location39)
    (usable spanner45)
    (at spanner46 location16)
    (usable spanner46)
    (at spanner47 location39)
    (usable spanner47)
    (at spanner48 location49)
    (usable spanner48)
    (at spanner49 location4)
    (usable spanner49)
    (at spanner50 location38)
    (usable spanner50)
    (at spanner51 location40)
    (usable spanner51)
    (at spanner52 location37)
    (usable spanner52)
    (at spanner53 location50)
    (usable spanner53)
    (at spanner54 location2)
    (usable spanner54)
    (at spanner55 location12)
    (usable spanner55)
    (at spanner56 location38)
    (usable spanner56)
    (at spanner57 location8)
    (usable spanner57)
    (at spanner58 location30)
    (usable spanner58)
    (at spanner59 location29)
    (usable spanner59)
    (at spanner60 location35)
    (usable spanner60)
    (at spanner61 location38)
    (usable spanner61)
    (at spanner62 location26)
    (usable spanner62)
    (at spanner63 location46)
    (usable spanner63)
    (at spanner64 location47)
    (usable spanner64)
    (at spanner65 location48)
    (usable spanner65)
    (at spanner66 location32)
    (usable spanner66)
    (at spanner67 location25)
    (usable spanner67)
    (at spanner68 location15)
    (usable spanner68)
    (at spanner69 location34)
    (usable spanner69)
    (at spanner70 location1)
    (usable spanner70)
    (at spanner71 location32)
    (usable spanner71)
    (at spanner72 location50)
    (usable spanner72)
    (at spanner73 location45)
    (usable spanner73)
    (at spanner74 location49)
    (usable spanner74)
    (at spanner75 location12)
    (usable spanner75)
    (at spanner76 location52)
    (usable spanner76)
    (at spanner77 location45)
    (usable spanner77)
    (at spanner78 location14)
    (usable spanner78)
    (at spanner79 location46)
    (usable spanner79)
    (at spanner80 location39)
    (usable spanner80)
    (at spanner81 location31)
    (usable spanner81)
    (at spanner82 location48)
    (usable spanner82)
    (at spanner83 location51)
    (usable spanner83)
    (at spanner84 location4)
    (usable spanner84)
    (at spanner85 location26)
    (usable spanner85)
    (at spanner86 location54)
    (usable spanner86)
    (at spanner87 location45)
    (usable spanner87)
    (at spanner88 location16)
    (usable spanner88)
    (at spanner89 location23)
    (usable spanner89)
    (at spanner90 location35)
    (usable spanner90)
    (at spanner91 location50)
    (usable spanner91)
    (at spanner92 location15)
    (usable spanner92)
    (at spanner93 location5)
    (usable spanner93)
    (at spanner94 location7)
    (usable spanner94)
    (at spanner95 location36)
    (usable spanner95)
    (at spanner96 location24)
    (usable spanner96)
    (at spanner97 location51)
    (usable spanner97)
    (at spanner98 location49)
    (usable spanner98)
    (at spanner99 location26)
    (usable spanner99)
    (at spanner100 location47)
    (usable spanner100)
    (at spanner101 location41)
    (usable spanner101)
    (at spanner102 location15)
    (usable spanner102)
    (at spanner103 location17)
    (usable spanner103)
    (at spanner104 location4)
    (usable spanner104)
    (at spanner105 location35)
    (usable spanner105)
    (at spanner106 location48)
    (usable spanner106)
    (at spanner107 location34)
    (usable spanner107)
    (at spanner108 location54)
    (usable spanner108)
    (at spanner109 location13)
    (usable spanner109)
    (at spanner110 location46)
    (usable spanner110)
    (at spanner111 location53)
    (usable spanner111)
    (at spanner112 location27)
    (usable spanner112)
    (at spanner113 location42)
    (usable spanner113)
    (at spanner114 location51)
    (usable spanner114)
    (at spanner115 location6)
    (usable spanner115)
    (at spanner116 location35)
    (usable spanner116)
    (at spanner117 location54)
    (usable spanner117)
    (at spanner118 location20)
    (usable spanner118)
    (at spanner119 location25)
    (usable spanner119)
    (at spanner120 location41)
    (usable spanner120)
    (at spanner121 location37)
    (usable spanner121)
    (at spanner122 location6)
    (usable spanner122)
    (at spanner123 location23)
    (usable spanner123)
    (at spanner124 location40)
    (usable spanner124)
    (at spanner125 location44)
    (usable spanner125)
    (at spanner126 location31)
    (usable spanner126)
    (at spanner127 location20)
    (usable spanner127)
    (at spanner128 location28)
    (usable spanner128)
    (at spanner129 location32)
    (usable spanner129)
    (at spanner130 location10)
    (usable spanner130)
    (at spanner131 location22)
    (usable spanner131)
    (at spanner132 location51)
    (usable spanner132)
    (at spanner133 location1)
    (usable spanner133)
    (at spanner134 location30)
    (usable spanner134)
    (at spanner135 location37)
    (usable spanner135)
    (at spanner136 location1)
    (usable spanner136)
    (at spanner137 location17)
    (usable spanner137)
    (at spanner138 location47)
    (usable spanner138)
    (at spanner139 location13)
    (usable spanner139)
    (at spanner140 location50)
    (usable spanner140)
    (at nut1 gate)
    (loose nut1)
    (at nut2 gate)
    (loose nut2)
    (at nut3 gate)
    (loose nut3)
    (at nut4 gate)
    (loose nut4)
    (at nut5 gate)
    (loose nut5)
    (at nut6 gate)
    (loose nut6)
    (at nut7 gate)
    (loose nut7)
    (at nut8 gate)
    (loose nut8)
    (at nut9 gate)
    (loose nut9)
    (at nut10 gate)
    (loose nut10)
    (at nut11 gate)
    (loose nut11)
    (at nut12 gate)
    (loose nut12)
    (at nut13 gate)
    (loose nut13)
    (at nut14 gate)
    (loose nut14)
    (at nut15 gate)
    (loose nut15)
    (at nut16 gate)
    (loose nut16)
    (at nut17 gate)
    (loose nut17)
    (at nut18 gate)
    (loose nut18)
    (at nut19 gate)
    (loose nut19)
    (at nut20 gate)
    (loose nut20)
    (at nut21 gate)
    (loose nut21)
    (at nut22 gate)
    (loose nut22)
    (at nut23 gate)
    (loose nut23)
    (at nut24 gate)
    (loose nut24)
    (at nut25 gate)
    (loose nut25)
    (at nut26 gate)
    (loose nut26)
    (at nut27 gate)
    (loose nut27)
    (at nut28 gate)
    (loose nut28)
    (at nut29 gate)
    (loose nut29)
    (at nut30 gate)
    (loose nut30)
    (at nut31 gate)
    (loose nut31)
    (at nut32 gate)
    (loose nut32)
    (at nut33 gate)
    (loose nut33)
    (at nut34 gate)
    (loose nut34)
    (at nut35 gate)
    (loose nut35)
    (at nut36 gate)
    (loose nut36)
    (at nut37 gate)
    (loose nut37)
    (at nut38 gate)
    (loose nut38)
    (at nut39 gate)
    (loose nut39)
    (at nut40 gate)
    (loose nut40)
    (at nut41 gate)
    (loose nut41)
    (at nut42 gate)
    (loose nut42)
    (at nut43 gate)
    (loose nut43)
    (at nut44 gate)
    (loose nut44)
    (at nut45 gate)
    (loose nut45)
    (at nut46 gate)
    (loose nut46)
    (at nut47 gate)
    (loose nut47)
    (at nut48 gate)
    (loose nut48)
    (at nut49 gate)
    (loose nut49)
    (at nut50 gate)
    (loose nut50)
    (at nut51 gate)
    (loose nut51)
    (at nut52 gate)
    (loose nut52)
    (at nut53 gate)
    (loose nut53)
    (at nut54 gate)
    (loose nut54)
    (at nut55 gate)
    (loose nut55)
    (at nut56 gate)
    (loose nut56)
    (at nut57 gate)
    (loose nut57)
    (at nut58 gate)
    (loose nut58)
    (at nut59 gate)
    (loose nut59)
    (at nut60 gate)
    (loose nut60)
    (at nut61 gate)
    (loose nut61)
    (at nut62 gate)
    (loose nut62)
    (at nut63 gate)
    (loose nut63)
    (at nut64 gate)
    (loose nut64)
    (at nut65 gate)
    (loose nut65)
    (at nut66 gate)
    (loose nut66)
    (at nut67 gate)
    (loose nut67)
    (at nut68 gate)
    (loose nut68)
    (at nut69 gate)
    (loose nut69)
    (at nut70 gate)
    (loose nut70)
    (link shed location1)
    (link location55 gate)
    (link location1 location2)
     (link location2 location3)
     (link location3 location4)
     (link location4 location5)
     (link location5 location6)
     (link location6 location7)
     (link location7 location8)
     (link location8 location9)
     (link location9 location10)
     (link location10 location11)
     (link location11 location12)
     (link location12 location13)
     (link location13 location14)
     (link location14 location15)
     (link location15 location16)
     (link location16 location17)
     (link location17 location18)
     (link location18 location19)
     (link location19 location20)
     (link location20 location21)
     (link location21 location22)
     (link location22 location23)
     (link location23 location24)
     (link location24 location25)
     (link location25 location26)
     (link location26 location27)
     (link location27 location28)
     (link location28 location29)
     (link location29 location30)
     (link location30 location31)
     (link location31 location32)
     (link location32 location33)
     (link location33 location34)
     (link location34 location35)
     (link location35 location36)
     (link location36 location37)
     (link location37 location38)
     (link location38 location39)
     (link location39 location40)
     (link location40 location41)
     (link location41 location42)
     (link location42 location43)
     (link location43 location44)
     (link location44 location45)
     (link location45 location46)
     (link location46 location47)
     (link location47 location48)
     (link location48 location49)
     (link location49 location50)
     (link location50 location51)
     (link location51 location52)
     (link location52 location53)
     (link location53 location54)
     (link location54 location55)
 )
 (:goal  (and (tightened nut1)
   (tightened nut2)
   (tightened nut3)
   (tightened nut4)
   (tightened nut5)
   (tightened nut6)
   (tightened nut7)
   (tightened nut8)
   (tightened nut9)
   (tightened nut10)
   (tightened nut11)
   (tightened nut12)
   (tightened nut13)
   (tightened nut14)
   (tightened nut15)
   (tightened nut16)
   (tightened nut17)
   (tightened nut18)
   (tightened nut19)
   (tightened nut20)
   (tightened nut21)
   (tightened nut22)
   (tightened nut23)
   (tightened nut24)
   (tightened nut25)
   (tightened nut26)
   (tightened nut27)
   (tightened nut28)
   (tightened nut29)
   (tightened nut30)
   (tightened nut31)
   (tightened nut32)
   (tightened nut33)
   (tightened nut34)
   (tightened nut35)
   (tightened nut36)
   (tightened nut37)
   (tightened nut38)
   (tightened nut39)
   (tightened nut40)
   (tightened nut41)
   (tightened nut42)
   (tightened nut43)
   (tightened nut44)
   (tightened nut45)
   (tightened nut46)
   (tightened nut47)
   (tightened nut48)
   (tightened nut49)
   (tightened nut50)
   (tightened nut51)
   (tightened nut52)
   (tightened nut53)
   (tightened nut54)
   (tightened nut55)
   (tightened nut56)
   (tightened nut57)
   (tightened nut58)
   (tightened nut59)
   (tightened nut60)
   (tightened nut61)
   (tightened nut62)
   (tightened nut63)
   (tightened nut64)
   (tightened nut65)
   (tightened nut66)
   (tightened nut67)
   (tightened nut68)
   (tightened nut69)
   (tightened nut70))))
