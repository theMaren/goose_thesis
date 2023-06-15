; The variants of the Organic Synthesis domain were created by
; Dr. Russell Viirre, Hadi Qovaizi, and Prof. Mikhail Soutchanski.
;
; This work is licensed under a Creative Commons Attribution,
; NonCommercial, ShareAlike 3.0 Unported License.
;
; For further information, please access the following web page:
; https://www.cs.ryerson.ca/~mes/publications/
(define (problem initialBonds7) (:domain Chemical)
(:objects
; setup for problem 7  - ethanol
h17 - hydrogen
h18 - hydrogen
h19 - hydrogen
h20 - hydrogen
h21 - hydrogen
h22 - hydrogen
c3 - carbon
c4 - carbon
o2 - oxygen
; Second EtOH
h23 - hydrogen
h24 - hydrogen
h25 - hydrogen
h26 - hydrogen
h27 - hydrogen
h28 - hydrogen
c5 - carbon
c6 - carbon
o3 - oxygen
; PBr3 
p1 - phosphorus
br1 - bromine
br2 - bromine
br3 - bromine
; NaCN 
c2 - carbon
na3 - sodium
n1 - nitrogen
h54 - hydrogen
; LiAlH4
li1 - lithium
al1 - aluminium
h5 - hydrogen
h6 - hydrogen
h7 - hydrogen
h8 - hydrogen
; potassium Permanganate 
k1 - potassium
mn1 - manganese
o14 - oxygen
o15 - oxygen
o16 - oxygen
o17 - oxygen
; Thionyl chloride 
su1 - sulfur
o18 - oxygen
cl2 - chlorine
cl3 - chlorine
; third EtOH
h29 - hydrogen
h30 - hydrogen
h31 - hydrogen
h32 - hydrogen
h33 - hydrogen
h34 - hydrogen
c7 - carbon
c8 - carbon
o1 - oxygen
; PCC 
c12 - carbon
c13 - carbon
c14 - carbon
c15 - carbon
c16 - carbon
n4 - nitrogen
h9 - hydrogen
h10 - hydrogen
h11 - hydrogen
h12 - hydrogen
h13 - hydrogen
h14 - hydrogen
cr - chromium
o4 - oxygen
o5 - oxygen
o6 - oxygen
cl1 - chlorine
; second LiAlH4
li2 - lithium
al2 - aluminium
h35 - hydrogen
h36 - hydrogen
h37 - hydrogen
h38 - hydrogen
; H2O_1
h50 - hydrogen
h51 - hydrogen
o50 - oxygen
; H2O_2
h52 - hydrogen
h53 - hydrogen
o51 - oxygen
)
(:init
; setup for problem 7  - ethanol
(bond c3 c4)
(bond c4 c3)
(bond c4 o2)
(bond o2 c4)
(bond h17 c3)
(bond h18 c3)
(bond h19 c3)
(bond c3 h17)
(bond c3 h18)
(bond c3 h19)
(bond c4 h20)
(bond c4 h21)
(bond h20 c4)
(bond h21 c4)
(bond o2 h22)
(bond h22 o2)
; Second EtOH
(bond c5 c6)
(bond c6 c5)
(bond c6 o3)
(bond o3 c6)
(bond h23 c5)
(bond h24 c5)
(bond h25 c5)
(bond c5 h23)
(bond c5 h24)
(bond c5 h25)
(bond c6 h26)
(bond c6 h27)
(bond h26 c6)
(bond h27 c6)
(bond o3 h28)
(bond h28 o3)
; PBr3 
(bond p1 br1)
(bond p1 br2)
(bond p1 br3)
(bond br1 p1)
(bond br2 p1)
(bond br3 p1)
; NaCN 
(bond na3 c2)
(bond c2 na3)
(triplebond c2 n1)
(triplebond n1 c2)
(bond h54 n1)
(bond n1 h54)
; LiAlH4
(bond al1 h5)
(bond al1 h6)
(bond al1 h7)
(bond al1 h8)
(bond h5 al1)
(bond h6 al1)
(bond h7 al1)
(bond h8 al1)
; potassium Permanganate 
(doublebond mn1 o14)
(doublebond mn1 o15)
(doublebond mn1 o16)
(bond mn1 o17)
(doublebond o14 mn1)
(doublebond o15 mn1)
(doublebond o16 mn1)
(bond o17 mn1)
(bond k1 mn1)
(bond mn1 k1)
; Thionyl chloride 
(doublebond su1 o18)
(doublebond o18 su1)
(bond cl2 su1)
(bond cl3 su1)
(bond su1 cl2)
(bond su1 cl3)
; third EtOH
(bond c7 c8)
(bond c8 c7)
(bond c8 o1)
(bond o1 c8)
(bond h29 c7)
(bond h30 c7)
(bond h31 c7)
(bond c7 h29)
(bond c7 h30)
(bond c7 h31)
(bond c8 h32)
(bond c8 h33)
(bond h32 c8)
(bond h33 c8)
(bond o1 h34)
(bond h34 o1)
; PCC 
(bond n4 h9)
(bond h9 n4)
(aromaticbond c12 n4)
(aromaticbond c12 c13)
(aromaticbond c13 c14)
(aromaticbond c14 c15)
(aromaticbond c15 c16)
(aromaticbond c16 n4)
(aromaticbond n4 c12)
(aromaticbond c13 c12)
(aromaticbond c14 c13)
(aromaticbond c15 c14)
(aromaticbond c16 c15)
(aromaticbond n4 c16)
(bond h10 c12)
(bond h11 c13)
(bond h12 c14)
(bond h13 c15)
(bond h14 c16)
(bond c12 h10)
(bond c13 h11)
(bond c14 h12)
(bond c15 h13)
(bond c16 h14)
(bond o4 cr)
(doublebond cr o5)
(doublebond cr o6)
(bond cr cl1)
(bond cr o4)
(doublebond o5 cr)
(doublebond o6 cr)
(bond cl1 cr)
; second LiAlH4
(bond al2 h35)
(bond al2 h36)
(bond al2 h37)
(bond al2 h38)
(bond h35 al2)
(bond h36 al2)
(bond h37 al2)
(bond h38 al2)
; H2O_1
(bond h50 o50)
(bond o50 h50)
(bond h51 o50)
(bond o50 h51)
; H2O_2
(bond h52 o51)
(bond o51 h52)
(bond h53 o51)
(bond o51 h53)
)
(:goal
(and
(bond c3 c4)
(bond c4 c2)
(bond c2 n1)
(bond n1 c8)
(bond c8 c7)
(bond n1 c6)
(doublebond c6 o3)
(bond c6 c5)
(bond c3 h17)
(bond c3 h18)
(bond c3 h19)
(bond c4 h20)
(bond c4 h21)
(bond c2 h5)
(bond c2 h6)
(bond c8 h33)
(bond c8 h35)
(bond c7 h29)
(bond c7 h30)
(bond c7 h31)
(bond c5 h23)
(bond c5 h24)
(bond c5 h25)
)
)
)
