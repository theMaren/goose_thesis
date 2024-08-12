;; vehicles=42, packages=128, locations=80, max_capacity=10, out_folder=testing_new/hard, instance_id=19, seed=2026

(define (problem transport-19)
 (:domain transport)
 (:objects 
    v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 - vehicle
    p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31 p32 p33 p34 p35 p36 p37 p38 p39 p40 p41 p42 p43 p44 p45 p46 p47 p48 p49 p50 p51 p52 p53 p54 p55 p56 p57 p58 p59 p60 p61 p62 p63 p64 p65 p66 p67 p68 p69 p70 p71 p72 p73 p74 p75 p76 p77 p78 p79 p80 p81 p82 p83 p84 p85 p86 p87 p88 p89 p90 p91 p92 p93 p94 p95 p96 p97 p98 p99 p100 p101 p102 p103 p104 p105 p106 p107 p108 p109 p110 p111 p112 p113 p114 p115 p116 p117 p118 p119 p120 p121 p122 p123 p124 p125 p126 p127 p128 - package
    l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l80 - location
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 - size
    )
 (:init (capacity v1 c2)
    (capacity v2 c6)
    (capacity v3 c9)
    (capacity v4 c9)
    (capacity v5 c2)
    (capacity v6 c4)
    (capacity v7 c10)
    (capacity v8 c10)
    (capacity v9 c9)
    (capacity v10 c7)
    (capacity v11 c10)
    (capacity v12 c9)
    (capacity v13 c8)
    (capacity v14 c10)
    (capacity v15 c8)
    (capacity v16 c4)
    (capacity v17 c1)
    (capacity v18 c10)
    (capacity v19 c2)
    (capacity v20 c2)
    (capacity v21 c5)
    (capacity v22 c2)
    (capacity v23 c8)
    (capacity v24 c1)
    (capacity v25 c8)
    (capacity v26 c6)
    (capacity v27 c4)
    (capacity v28 c7)
    (capacity v29 c5)
    (capacity v30 c6)
    (capacity v31 c6)
    (capacity v32 c7)
    (capacity v33 c9)
    (capacity v34 c2)
    (capacity v35 c6)
    (capacity v36 c2)
    (capacity v37 c9)
    (capacity v38 c9)
    (capacity v39 c5)
    (capacity v40 c5)
    (capacity v41 c8)
    (capacity v42 c3)
    (capacity-predecessor c0 c1)
    (capacity-predecessor c1 c2)
    (capacity-predecessor c2 c3)
    (capacity-predecessor c3 c4)
    (capacity-predecessor c4 c5)
    (capacity-predecessor c5 c6)
    (capacity-predecessor c6 c7)
    (capacity-predecessor c7 c8)
    (capacity-predecessor c8 c9)
    (capacity-predecessor c9 c10)
    (at p1 l74)
    (at p2 l40)
    (at p3 l4)
    (at p4 l48)
    (at p5 l47)
    (at p6 l60)
    (at p7 l55)
    (at p8 l12)
    (at p9 l52)
    (at p10 l75)
    (at p11 l71)
    (at p12 l64)
    (at p13 l15)
    (at p14 l55)
    (at p15 l65)
    (at p16 l77)
    (at p17 l63)
    (at p18 l51)
    (at p19 l68)
    (at p20 l34)
    (at p21 l54)
    (at p22 l73)
    (at p23 l62)
    (at p24 l66)
    (at p25 l67)
    (at p26 l4)
    (at p27 l74)
    (at p28 l29)
    (at p29 l17)
    (at p30 l7)
    (at p31 l67)
    (at p32 l14)
    (at p33 l80)
    (at p34 l55)
    (at p35 l61)
    (at p36 l19)
    (at p37 l62)
    (at p38 l30)
    (at p39 l16)
    (at p40 l59)
    (at p41 l61)
    (at p42 l70)
    (at p43 l33)
    (at p44 l72)
    (at p45 l52)
    (at p46 l40)
    (at p47 l71)
    (at p48 l65)
    (at p49 l62)
    (at p50 l52)
    (at p51 l56)
    (at p52 l26)
    (at p53 l38)
    (at p54 l35)
    (at p55 l2)
    (at p56 l5)
    (at p57 l20)
    (at p58 l77)
    (at p59 l59)
    (at p60 l67)
    (at p61 l63)
    (at p62 l47)
    (at p63 l26)
    (at p64 l34)
    (at p65 l63)
    (at p66 l60)
    (at p67 l66)
    (at p68 l64)
    (at p69 l56)
    (at p70 l72)
    (at p71 l5)
    (at p72 l38)
    (at p73 l41)
    (at p74 l59)
    (at p75 l13)
    (at p76 l57)
    (at p77 l66)
    (at p78 l34)
    (at p79 l58)
    (at p80 l24)
    (at p81 l55)
    (at p82 l62)
    (at p83 l9)
    (at p84 l35)
    (at p85 l64)
    (at p86 l48)
    (at p87 l13)
    (at p88 l25)
    (at p89 l53)
    (at p90 l19)
    (at p91 l58)
    (at p92 l2)
    (at p93 l12)
    (at p94 l1)
    (at p95 l58)
    (at p96 l75)
    (at p97 l70)
    (at p98 l11)
    (at p99 l79)
    (at p100 l26)
    (at p101 l32)
    (at p102 l16)
    (at p103 l80)
    (at p104 l6)
    (at p105 l70)
    (at p106 l35)
    (at p107 l46)
    (at p108 l20)
    (at p109 l1)
    (at p110 l41)
    (at p111 l75)
    (at p112 l74)
    (at p113 l63)
    (at p114 l18)
    (at p115 l70)
    (at p116 l29)
    (at p117 l61)
    (at p118 l76)
    (at p119 l17)
    (at p120 l77)
    (at p121 l65)
    (at p122 l38)
    (at p123 l11)
    (at p124 l54)
    (at p125 l21)
    (at p126 l18)
    (at p127 l76)
    (at p128 l40)
    (at v1 l65)
    (at v2 l53)
    (at v3 l50)
    (at v4 l20)
    (at v5 l67)
    (at v6 l26)
    (at v7 l49)
    (at v8 l2)
    (at v9 l14)
    (at v10 l50)
    (at v11 l10)
    (at v12 l7)
    (at v13 l45)
    (at v14 l46)
    (at v15 l4)
    (at v16 l4)
    (at v17 l66)
    (at v18 l61)
    (at v19 l42)
    (at v20 l39)
    (at v21 l58)
    (at v22 l42)
    (at v23 l16)
    (at v24 l64)
    (at v25 l60)
    (at v26 l74)
    (at v27 l80)
    (at v28 l22)
    (at v29 l20)
    (at v30 l76)
    (at v31 l67)
    (at v32 l27)
    (at v33 l56)
    (at v34 l80)
    (at v35 l44)
    (at v36 l59)
    (at v37 l77)
    (at v38 l30)
    (at v39 l9)
    (at v40 l71)
    (at v41 l52)
    (at v42 l5)
    (road l75 l42)
    (road l23 l4)
    (road l13 l33)
    (road l41 l49)
    (road l4 l64)
    (road l64 l69)
    (road l10 l43)
    (road l54 l13)
    (road l44 l54)
    (road l1 l49)
    (road l47 l16)
    (road l34 l19)
    (road l16 l47)
    (road l42 l11)
    (road l42 l75)
    (road l10 l70)
    (road l63 l52)
    (road l39 l60)
    (road l46 l20)
    (road l72 l2)
    (road l43 l39)
    (road l54 l6)
    (road l65 l36)
    (road l64 l16)
    (road l53 l34)
    (road l34 l30)
    (road l29 l43)
    (road l31 l40)
    (road l4 l23)
    (road l65 l72)
    (road l52 l17)
    (road l33 l13)
    (road l38 l64)
    (road l60 l39)
    (road l35 l74)
    (road l79 l28)
    (road l31 l24)
    (road l8 l68)
    (road l57 l52)
    (road l15 l46)
    (road l54 l44)
    (road l72 l52)
    (road l43 l16)
    (road l38 l66)
    (road l77 l48)
    (road l27 l38)
    (road l34 l53)
    (road l19 l34)
    (road l30 l34)
    (road l56 l53)
    (road l34 l62)
    (road l59 l61)
    (road l49 l41)
    (road l4 l18)
    (road l72 l45)
    (road l68 l8)
    (road l6 l54)
    (road l67 l40)
    (road l75 l32)
    (road l64 l4)
    (road l25 l22)
    (road l3 l58)
    (road l40 l22)
    (road l71 l37)
    (road l58 l3)
    (road l40 l31)
    (road l66 l50)
    (road l22 l25)
    (road l32 l75)
    (road l26 l4)
    (road l48 l79)
    (road l72 l65)
    (road l11 l34)
    (road l42 l49)
    (road l54 l14)
    (road l34 l11)
    (road l38 l27)
    (road l5 l2)
    (road l37 l71)
    (road l37 l16)
    (road l74 l65)
    (road l55 l33)
    (road l13 l73)
    (road l65 l74)
    (road l46 l15)
    (road l8 l40)
    (road l40 l8)
    (road l64 l72)
    (road l17 l52)
    (road l36 l65)
    (road l78 l19)
    (road l79 l48)
    (road l41 l9)
    (road l64 l38)
    (road l18 l4)
    (road l2 l72)
    (road l53 l56)
    (road l46 l72)
    (road l73 l13)
    (road l9 l41)
    (road l66 l38)
    (road l63 l21)
    (road l41 l48)
    (road l69 l64)
    (road l52 l57)
    (road l16 l37)
    (road l62 l34)
    (road l4 l26)
    (road l16 l64)
    (road l20 l46)
    (road l22 l43)
    (road l74 l35)
    (road l40 l67)
    (road l43 l29)
    (road l21 l63)
    (road l51 l42)
    (road l52 l41)
    (road l55 l12)
    (road l49 l42)
    (road l70 l10)
    (road l33 l55)
    (road l13 l52)
    (road l52 l13)
    (road l72 l46)
    (road l13 l61)
    (road l39 l43)
    (road l68 l3)
    (road l43 l22)
    (road l11 l42)
    (road l3 l68)
    (road l41 l52)
    (road l42 l51)
    (road l13 l54)
    (road l28 l79)
    (road l2 l5)
    (road l49 l1)
    (road l12 l55)
    (road l50 l66)
    (road l52 l63)
    (road l16 l43)
    (road l48 l41)
    (road l52 l72)
    (road l42 l7)
    (road l76 l11)
    (road l19 l78)
    (road l22 l40)
    (road l48 l77)
    (road l14 l54)
    (road l24 l31)
    (road l72 l80)
    (road l61 l59)
    (road l80 l72)
    (road l45 l72)
    (road l61 l13)
    (road l7 l42)
    (road l43 l10)
    (road l11 l76)
    (road l72 l64)
    (road l33 l52)
    (road l52 l33)
    (road l1 l4)
    (road l4 l1)
    (road l2 l28)
    (road l28 l2)
    (road l14 l58)
    (road l58 l14)
    (road l15 l78)
    (road l78 l15)
    (road l8 l43)
    (road l43 l8)
    (road l26 l48)
    (road l48 l26)
    (road l10 l26)
    (road l26 l10)
    (road l21 l43)
    (road l43 l21)
    (road l18 l33)
    (road l33 l18)
    (road l9 l34)
    (road l34 l9)
    (road l6 l60)
    (road l60 l6)
    (road l44 l67)
    (road l67 l44)
    (road l34 l74)
    (road l74 l34)
    (road l26 l39)
    (road l39 l26)
    (road l34 l35)
    (road l35 l34)
    (road l5 l33)
    (road l33 l5)
    (road l7 l27)
    (road l27 l7)
    (road l38 l53)
    (road l53 l38)
    (road l27 l62)
    (road l62 l27)
    (road l2 l36)
    (road l36 l2)
    (road l56 l74)
    (road l74 l56)
    (road l10 l54)
    (road l54 l10)
    (road l18 l21)
    (road l21 l18)
    (road l43 l53)
    (road l53 l43)
    (road l8 l27)
    (road l27 l8)
    (road l8 l66)
    (road l66 l8)
    (road l38 l54)
    (road l54 l38)
    (road l34 l68)
    (road l68 l34)
    (road l1 l35)
    (road l35 l1)
    (road l53 l62)
    (road l62 l53)
    (road l28 l67)
    (road l67 l28)
    (road l1 l65)
    (road l65 l1)
    (road l33 l57)
    (road l57 l33)
    (road l27 l31)
    (road l31 l27)
    (road l19 l53)
    (road l53 l19)
    (road l14 l31)
    (road l31 l14)
    (road l42 l65)
    (road l65 l42)
    (road l6 l30)
    (road l30 l6)
    (road l17 l63)
    (road l63 l17)
    (road l12 l71)
    (road l71 l12)
    (road l2 l59)
    (road l59 l2)
    (road l32 l50)
    (road l50 l32)
    (road l28 l30)
    (road l30 l28)
    (road l33 l38)
    (road l38 l33)
    (road l60 l77)
    (road l77 l60)
    (road l29 l69)
    (road l69 l29)
    (road l32 l40)
    (road l40 l32)
    (road l18 l27)
    (road l27 l18)
    (road l40 l42)
    (road l42 l40)
    (road l60 l76)
    (road l76 l60)
    (road l75 l78)
    (road l78 l75)
    (road l25 l44)
    (road l44 l25)
    (road l6 l41)
    (road l41 l6)
    (road l31 l60)
    (road l60 l31)
    (road l10 l61)
    (road l61 l10)
    (road l34 l69)
    (road l69 l34)
    (road l59 l68)
    (road l68 l59)
    (road l42 l62)
    (road l62 l42)
    (road l21 l34)
    (road l34 l21)
    (road l18 l22)
    (road l22 l18)
    (road l32 l34)
    (road l34 l32)
    (road l26 l32)
    (road l32 l26)
    (road l40 l61)
    (road l61 l40)
    (road l40 l76)
    (road l76 l40)
    (road l29 l78)
    (road l78 l29)
    (road l19 l47)
    (road l47 l19)
    (road l40 l52)
    (road l52 l40)
    (road l21 l77)
    (road l77 l21)
    (road l49 l64)
    (road l64 l49)
    (road l17 l55)
    (road l55 l17)
    (road l41 l69)
    (road l69 l41)
    (road l17 l27)
    (road l27 l17)
    (road l45 l50)
    (road l50 l45)
    (road l14 l48)
    (road l48 l14)
    (road l9 l73)
    (road l73 l9)
    (road l20 l21)
    (road l21 l20)
    (road l10 l37)
    (road l37 l10)
    (road l54 l65)
    (road l65 l54)
    (road l20 l36)
    (road l36 l20)
    (road l1 l71)
    (road l71 l1)
    (road l54 l55)
    (road l55 l54)
    (road l8 l61)
    (road l61 l8)
    (road l14 l65)
    (road l65 l14)
    (road l2 l56)
    (road l56 l2)
    (road l19 l61)
    (road l61 l19)
    (road l7 l44)
    (road l44 l7)
    (road l1 l14)
    (road l14 l1)
    (road l15 l17)
    (road l17 l15)
    (road l42 l61)
    (road l61 l42)
    (road l42 l44)
    (road l44 l42)
    (road l2 l61)
    (road l61 l2)
    (road l24 l77)
    (road l77 l24)
    (road l2 l78)
    (road l78 l2)
    (road l8 l18)
    (road l18 l8)
    (road l32 l47)
    (road l47 l32)
    (road l26 l53)
    (road l53 l26)
    (road l8 l64)
    (road l64 l8)
    (road l18 l36)
    (road l36 l18)
    (road l32 l56)
    (road l56 l32)
    (road l20 l37)
    (road l37 l20)
    (road l9 l25)
    (road l25 l9)
    (road l24 l66)
    (road l66 l24)
    (road l50 l69)
    (road l69 l50)
    (road l10 l78)
    (road l78 l10)
    (road l23 l73)
    (road l73 l23)
    (road l58 l69)
    (road l69 l58)
    (road l14 l73)
    (road l73 l14)
    (road l43 l44)
    (road l44 l43)
    (road l35 l70)
    (road l70 l35)
    (road l5 l38)
    (road l38 l5)
    (road l8 l16)
    (road l16 l8)
    (road l24 l34)
    (road l34 l24)
    (road l30 l38)
    (road l38 l30)
    (road l14 l80)
    (road l80 l14)
    (road l3 l4)
    (road l4 l3)
    (road l12 l59)
    (road l59 l12)
    (road l35 l46)
    (road l46 l35)
    (road l23 l64)
    (road l64 l23)
    (road l3 l44)
    (road l44 l3)
    (road l18 l24)
    (road l24 l18)
    (road l13 l19)
    (road l19 l13)
    (road l7 l11)
    (road l11 l7)
    (road l26 l37)
    (road l37 l26)
    (road l26 l77)
    (road l77 l26)
    (road l26 l47)
    (road l47 l26)
    (road l23 l42)
    (road l42 l23)
    (road l24 l30)
    (road l30 l24)
    (road l36 l64)
    (road l64 l36)
    (road l6 l55)
    (road l55 l6)
    (road l8 l26)
    (road l26 l8)
    (road l7 l39)
    (road l39 l7)
    (road l17 l72)
    (road l72 l17)
    (road l26 l40)
    (road l40 l26)
    (road l14 l34)
    (road l34 l14)
    (road l27 l69)
    (road l69 l27)
    (road l57 l65)
    (road l65 l57)
    (road l13 l79)
    (road l79 l13)
    (road l17 l66)
    (road l66 l17)
    (road l22 l70)
    (road l70 l22)
    (road l48 l71)
    (road l71 l48)
    (road l27 l70)
    (road l70 l27)
    (road l13 l44)
    (road l44 l13)
    (road l18 l65)
    (road l65 l18)
    (road l28 l34)
    (road l34 l28)
    (road l47 l55)
    (road l55 l47)
    (road l34 l80)
    (road l80 l34)
    (road l16 l73)
    (road l73 l16)
    (road l17 l33)
    (road l33 l17)
    (road l31 l45)
    (road l45 l31)
    (road l17 l58)
    (road l58 l17)
    (road l7 l58)
    (road l58 l7)
    )
 (:goal  (and 
    (at p1 l52)
    (at p2 l70)
    (at p3 l25)
    (at p4 l77)
    (at p5 l80)
    (at p6 l44)
    (at p7 l50)
    (at p8 l66)
    (at p9 l13)
    (at p10 l56)
    (at p11 l23)
    (at p12 l10)
    (at p13 l3)
    (at p14 l34)
    (at p15 l15)
    (at p16 l8)
    (at p17 l31)
    (at p18 l42)
    (at p19 l24)
    (at p20 l37)
    (at p21 l62)
    (at p22 l66)
    (at p23 l58)
    (at p24 l47)
    (at p25 l26)
    (at p26 l5)
    (at p27 l5)
    (at p28 l46)
    (at p29 l6)
    (at p30 l24)
    (at p31 l79)
    (at p32 l63)
    (at p33 l59)
    (at p34 l48)
    (at p35 l58)
    (at p36 l40)
    (at p37 l65)
    (at p38 l69)
    (at p39 l62)
    (at p40 l70)
    (at p41 l11)
    (at p42 l5)
    (at p43 l10)
    (at p44 l2)
    (at p45 l65)
    (at p46 l13)
    (at p47 l6)
    (at p48 l18)
    (at p49 l26)
    (at p50 l18)
    (at p51 l5)
    (at p52 l79)
    (at p53 l27)
    (at p54 l8)
    (at p55 l56)
    (at p56 l48)
    (at p57 l35)
    (at p58 l64)
    (at p59 l64)
    (at p60 l11)
    (at p61 l17)
    (at p62 l31)
    (at p63 l64)
    (at p64 l55)
    (at p65 l21)
    (at p66 l69)
    (at p67 l63)
    (at p68 l23)
    (at p69 l5)
    (at p70 l35)
    (at p71 l45)
    (at p72 l32)
    (at p73 l26)
    (at p74 l9)
    (at p75 l52)
    (at p76 l15)
    (at p77 l68)
    (at p78 l71)
    (at p79 l13)
    (at p80 l34)
    (at p81 l40)
    (at p82 l26)
    (at p83 l25)
    (at p84 l57)
    (at p85 l51)
    (at p86 l3)
    (at p87 l5)
    (at p88 l12)
    (at p89 l21)
    (at p90 l11)
    (at p91 l24)
    (at p92 l7)
    (at p93 l49)
    (at p94 l50)
    (at p95 l55)
    (at p96 l15)
    (at p97 l19)
    (at p98 l2)
    (at p99 l42)
    (at p100 l43)
    (at p101 l73)
    (at p102 l30)
    (at p103 l20)
    (at p104 l71)
    (at p105 l17)
    (at p106 l43)
    (at p107 l3)
    (at p108 l64)
    (at p109 l7)
    (at p110 l12)
    (at p111 l33)
    (at p112 l53)
    (at p113 l47)
    (at p114 l58)
    (at p115 l54)
    (at p116 l9)
    (at p117 l17)
    (at p118 l55)
    (at p119 l46)
    (at p120 l18)
    (at p121 l33)
    (at p122 l74)
    (at p123 l12)
    (at p124 l26)
    (at p125 l71)
    (at p126 l13)
    (at p127 l16)
    (at p128 l35))))
