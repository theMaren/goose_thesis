;; vehicles=32, packages=44, locations=56, max_capacity=10, out_folder=testing_new/hard, instance_id=5, seed=2012

(define (problem transport-05)
 (:domain transport)
 (:objects 
    v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 - vehicle
    p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31 p32 p33 p34 p35 p36 p37 p38 p39 p40 p41 p42 p43 p44 - package
    l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l50 l51 l52 l53 l54 l55 l56 - location
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 - size
    )
 (:init (capacity v1 c2)
    (capacity v2 c8)
    (capacity v3 c6)
    (capacity v4 c4)
    (capacity v5 c6)
    (capacity v6 c10)
    (capacity v7 c1)
    (capacity v8 c9)
    (capacity v9 c10)
    (capacity v10 c5)
    (capacity v11 c6)
    (capacity v12 c8)
    (capacity v13 c2)
    (capacity v14 c5)
    (capacity v15 c7)
    (capacity v16 c4)
    (capacity v17 c8)
    (capacity v18 c5)
    (capacity v19 c3)
    (capacity v20 c7)
    (capacity v21 c10)
    (capacity v22 c10)
    (capacity v23 c10)
    (capacity v24 c5)
    (capacity v25 c9)
    (capacity v26 c6)
    (capacity v27 c10)
    (capacity v28 c5)
    (capacity v29 c10)
    (capacity v30 c1)
    (capacity v31 c4)
    (capacity v32 c7)
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
    (at p1 l2)
    (at p2 l3)
    (at p3 l27)
    (at p4 l12)
    (at p5 l24)
    (at p6 l25)
    (at p7 l13)
    (at p8 l23)
    (at p9 l10)
    (at p10 l27)
    (at p11 l54)
    (at p12 l32)
    (at p13 l8)
    (at p14 l38)
    (at p15 l17)
    (at p16 l36)
    (at p17 l43)
    (at p18 l9)
    (at p19 l28)
    (at p20 l33)
    (at p21 l33)
    (at p22 l21)
    (at p23 l45)
    (at p24 l29)
    (at p25 l56)
    (at p26 l10)
    (at p27 l53)
    (at p28 l38)
    (at p29 l27)
    (at p30 l55)
    (at p31 l54)
    (at p32 l23)
    (at p33 l49)
    (at p34 l4)
    (at p35 l30)
    (at p36 l53)
    (at p37 l56)
    (at p38 l30)
    (at p39 l45)
    (at p40 l55)
    (at p41 l45)
    (at p42 l50)
    (at p43 l17)
    (at p44 l37)
    (at v1 l10)
    (at v2 l9)
    (at v3 l29)
    (at v4 l5)
    (at v5 l10)
    (at v6 l31)
    (at v7 l31)
    (at v8 l1)
    (at v9 l9)
    (at v10 l4)
    (at v11 l8)
    (at v12 l34)
    (at v13 l22)
    (at v14 l51)
    (at v15 l1)
    (at v16 l10)
    (at v17 l1)
    (at v18 l35)
    (at v19 l50)
    (at v20 l3)
    (at v21 l37)
    (at v22 l56)
    (at v23 l41)
    (at v24 l47)
    (at v25 l8)
    (at v26 l46)
    (at v27 l33)
    (at v28 l56)
    (at v29 l45)
    (at v30 l27)
    (at v31 l8)
    (at v32 l33)
    (road l1 l31)
    (road l15 l30)
    (road l47 l25)
    (road l3 l22)
    (road l11 l5)
    (road l24 l8)
    (road l48 l29)
    (road l55 l22)
    (road l11 l53)
    (road l37 l17)
    (road l11 l7)
    (road l29 l43)
    (road l38 l55)
    (road l42 l25)
    (road l10 l20)
    (road l8 l32)
    (road l13 l28)
    (road l2 l25)
    (road l22 l3)
    (road l33 l49)
    (road l19 l4)
    (road l23 l20)
    (road l56 l32)
    (road l50 l31)
    (road l16 l8)
    (road l53 l11)
    (road l25 l47)
    (road l19 l6)
    (road l54 l37)
    (road l26 l48)
    (road l55 l38)
    (road l25 l13)
    (road l4 l45)
    (road l18 l7)
    (road l31 l1)
    (road l20 l10)
    (road l48 l26)
    (road l47 l6)
    (road l12 l6)
    (road l14 l3)
    (road l31 l28)
    (road l9 l7)
    (road l6 l47)
    (road l8 l29)
    (road l7 l9)
    (road l24 l25)
    (road l35 l25)
    (road l25 l24)
    (road l22 l55)
    (road l25 l42)
    (road l6 l22)
    (road l15 l34)
    (road l3 l14)
    (road l5 l11)
    (road l38 l36)
    (road l51 l8)
    (road l36 l38)
    (road l7 l11)
    (road l29 l8)
    (road l45 l4)
    (road l25 l35)
    (road l19 l49)
    (road l20 l23)
    (road l30 l15)
    (road l24 l11)
    (road l8 l24)
    (road l13 l41)
    (road l12 l21)
    (road l40 l37)
    (road l30 l8)
    (road l55 l46)
    (road l44 l25)
    (road l22 l52)
    (road l17 l47)
    (road l43 l29)
    (road l22 l6)
    (road l13 l25)
    (road l6 l19)
    (road l7 l18)
    (road l49 l33)
    (road l32 l56)
    (road l4 l19)
    (road l10 l53)
    (road l47 l17)
    (road l41 l13)
    (road l28 l13)
    (road l28 l31)
    (road l9 l27)
    (road l39 l6)
    (road l11 l24)
    (road l6 l12)
    (road l6 l39)
    (road l25 l44)
    (road l31 l50)
    (road l8 l30)
    (road l37 l54)
    (road l49 l19)
    (road l21 l12)
    (road l53 l10)
    (road l8 l51)
    (road l34 l15)
    (road l27 l9)
    (road l32 l8)
    (road l46 l55)
    (road l8 l16)
    (road l25 l2)
    (road l29 l48)
    (road l37 l40)
    (road l17 l37)
    (road l52 l22)
    (road l6 l43)
    (road l43 l6)
    (road l3 l42)
    (road l42 l3)
    (road l26 l55)
    (road l55 l26)
    (road l16 l55)
    (road l55 l16)
    (road l22 l41)
    (road l41 l22)
    (road l11 l38)
    (road l38 l11)
    (road l4 l28)
    (road l28 l4)
    (road l23 l29)
    (road l29 l23)
    (road l38 l40)
    (road l40 l38)
    (road l4 l55)
    (road l55 l4)
    (road l24 l48)
    (road l48 l24)
    (road l2 l5)
    (road l5 l2)
    (road l13 l48)
    (road l48 l13)
    (road l18 l56)
    (road l56 l18)
    (road l8 l15)
    (road l15 l8)
    (road l4 l16)
    (road l16 l4)
    (road l33 l40)
    (road l40 l33)
    (road l17 l40)
    (road l40 l17)
    (road l27 l55)
    (road l55 l27)
    (road l3 l36)
    (road l36 l3)
    (road l48 l52)
    (road l52 l48)
    (road l19 l55)
    (road l55 l19)
    (road l34 l54)
    (road l54 l34)
    (road l28 l51)
    (road l51 l28)
    (road l24 l30)
    (road l30 l24)
    (road l2 l48)
    (road l48 l2)
    (road l22 l29)
    (road l29 l22)
    (road l15 l36)
    (road l36 l15)
    (road l14 l43)
    (road l43 l14)
    (road l38 l46)
    (road l46 l38)
    (road l18 l43)
    (road l43 l18)
    (road l39 l51)
    (road l51 l39)
    (road l39 l42)
    (road l42 l39)
    (road l10 l12)
    (road l12 l10)
    (road l9 l28)
    (road l28 l9)
    (road l41 l46)
    (road l46 l41)
    (road l17 l26)
    (road l26 l17)
    (road l9 l24)
    (road l24 l9)
    (road l1 l20)
    (road l20 l1)
    (road l29 l32)
    (road l32 l29)
    (road l13 l21)
    (road l21 l13)
    (road l22 l42)
    (road l42 l22)
    (road l19 l52)
    (road l52 l19)
    (road l13 l23)
    (road l23 l13)
    (road l4 l15)
    (road l15 l4)
    (road l19 l21)
    (road l21 l19)
    (road l28 l37)
    (road l37 l28)
    (road l21 l55)
    (road l55 l21)
    (road l20 l39)
    (road l39 l20)
    (road l9 l45)
    (road l45 l9)
    (road l1 l25)
    (road l25 l1)
    (road l22 l30)
    (road l30 l22)
    (road l39 l45)
    (road l45 l39)
    (road l14 l21)
    (road l21 l14)
    (road l13 l15)
    (road l15 l13)
    (road l3 l25)
    (road l25 l3)
    (road l23 l35)
    (road l35 l23)
    (road l6 l25)
    (road l25 l6)
    (road l6 l44)
    (road l44 l6)
    (road l6 l31)
    (road l31 l6)
    (road l7 l23)
    (road l23 l7)
    (road l40 l41)
    (road l41 l40)
    (road l10 l47)
    (road l47 l10)
    (road l20 l36)
    (road l36 l20)
    (road l48 l53)
    (road l53 l48)
    (road l16 l20)
    (road l20 l16)
    (road l1 l36)
    (road l36 l1)
    (road l11 l44)
    (road l44 l11)
    (road l2 l22)
    (road l22 l2)
    (road l13 l49)
    (road l49 l13)
    (road l24 l37)
    (road l37 l24)
    (road l28 l38)
    (road l38 l28)
    (road l18 l20)
    (road l20 l18)
    (road l2 l55)
    (road l55 l2)
    (road l21 l48)
    (road l48 l21)
    (road l9 l33)
    (road l33 l9)
    (road l32 l46)
    (road l46 l32)
    (road l27 l28)
    (road l28 l27)
    (road l1 l41)
    (road l41 l1)
    (road l21 l50)
    (road l50 l21)
    (road l10 l56)
    (road l56 l10)
    (road l11 l52)
    (road l52 l11)
    (road l17 l51)
    (road l51 l17)
    (road l3 l20)
    (road l20 l3)
    (road l1 l50)
    (road l50 l1)
    (road l10 l33)
    (road l33 l10)
    (road l1 l16)
    (road l16 l1)
    (road l16 l48)
    (road l48 l16)
    (road l21 l24)
    (road l24 l21)
    (road l20 l35)
    (road l35 l20)
    (road l21 l41)
    (road l41 l21)
    (road l25 l28)
    (road l28 l25)
    (road l12 l39)
    (road l39 l12)
    (road l20 l42)
    (road l42 l20)
    (road l17 l41)
    (road l41 l17)
    (road l16 l51)
    (road l51 l16)
    (road l25 l27)
    (road l27 l25)
    (road l18 l39)
    (road l39 l18)
    (road l34 l47)
    (road l47 l34)
    (road l20 l55)
    (road l55 l20)
    (road l18 l45)
    (road l45 l18)
    (road l53 l55)
    (road l55 l53)
    (road l45 l54)
    (road l54 l45)
    (road l26 l41)
    (road l41 l26)
    (road l2 l10)
    (road l10 l2)
    (road l12 l42)
    (road l42 l12)
    (road l11 l39)
    (road l39 l11)
    (road l6 l53)
    (road l53 l6)
    (road l36 l39)
    (road l39 l36)
    (road l2 l31)
    (road l31 l2)
    (road l14 l18)
    (road l18 l14)
    (road l44 l51)
    (road l51 l44)
    (road l16 l26)
    (road l26 l16)
    (road l19 l32)
    (road l32 l19)
    (road l10 l42)
    (road l42 l10)
    (road l53 l54)
    (road l54 l53)
    (road l8 l27)
    (road l27 l8)
    (road l8 l28)
    (road l28 l8)
    (road l35 l44)
    (road l44 l35)
    (road l1 l4)
    (road l4 l1)
    (road l5 l7)
    (road l7 l5)
    (road l7 l20)
    (road l20 l7)
    (road l21 l46)
    (road l46 l21)
    (road l9 l11)
    (road l11 l9)
    (road l20 l49)
    (road l49 l20)
    (road l8 l52)
    (road l52 l8)
    (road l11 l56)
    (road l56 l11)
    (road l19 l56)
    (road l56 l19)
    (road l29 l50)
    (road l50 l29)
    (road l3 l53)
    (road l53 l3)
    (road l3 l54)
    (road l54 l3)
    (road l27 l52)
    (road l52 l27)
    (road l4 l9)
    (road l9 l4)
    (road l10 l35)
    (road l35 l10)
    (road l28 l45)
    (road l45 l28)
    (road l1 l46)
    (road l46 l1)
    (road l27 l54)
    (road l54 l27)
    (road l3 l19)
    (road l19 l3)
    (road l46 l50)
    (road l50 l46)
    (road l1 l55)
    (road l55 l1)
    (road l23 l47)
    (road l47 l23)
    (road l24 l51)
    (road l51 l24)
    (road l33 l53)
    (road l53 l33)
    (road l42 l49)
    (road l49 l42)
    (road l22 l27)
    (road l27 l22)
    (road l10 l23)
    (road l23 l10)
    (road l9 l49)
    (road l49 l9)
    (road l15 l29)
    (road l29 l15)
    (road l7 l38)
    (road l38 l7)
    (road l7 l13)
    (road l13 l7)
    (road l23 l52)
    (road l52 l23)
    (road l3 l49)
    (road l49 l3)
    (road l5 l53)
    (road l53 l5)
    (road l7 l26)
    (road l26 l7)
    (road l7 l19)
    (road l19 l7)
    (road l52 l56)
    (road l56 l52)
    (road l22 l23)
    (road l23 l22)
    (road l14 l34)
    (road l34 l14)
    (road l1 l38)
    (road l38 l1)
    (road l28 l39)
    (road l39 l28)
    (road l5 l45)
    (road l45 l5)
    (road l11 l36)
    (road l36 l11)
    (road l8 l55)
    (road l55 l8)
    (road l18 l44)
    (road l44 l18)
    (road l31 l42)
    (road l42 l31)
    (road l28 l36)
    (road l36 l28)
    (road l1 l45)
    (road l45 l1)
    (road l15 l48)
    (road l48 l15)
    (road l12 l48)
    (road l48 l12)
    (road l8 l18)
    (road l18 l8)
    (road l29 l37)
    (road l37 l29)
    (road l4 l13)
    (road l13 l4)
    (road l3 l56)
    (road l56 l3)
    (road l9 l10)
    (road l10 l9)
    (road l13 l36)
    (road l36 l13)
    (road l35 l50)
    (road l50 l35)
    (road l3 l45)
    (road l45 l3)
    (road l8 l10)
    (road l10 l8)
    (road l2 l54)
    (road l54 l2)
    (road l20 l51)
    (road l51 l20)
    (road l37 l38)
    (road l38 l37)
    (road l12 l20)
    (road l20 l12)
    (road l13 l53)
    (road l53 l13)
    (road l12 l33)
    (road l33 l12)
    (road l7 l30)
    (road l30 l7)
    (road l5 l16)
    (road l16 l5)
    (road l14 l26)
    (road l26 l14)
    (road l2 l28)
    (road l28 l2)
    (road l3 l47)
    (road l47 l3)
    (road l37 l52)
    (road l52 l37)
    (road l2 l51)
    (road l51 l2)
    (road l21 l44)
    (road l44 l21)
    (road l17 l28)
    (road l28 l17)
    (road l23 l25)
    (road l25 l23)
    (road l46 l47)
    (road l47 l46)
    (road l17 l38)
    (road l38 l17)
    (road l51 l55)
    (road l55 l51)
    (road l25 l26)
    (road l26 l25)
    (road l24 l28)
    (road l28 l24)
    (road l15 l18)
    (road l18 l15)
    (road l33 l43)
    (road l43 l33)
    (road l9 l13)
    (road l13 l9)
    (road l2 l36)
    (road l36 l2)
    )
 (:goal  (and 
    (at p1 l3)
    (at p2 l7)
    (at p3 l54)
    (at p4 l24)
    (at p5 l42)
    (at p6 l42)
    (at p7 l42)
    (at p8 l47)
    (at p9 l31)
    (at p10 l8)
    (at p11 l51)
    (at p12 l7)
    (at p13 l37)
    (at p14 l5)
    (at p15 l45)
    (at p16 l25)
    (at p17 l36)
    (at p18 l13)
    (at p19 l56)
    (at p20 l45)
    (at p21 l19)
    (at p22 l37)
    (at p23 l15)
    (at p24 l40)
    (at p25 l45)
    (at p26 l40)
    (at p27 l56)
    (at p28 l25)
    (at p29 l22)
    (at p30 l36)
    (at p31 l51)
    (at p32 l56)
    (at p33 l21)
    (at p34 l10)
    (at p35 l43)
    (at p36 l9)
    (at p37 l31)
    (at p38 l34)
    (at p39 l7)
    (at p40 l12)
    (at p41 l26)
    (at p42 l32)
    (at p43 l3)
    (at p44 l19))))
