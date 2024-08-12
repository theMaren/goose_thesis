;; vehicles=18, packages=37, locations=36, max_capacity=4, out_folder=testing_new/medium, instance_id=25, seed=2032

(define (problem transport-25)
 (:domain transport)
 (:objects 
    v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 - vehicle
    p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31 p32 p33 p34 p35 p36 p37 - package
    l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l30 l31 l32 l33 l34 l35 l36 - location
    c0 c1 c2 c3 c4 - size
    )
 (:init (capacity v1 c1)
    (capacity v2 c4)
    (capacity v3 c1)
    (capacity v4 c4)
    (capacity v5 c2)
    (capacity v6 c2)
    (capacity v7 c2)
    (capacity v8 c3)
    (capacity v9 c1)
    (capacity v10 c2)
    (capacity v11 c4)
    (capacity v12 c2)
    (capacity v13 c2)
    (capacity v14 c3)
    (capacity v15 c2)
    (capacity v16 c4)
    (capacity v17 c3)
    (capacity v18 c2)
    (capacity-predecessor c0 c1)
    (capacity-predecessor c1 c2)
    (capacity-predecessor c2 c3)
    (capacity-predecessor c3 c4)
    (at p1 l6)
    (at p2 l9)
    (at p3 l35)
    (at p4 l18)
    (at p5 l14)
    (at p6 l8)
    (at p7 l23)
    (at p8 l22)
    (at p9 l30)
    (at p10 l9)
    (at p11 l22)
    (at p12 l13)
    (at p13 l26)
    (at p14 l9)
    (at p15 l3)
    (at p16 l34)
    (at p17 l10)
    (at p18 l22)
    (at p19 l17)
    (at p20 l1)
    (at p21 l28)
    (at p22 l1)
    (at p23 l34)
    (at p24 l33)
    (at p25 l1)
    (at p26 l2)
    (at p27 l12)
    (at p28 l8)
    (at p29 l4)
    (at p30 l14)
    (at p31 l14)
    (at p32 l9)
    (at p33 l2)
    (at p34 l14)
    (at p35 l16)
    (at p36 l10)
    (at p37 l12)
    (at v1 l35)
    (at v2 l23)
    (at v3 l29)
    (at v4 l30)
    (at v5 l8)
    (at v6 l33)
    (at v7 l34)
    (at v8 l33)
    (at v9 l10)
    (at v10 l14)
    (at v11 l11)
    (at v12 l32)
    (at v13 l19)
    (at v14 l1)
    (at v15 l11)
    (at v16 l3)
    (at v17 l23)
    (at v18 l15)
    (road l2 l33)
    (road l6 l15)
    (road l12 l10)
    (road l23 l16)
    (road l4 l6)
    (road l33 l2)
    (road l30 l9)
    (road l8 l6)
    (road l35 l2)
    (road l33 l11)
    (road l16 l4)
    (road l10 l12)
    (road l36 l31)
    (road l32 l21)
    (road l10 l24)
    (road l30 l33)
    (road l13 l17)
    (road l10 l33)
    (road l6 l8)
    (road l1 l24)
    (road l2 l35)
    (road l18 l25)
    (road l6 l29)
    (road l27 l9)
    (road l35 l21)
    (road l17 l8)
    (road l14 l21)
    (road l33 l10)
    (road l8 l17)
    (road l24 l1)
    (road l28 l26)
    (road l9 l22)
    (road l10 l17)
    (road l24 l10)
    (road l6 l4)
    (road l29 l6)
    (road l24 l7)
    (road l26 l10)
    (road l10 l26)
    (road l25 l18)
    (road l29 l5)
    (road l33 l31)
    (road l34 l2)
    (road l21 l14)
    (road l26 l28)
    (road l22 l9)
    (road l2 l34)
    (road l7 l24)
    (road l20 l12)
    (road l4 l16)
    (road l12 l20)
    (road l17 l10)
    (road l21 l35)
    (road l31 l33)
    (road l21 l32)
    (road l17 l13)
    (road l19 l10)
    (road l3 l26)
    (road l31 l36)
    (road l15 l6)
    (road l17 l25)
    (road l5 l29)
    (road l10 l19)
    (road l26 l3)
    (road l9 l27)
    (road l25 l17)
    (road l33 l30)
    (road l9 l30)
    (road l11 l33)
    (road l16 l23)
    (road l20 l28)
    (road l28 l20)
    (road l1 l36)
    (road l36 l1)
    (road l11 l31)
    (road l31 l11)
    (road l24 l35)
    (road l35 l24)
    (road l6 l20)
    (road l20 l6)
    (road l9 l19)
    (road l19 l9)
    (road l1 l35)
    (road l35 l1)
    (road l21 l22)
    (road l22 l21)
    (road l5 l15)
    (road l15 l5)
    (road l3 l25)
    (road l25 l3)
    (road l3 l27)
    (road l27 l3)
    (road l13 l32)
    (road l32 l13)
    (road l2 l12)
    (road l12 l2)
    (road l24 l26)
    (road l26 l24)
    (road l3 l33)
    (road l33 l3)
    (road l4 l22)
    (road l22 l4)
    (road l5 l25)
    (road l25 l5)
    (road l26 l34)
    (road l34 l26)
    (road l24 l29)
    (road l29 l24)
    (road l18 l33)
    (road l33 l18)
    (road l12 l30)
    (road l30 l12)
    (road l11 l17)
    (road l17 l11)
    (road l4 l23)
    (road l23 l4)
    (road l11 l27)
    (road l27 l11)
    (road l5 l22)
    (road l22 l5)
    (road l8 l25)
    (road l25 l8)
    (road l8 l35)
    (road l35 l8)
    (road l15 l29)
    (road l29 l15)
    (road l2 l31)
    (road l31 l2)
    (road l20 l31)
    (road l31 l20)
    (road l4 l32)
    (road l32 l4)
    (road l23 l28)
    (road l28 l23)
    (road l30 l36)
    (road l36 l30)
    (road l16 l17)
    (road l17 l16)
    (road l7 l27)
    (road l27 l7)
    (road l12 l35)
    (road l35 l12)
    (road l7 l19)
    (road l19 l7)
    (road l4 l10)
    (road l10 l4)
    (road l5 l20)
    (road l20 l5)
    (road l12 l14)
    (road l14 l12)
    (road l10 l13)
    (road l13 l10)
    (road l4 l29)
    (road l29 l4)
    (road l18 l27)
    (road l27 l18)
    (road l13 l25)
    (road l25 l13)
    (road l29 l32)
    (road l32 l29)
    (road l10 l15)
    (road l15 l10)
    (road l2 l23)
    (road l23 l2)
    (road l21 l23)
    (road l23 l21)
    (road l2 l5)
    (road l5 l2)
    (road l14 l25)
    (road l25 l14)
    (road l20 l29)
    (road l29 l20)
    (road l8 l18)
    (road l18 l8)
    (road l7 l15)
    (road l15 l7)
    (road l22 l24)
    (road l24 l22)
    (road l7 l35)
    (road l35 l7)
    (road l11 l22)
    (road l22 l11)
    (road l1 l22)
    (road l22 l1)
    (road l6 l13)
    (road l13 l6)
    (road l5 l24)
    (road l24 l5)
    (road l5 l30)
    (road l30 l5)
    (road l25 l30)
    (road l30 l25)
    (road l14 l28)
    (road l28 l14)
    (road l25 l28)
    (road l28 l25)
    (road l15 l26)
    (road l26 l15)
    (road l34 l36)
    (road l36 l34)
    (road l7 l29)
    (road l29 l7)
    (road l20 l30)
    (road l30 l20)
    (road l4 l28)
    (road l28 l4)
    (road l5 l11)
    (road l11 l5)
    (road l11 l32)
    (road l32 l11)
    (road l22 l30)
    (road l30 l22)
    (road l5 l14)
    (road l14 l5)
    (road l19 l28)
    (road l28 l19)
    (road l20 l34)
    (road l34 l20)
    (road l5 l19)
    (road l19 l5)
    (road l6 l19)
    (road l19 l6)
    (road l2 l21)
    (road l21 l2)
    (road l8 l36)
    (road l36 l8)
    (road l1 l32)
    (road l32 l1)
    (road l17 l34)
    (road l34 l17)
    (road l2 l36)
    (road l36 l2)
    (road l15 l17)
    (road l17 l15)
    (road l6 l34)
    (road l34 l6)
    (road l6 l31)
    (road l31 l6)
    (road l17 l28)
    (road l28 l17)
    (road l7 l21)
    (road l21 l7)
    (road l5 l23)
    (road l23 l5)
    (road l18 l35)
    (road l35 l18)
    (road l19 l24)
    (road l24 l19)
    (road l1 l30)
    (road l30 l1)
    (road l1 l18)
    (road l18 l1)
    (road l15 l33)
    (road l33 l15)
    (road l6 l17)
    (road l17 l6)
    (road l14 l17)
    (road l17 l14)
    (road l15 l31)
    (road l31 l15)
    (road l7 l18)
    (road l18 l7)
    (road l6 l36)
    (road l36 l6)
    (road l11 l25)
    (road l25 l11)
    (road l22 l32)
    (road l32 l22)
    (road l16 l27)
    (road l27 l16)
    (road l12 l16)
    (road l16 l12)
    (road l11 l21)
    (road l21 l11)
    (road l29 l30)
    (road l30 l29)
    (road l23 l26)
    (road l26 l23)
    (road l13 l16)
    (road l16 l13)
    (road l32 l36)
    (road l36 l32)
    (road l10 l14)
    (road l14 l10)
    (road l33 l35)
    (road l35 l33)
    (road l4 l7)
    (road l7 l4)
    (road l9 l29)
    (road l29 l9)
    (road l3 l36)
    (road l36 l3)
    (road l5 l32)
    (road l32 l5)
    (road l18 l26)
    (road l26 l18)
    (road l7 l9)
    (road l9 l7)
    (road l28 l31)
    (road l31 l28)
    (road l2 l14)
    (road l14 l2)
    (road l3 l34)
    (road l34 l3)
    (road l26 l36)
    (road l36 l26)
    (road l3 l17)
    (road l17 l3)
    (road l15 l30)
    (road l30 l15)
    (road l25 l26)
    (road l26 l25)
    (road l1 l29)
    (road l29 l1)
    (road l28 l35)
    (road l35 l28)
    (road l12 l36)
    (road l36 l12)
    (road l12 l34)
    (road l34 l12)
    (road l17 l29)
    (road l29 l17)
    (road l8 l28)
    (road l28 l8)
    (road l25 l33)
    (road l33 l25)
    (road l2 l19)
    (road l19 l2)
    (road l5 l10)
    (road l10 l5)
    (road l6 l33)
    (road l33 l6)
    (road l18 l28)
    (road l28 l18)
    (road l31 l32)
    (road l32 l31)
    (road l16 l19)
    (road l19 l16)
    (road l23 l33)
    (road l33 l23)
    (road l18 l23)
    (road l23 l18)
    (road l10 l32)
    (road l32 l10)
    (road l26 l27)
    (road l27 l26)
    (road l27 l36)
    (road l36 l27)
    (road l9 l13)
    (road l13 l9)
    (road l7 l30)
    (road l30 l7)
    (road l2 l10)
    (road l10 l2)
    (road l7 l8)
    (road l8 l7)
    (road l16 l32)
    (road l32 l16)
    (road l23 l29)
    (road l29 l23)
    (road l1 l31)
    (road l31 l1)
    (road l4 l20)
    (road l20 l4)
    (road l7 l28)
    (road l28 l7)
    (road l7 l13)
    (road l13 l7)
    (road l13 l31)
    (road l31 l13)
    (road l16 l18)
    (road l18 l16)
    (road l3 l8)
    (road l8 l3)
    (road l8 l9)
    (road l9 l8)
    (road l7 l11)
    (road l11 l7)
    (road l15 l32)
    (road l32 l15)
    (road l1 l17)
    (road l17 l1)
    (road l13 l24)
    (road l24 l13)
    (road l9 l17)
    (road l17 l9)
    (road l19 l20)
    (road l20 l19)
    (road l12 l21)
    (road l21 l12)
    (road l4 l14)
    (road l14 l4)
    (road l10 l29)
    (road l29 l10)
    (road l8 l13)
    (road l13 l8)
    (road l17 l20)
    (road l20 l17)
    (road l3 l5)
    (road l5 l3)
    (road l8 l16)
    (road l16 l8)
    (road l25 l29)
    (road l29 l25)
    (road l8 l22)
    (road l22 l8)
    (road l7 l26)
    (road l26 l7)
    (road l3 l13)
    (road l13 l3)
    (road l17 l35)
    (road l35 l17)
    (road l14 l20)
    (road l20 l14)
    (road l2 l20)
    (road l20 l2)
    (road l4 l25)
    (road l25 l4)
    (road l21 l30)
    (road l30 l21)
    (road l5 l31)
    (road l31 l5)
    (road l4 l5)
    (road l5 l4)
    (road l6 l18)
    (road l18 l6)
    (road l1 l11)
    (road l11 l1)
    (road l18 l21)
    (road l21 l18)
    (road l7 l34)
    (road l34 l7)
    (road l23 l25)
    (road l25 l23)
    (road l3 l6)
    (road l6 l3)
    (road l13 l20)
    (road l20 l13)
    (road l2 l16)
    (road l16 l2)
    (road l14 l35)
    (road l35 l14)
    (road l17 l27)
    (road l27 l17)
    (road l18 l22)
    (road l22 l18)
    (road l17 l21)
    (road l21 l17)
    (road l7 l17)
    (road l17 l7)
    (road l26 l30)
    (road l30 l26)
    (road l5 l7)
    (road l7 l5)
    (road l4 l19)
    (road l19 l4)
    (road l8 l19)
    (road l19 l8)
    (road l8 l24)
    (road l24 l8)
    (road l9 l16)
    (road l16 l9)
    (road l5 l35)
    (road l35 l5)
    (road l2 l30)
    (road l30 l2)
    (road l1 l15)
    (road l15 l1)
    (road l3 l31)
    (road l31 l3)
    (road l3 l7)
    (road l7 l3)
    (road l2 l6)
    (road l6 l2)
    (road l4 l13)
    (road l13 l4)
    (road l6 l11)
    (road l11 l6)
    (road l18 l32)
    (road l32 l18)
    (road l14 l30)
    (road l30 l14)
    (road l6 l14)
    (road l14 l6)
    (road l3 l19)
    (road l19 l3)
    (road l30 l35)
    (road l35 l30)
    (road l23 l35)
    (road l35 l23)
    (road l18 l34)
    (road l34 l18)
    (road l22 l29)
    (road l29 l22)
    (road l14 l22)
    (road l22 l14)
    (road l12 l27)
    (road l27 l12)
    (road l29 l34)
    (road l34 l29)
    (road l3 l11)
    (road l11 l3)
    (road l18 l19)
    (road l19 l18)
    (road l4 l35)
    (road l35 l4)
    (road l9 l18)
    (road l18 l9)
    (road l1 l5)
    (road l5 l1)
    (road l13 l23)
    (road l23 l13)
    (road l2 l8)
    (road l8 l2)
    (road l11 l15)
    (road l15 l11)
    (road l4 l33)
    (road l33 l4)
    (road l2 l32)
    (road l32 l2)
    (road l3 l4)
    (road l4 l3)
    (road l18 l24)
    (road l24 l18)
    (road l13 l36)
    (road l36 l13)
    (road l31 l35)
    (road l35 l31)
    (road l2 l22)
    (road l22 l2)
    (road l5 l16)
    (road l16 l5)
    (road l2 l28)
    (road l28 l2)
    (road l22 l36)
    (road l36 l22)
    (road l3 l10)
    (road l10 l3)
    (road l1 l6)
    (road l6 l1)
    (road l1 l25)
    (road l25 l1)
    (road l11 l12)
    (road l12 l11)
    (road l9 l26)
    (road l26 l9)
    (road l19 l35)
    (road l35 l19)
    (road l3 l23)
    (road l23 l3)
    (road l12 l29)
    (road l29 l12)
    (road l2 l27)
    (road l27 l2)
    (road l19 l34)
    (road l34 l19)
    (road l22 l31)
    (road l31 l22)
    (road l5 l6)
    (road l6 l5)
    (road l33 l36)
    (road l36 l33)
    (road l3 l35)
    (road l35 l3)
    (road l17 l30)
    (road l30 l17)
    (road l18 l29)
    (road l29 l18)
    (road l2 l17)
    (road l17 l2)
    (road l7 l23)
    (road l23 l7)
    (road l19 l29)
    (road l29 l19)
    (road l25 l36)
    (road l36 l25)
    (road l9 l24)
    (road l24 l9)
    (road l25 l32)
    (road l32 l25)
    (road l17 l18)
    (road l18 l17)
    (road l5 l26)
    (road l26 l5)
    (road l30 l31)
    (road l31 l30)
    (road l18 l20)
    (road l20 l18)
    (road l4 l15)
    (road l15 l4)
    (road l1 l3)
    (road l3 l1)
    (road l3 l9)
    (road l9 l3)
    (road l7 l12)
    (road l12 l7)
    (road l10 l36)
    (road l36 l10)
    (road l10 l22)
    (road l22 l10)
    (road l8 l34)
    (road l34 l8)
    (road l13 l15)
    (road l15 l13)
    (road l8 l15)
    (road l15 l8)
    (road l12 l13)
    (road l13 l12)
    (road l16 l29)
    (road l29 l16)
    (road l23 l27)
    (road l27 l23)
    (road l25 l31)
    (road l31 l25)
    (road l1 l20)
    (road l20 l1)
    (road l15 l25)
    (road l25 l15)
    (road l16 l33)
    (road l33 l16)
    (road l4 l24)
    (road l24 l4)
    (road l24 l25)
    (road l25 l24)
    (road l17 l24)
    (road l24 l17)
    (road l15 l22)
    (road l22 l15)
    (road l18 l36)
    (road l36 l18)
    (road l14 l19)
    (road l19 l14)
    (road l27 l30)
    (road l30 l27)
    (road l7 l16)
    (road l16 l7)
    (road l10 l30)
    (road l30 l10)
    (road l19 l32)
    (road l32 l19)
    (road l8 l21)
    (road l21 l8)
    (road l6 l9)
    (road l9 l6)
    (road l27 l28)
    (road l28 l27)
    (road l2 l24)
    (road l24 l2)
    (road l21 l25)
    (road l25 l21)
    (road l7 l31)
    (road l31 l7)
    (road l24 l36)
    (road l36 l24)
    (road l3 l21)
    (road l21 l3)
    (road l29 l36)
    (road l36 l29)
    (road l27 l32)
    (road l32 l27)
    (road l22 l23)
    (road l23 l22)
    (road l4 l8)
    (road l8 l4)
    (road l23 l36)
    (road l36 l23)
    (road l12 l32)
    (road l32 l12)
    (road l14 l29)
    (road l29 l14)
    (road l24 l28)
    (road l28 l24)
    (road l9 l10)
    (road l10 l9)
    (road l21 l27)
    (road l27 l21)
    (road l11 l14)
    (road l14 l11)
    (road l8 l12)
    (road l12 l8)
    (road l11 l13)
    (road l13 l11)
    (road l21 l34)
    (road l34 l21)
    (road l31 l34)
    (road l34 l31)
    (road l16 l22)
    (road l22 l16)
    (road l14 l32)
    (road l32 l14)
    (road l20 l33)
    (road l33 l20)
    (road l15 l20)
    (road l20 l15)
    (road l20 l21)
    (road l21 l20)
    (road l16 l31)
    (road l31 l16)
    (road l21 l29)
    (road l29 l21)
    (road l16 l34)
    (road l34 l16)
    (road l15 l16)
    (road l16 l15)
    (road l1 l14)
    (road l14 l1)
    (road l9 l15)
    (road l15 l9)
    (road l8 l30)
    (road l30 l8)
    (road l16 l35)
    (road l35 l16)
    (road l3 l28)
    (road l28 l3)
    (road l3 l20)
    (road l20 l3)
    (road l16 l25)
    (road l25 l16)
    (road l19 l26)
    (road l26 l19)
    (road l30 l34)
    (road l34 l30)
    (road l6 l25)
    (road l25 l6)
    (road l11 l30)
    (road l30 l11)
    (road l34 l35)
    (road l35 l34)
    (road l16 l28)
    (road l28 l16)
    (road l23 l34)
    (road l34 l23)
    (road l17 l33)
    (road l33 l17)
    (road l13 l22)
    (road l22 l13)
    (road l10 l20)
    (road l20 l10)
    (road l11 l18)
    (road l18 l11)
    (road l5 l28)
    (road l28 l5)
    (road l13 l29)
    (road l29 l13)
    (road l7 l36)
    (road l36 l7)
    (road l28 l32)
    (road l32 l28)
    (road l16 l26)
    (road l26 l16)
    (road l11 l19)
    (road l19 l11)
    (road l6 l22)
    (road l22 l6)
    (road l3 l12)
    (road l12 l3)
    (road l7 l32)
    (road l32 l7)
    (road l10 l31)
    (road l31 l10)
    (road l10 l28)
    (road l28 l10)
    (road l2 l29)
    (road l29 l2)
    (road l1 l12)
    (road l12 l1)
    (road l22 l33)
    (road l33 l22)
    (road l21 l33)
    (road l33 l21)
    (road l19 l36)
    (road l36 l19)
    (road l8 l29)
    (road l29 l8)
    (road l14 l31)
    (road l31 l14)
    (road l13 l18)
    (road l18 l13)
    (road l6 l32)
    (road l32 l6)
    (road l5 l9)
    (road l9 l5)
    (road l14 l18)
    (road l18 l14)
    (road l10 l25)
    (road l25 l10)
    (road l14 l26)
    (road l26 l14)
    (road l15 l24)
    (road l24 l15)
    (road l10 l21)
    (road l21 l10)
    (road l33 l34)
    (road l34 l33)
    (road l13 l26)
    (road l26 l13)
    (road l16 l20)
    (road l20 l16)
    (road l25 l34)
    (road l34 l25)
    (road l11 l23)
    (road l23 l11)
    (road l35 l36)
    (road l36 l35)
    (road l9 l36)
    (road l36 l9)
    (road l6 l24)
    (road l24 l6)
    (road l5 l8)
    (road l8 l5)
    (road l1 l28)
    (road l28 l1)
    (road l11 l24)
    (road l24 l11)
    (road l13 l19)
    (road l19 l13)
    (road l5 l17)
    (road l17 l5)
    (road l11 l35)
    (road l35 l11)
    (road l9 l34)
    (road l34 l9)
    (road l13 l21)
    (road l21 l13)
    (road l20 l32)
    (road l32 l20)
    (road l1 l8)
    (road l8 l1)
    (road l32 l33)
    (road l33 l32)
    (road l19 l30)
    (road l30 l19)
    (road l16 l21)
    (road l21 l16)
    (road l23 l32)
    (road l32 l23)
    (road l13 l27)
    (road l27 l13)
    (road l7 l22)
    (road l22 l7)
    (road l8 l26)
    (road l26 l8)
    (road l6 l21)
    (road l21 l6)
    (road l5 l34)
    (road l34 l5)
    (road l16 l24)
    (road l24 l16)
    (road l27 l31)
    (road l31 l27)
    (road l14 l36)
    (road l36 l14)
    (road l19 l25)
    (road l25 l19)
    (road l7 l14)
    (road l14 l7)
    (road l4 l17)
    (road l17 l4)
    (road l13 l14)
    (road l14 l13)
    (road l20 l36)
    (road l36 l20)
    (road l27 l34)
    (road l34 l27)
    (road l6 l26)
    (road l26 l6)
    (road l5 l13)
    (road l13 l5)
    (road l9 l20)
    (road l20 l9)
    (road l11 l26)
    (road l26 l11)
    (road l28 l34)
    (road l34 l28)
    (road l3 l24)
    (road l24 l3)
    (road l6 l35)
    (road l35 l6)
    (road l19 l33)
    (road l33 l19)
    (road l21 l31)
    (road l31 l21)
    (road l15 l34)
    (road l34 l15)
    (road l12 l28)
    (road l28 l12)
    (road l3 l18)
    (road l18 l3)
    (road l16 l30)
    (road l30 l16)
    (road l17 l36)
    (road l36 l17)
    (road l13 l33)
    (road l33 l13)
    (road l25 l27)
    (road l27 l25)
    (road l2 l4)
    (road l4 l2)
    (road l2 l3)
    (road l3 l2)
    (road l8 l23)
    (road l23 l8)
    (road l29 l31)
    (road l31 l29)
    (road l28 l36)
    (road l36 l28)
    (road l15 l35)
    (road l35 l15)
    (road l3 l15)
    (road l15 l3)
    (road l26 l29)
    (road l29 l26)
    (road l12 l24)
    (road l24 l12)
    (road l9 l12)
    (road l12 l9)
    (road l1 l9)
    (road l9 l1)
    (road l3 l14)
    (road l14 l3)
    (road l2 l7)
    (road l7 l2)
    (road l24 l30)
    (road l30 l24)
    (road l8 l33)
    (road l33 l8)
    (road l8 l31)
    (road l31 l8)
    (road l28 l30)
    (road l30 l28)
    (road l18 l30)
    (road l30 l18)
    (road l2 l26)
    (road l26 l2)
    (road l9 l35)
    (road l35 l9)
    (road l15 l27)
    (road l27 l15)
    (road l24 l31)
    (road l31 l24)
    (road l3 l30)
    (road l30 l3)
    (road l23 l24)
    (road l24 l23)
    (road l9 l31)
    (road l31 l9)
    (road l22 l27)
    (road l27 l22)
    (road l17 l22)
    (road l22 l17)
    (road l21 l36)
    (road l36 l21)
    (road l4 l30)
    (road l30 l4)
    (road l4 l11)
    (road l11 l4)
    (road l22 l35)
    (road l35 l22)
    (road l5 l12)
    (road l12 l5)
    (road l20 l26)
    (road l26 l20)
    (road l8 l10)
    (road l10 l8)
    (road l6 l12)
    (road l12 l6)
    (road l6 l23)
    (road l23 l6)
    (road l10 l34)
    (road l34 l10)
    (road l8 l27)
    (road l27 l8)
    (road l29 l33)
    (road l33 l29)
    (road l14 l15)
    (road l15 l14)
    (road l2 l25)
    (road l25 l2)
    (road l20 l27)
    (road l27 l20)
    (road l11 l36)
    (road l36 l11)
    (road l4 l27)
    (road l27 l4)
    (road l25 l35)
    (road l35 l25)
    (road l13 l28)
    (road l28 l13)
    (road l4 l18)
    (road l18 l4)
    (road l9 l23)
    (road l23 l9)
    (road l8 l14)
    (road l14 l8)
    (road l12 l19)
    (road l19 l12)
    (road l26 l32)
    (road l32 l26)
    (road l1 l26)
    (road l26 l1)
    (road l1 l21)
    (road l21 l1)
    (road l2 l9)
    (road l9 l2)
    (road l6 l30)
    (road l30 l6)
    (road l22 l34)
    (road l34 l22)
    (road l15 l23)
    (road l23 l15)
    (road l11 l16)
    (road l16 l11)
    (road l5 l36)
    (road l36 l5)
    (road l7 l25)
    (road l25 l7)
    (road l4 l31)
    (road l31 l4)
    (road l4 l12)
    (road l12 l4)
    (road l14 l23)
    (road l23 l14)
    (road l1 l2)
    (road l2 l1)
    (road l14 l34)
    (road l34 l14)
    (road l27 l29)
    (road l29 l27)
    (road l10 l23)
    (road l23 l10)
    (road l1 l27)
    (road l27 l1)
    (road l32 l35)
    (road l35 l32)
    (road l19 l21)
    (road l21 l19)
    (road l10 l27)
    (road l27 l10)
    (road l19 l23)
    (road l23 l19)
    (road l22 l25)
    (road l25 l22)
    (road l20 l25)
    (road l25 l20)
    (road l11 l34)
    (road l34 l11)
    (road l9 l33)
    (road l33 l9)
    (road l28 l33)
    (road l33 l28)
    (road l6 l7)
    (road l7 l6)
    (road l17 l23)
    (road l23 l17)
    (road l30 l32)
    (road l32 l30)
    (road l1 l16)
    (road l16 l1)
    (road l6 l16)
    (road l16 l6)
    (road l26 l31)
    (road l31 l26)
    (road l11 l29)
    (road l29 l11)
    (road l21 l28)
    (road l28 l21)
    (road l32 l34)
    (road l34 l32)
    (road l23 l31)
    (road l31 l23)
    (road l18 l31)
    (road l31 l18)
    (road l6 l27)
    (road l27 l6)
    (road l5 l27)
    (road l27 l5)
    (road l23 l30)
    (road l30 l23)
    (road l3 l22)
    (road l22 l3)
    (road l15 l19)
    (road l19 l15)
    (road l28 l29)
    (road l29 l28)
    (road l9 l25)
    (road l25 l9)
    (road l27 l35)
    (road l35 l27)
    (road l19 l22)
    (road l22 l19)
    (road l12 l33)
    (road l33 l12)
    (road l5 l33)
    (road l33 l5)
    (road l8 l32)
    (road l32 l8)
    (road l12 l31)
    (road l31 l12)
    (road l9 l14)
    (road l14 l9)
    (road l7 l20)
    (road l20 l7)
    (road l20 l24)
    (road l24 l20)
    (road l13 l35)
    (road l35 l13)
    (road l12 l22)
    (road l22 l12)
    (road l14 l27)
    (road l27 l14)
    (road l12 l17)
    (road l17 l12)
    (road l15 l36)
    (road l36 l15)
    (road l4 l26)
    (road l26 l4)
    (road l7 l33)
    (road l33 l7)
    (road l5 l18)
    (road l18 l5)
    (road l17 l32)
    (road l32 l17)
    (road l4 l36)
    (road l36 l4)
    (road l1 l7)
    (road l7 l1)
    (road l27 l33)
    (road l33 l27)
    (road l3 l16)
    (road l16 l3)
    (road l24 l32)
    (road l32 l24)
    (road l17 l19)
    (road l19 l17)
    (road l2 l13)
    (road l13 l2)
    (road l20 l35)
    (road l35 l20)
    (road l2 l11)
    (road l11 l2)
    (road l20 l23)
    (road l23 l20)
    (road l13 l30)
    (road l30 l13)
    (road l22 l26)
    (road l26 l22)
    (road l1 l13)
    (road l13 l1)
    (road l14 l24)
    (road l24 l14)
    (road l15 l21)
    (road l21 l15)
    (road l2 l15)
    (road l15 l2)
    (road l9 l11)
    (road l11 l9)
    (road l13 l34)
    (road l34 l13)
    (road l3 l32)
    (road l32 l3)
    (road l15 l18)
    (road l18 l15)
    (road l10 l11)
    (road l11 l10)
    (road l9 l28)
    (road l28 l9)
    (road l4 l34)
    (road l34 l4)
    (road l4 l9)
    (road l9 l4)
    (road l19 l27)
    (road l27 l19)
    (road l6 l28)
    (road l28 l6)
    (road l21 l24)
    (road l24 l21)
    (road l20 l22)
    (road l22 l20)
    (road l24 l34)
    (road l34 l24)
    (road l12 l25)
    (road l25 l12)
    (road l9 l21)
    (road l21 l9)
    (road l17 l31)
    (road l31 l17)
    (road l12 l18)
    (road l18 l12)
    (road l4 l21)
    (road l21 l4)
    (road l11 l28)
    (road l28 l11)
    (road l12 l23)
    (road l23 l12)
    (road l12 l26)
    (road l26 l12)
    (road l10 l18)
    (road l18 l10)
    (road l1 l10)
    (road l10 l1)
    (road l26 l33)
    (road l33 l26)
    (road l1 l34)
    (road l34 l1)
    (road l17 l26)
    (road l26 l17)
    (road l1 l19)
    (road l19 l1)
    (road l9 l32)
    (road l32 l9)
    (road l1 l33)
    (road l33 l1)
    (road l2 l18)
    (road l18 l2)
    (road l1 l23)
    (road l23 l1)
    (road l12 l15)
    (road l15 l12)
    (road l10 l35)
    (road l35 l10)
    (road l1 l4)
    (road l4 l1)
    (road l16 l36)
    (road l36 l16)
    (road l15 l28)
    (road l28 l15)
    (road l21 l26)
    (road l26 l21)
    (road l24 l27)
    (road l27 l24)
    (road l10 l16)
    (road l16 l10)
    (road l14 l16)
    (road l16 l14)
    )
 (:goal  (and 
    (at p1 l21)
    (at p2 l11)
    (at p3 l17)
    (at p4 l2)
    (at p5 l23)
    (at p6 l28)
    (at p7 l10)
    (at p8 l18)
    (at p9 l16)
    (at p10 l14)
    (at p11 l32)
    (at p12 l12)
    (at p13 l9)
    (at p14 l28)
    (at p15 l30)
    (at p16 l22)
    (at p17 l1)
    (at p18 l32)
    (at p19 l2)
    (at p20 l4)
    (at p21 l10)
    (at p22 l31)
    (at p23 l28)
    (at p24 l4)
    (at p25 l15)
    (at p26 l36)
    (at p27 l16)
    (at p28 l33)
    (at p29 l36)
    (at p30 l20)
    (at p31 l24)
    (at p32 l12)
    (at p33 l6)
    (at p34 l29)
    (at p35 l3)
    (at p36 l3)
    (at p37 l33))))
