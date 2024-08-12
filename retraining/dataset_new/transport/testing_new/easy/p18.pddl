;; vehicles=5, packages=9, locations=11, max_capacity=2, out_folder=testing_new/easy, instance_id=18, seed=2025

(define (problem transport-18)
 (:domain transport)
 (:objects 
    v1 v2 v3 v4 v5 - vehicle
    p1 p2 p3 p4 p5 p6 p7 p8 p9 - package
    l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 l11 - location
    c0 c1 c2 - size
    )
 (:init (capacity v1 c1)
    (capacity v2 c2)
    (capacity v3 c1)
    (capacity v4 c1)
    (capacity v5 c2)
    (capacity-predecessor c0 c1)
    (capacity-predecessor c1 c2)
    (at p1 l7)
    (at p2 l10)
    (at p3 l4)
    (at p4 l2)
    (at p5 l7)
    (at p6 l7)
    (at p7 l2)
    (at p8 l1)
    (at p9 l2)
    (at v1 l4)
    (at v2 l7)
    (at v3 l2)
    (at v4 l7)
    (at v5 l1)
    (road l3 l4)
    (road l4 l3)
    (road l5 l4)
    (road l5 l7)
    (road l8 l6)
    (road l9 l11)
    (road l6 l2)
    (road l6 l5)
    (road l6 l8)
    (road l4 l5)
    (road l5 l6)
    (road l4 l11)
    (road l11 l4)
    (road l11 l1)
    (road l1 l11)
    (road l4 l10)
    (road l10 l4)
    (road l11 l9)
    (road l2 l6)
    (road l7 l5)
    (road l4 l8)
    (road l8 l4)
    (road l4 l9)
    (road l9 l4)
    (road l2 l11)
    (road l11 l2)
    (road l1 l3)
    (road l3 l1)
    (road l3 l7)
    (road l7 l3)
    (road l1 l8)
    (road l8 l1)
    (road l2 l10)
    (road l10 l2)
    (road l10 l11)
    (road l11 l10)
    (road l5 l10)
    (road l10 l5)
    (road l4 l6)
    (road l6 l4)
    (road l1 l6)
    (road l6 l1)
    (road l2 l3)
    (road l3 l2)
    (road l3 l9)
    (road l9 l3)
    (road l8 l10)
    (road l10 l8)
    (road l1 l5)
    (road l5 l1)
    )
 (:goal  (and 
    (at p1 l9)
    (at p2 l5)
    (at p3 l10)
    (at p4 l11)
    (at p5 l3)
    (at p6 l1)
    (at p7 l6)
    (at p8 l10)
    (at p9 l9))))
