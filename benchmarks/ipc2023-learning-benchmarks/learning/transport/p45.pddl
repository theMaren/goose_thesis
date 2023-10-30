;; vehicles=4, packages=8, locations=10, max_capacity=2, out_folder=training/easy, instance_id=45, seed=86

(define (problem transport-45)
 (:domain transport)
 (:objects 
    v1 v2 v3 v4 - vehicle
    p1 p2 p3 p4 p5 p6 p7 p8 - package
    l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 - location
    c0 c1 c2 - size
    )
 (:init (capacity v1 c1)
    (capacity v2 c2)
    (capacity v3 c1)
    (capacity v4 c1)
    (capacity-predecessor c0 c1)
    (capacity-predecessor c1 c2)
    (at p1 l9)
    (at p2 l5)
    (at p3 l7)
    (at p4 l2)
    (at p5 l9)
    (at p6 l2)
    (at p7 l7)
    (at p8 l8)
    (at v1 l1)
    (at v2 l10)
    (at v3 l4)
    (at v4 l4)
    (road l7 l4)
    (road l10 l5)
    (road l2 l8)
    (road l4 l1)
    (road l7 l10)
    (road l3 l7)
    (road l10 l7)
    (road l1 l4)
    (road l7 l3)
    (road l7 l9)
    (road l6 l7)
    (road l5 l10)
    (road l7 l6)
    (road l8 l2)
    (road l2 l5)
    (road l9 l7)
    (road l4 l7)
    (road l5 l2)
    (road l6 l10)
    (road l10 l6)
    (road l2 l6)
    (road l6 l2)
    (road l4 l8)
    (road l8 l4)
    (road l2 l4)
    (road l4 l2)
    (road l5 l8)
    (road l8 l5)
    (road l5 l9)
    (road l9 l5)
    (road l1 l7)
    (road l7 l1)
    (road l1 l6)
    (road l6 l1)
    (road l1 l10)
    (road l10 l1)
    (road l1 l9)
    (road l9 l1)
    (road l3 l8)
    (road l8 l3)
    (road l9 l10)
    (road l10 l9)
    (road l1 l2)
    (road l2 l1)
    )
 (:goal  (and 
    (at p1 l8)
    (at p2 l4)
    (at p3 l8)
    (at p4 l1)
    (at p5 l5)
    (at p6 l3)
    (at p7 l6)
    (at p8 l7))))
