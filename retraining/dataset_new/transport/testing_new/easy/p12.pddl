;; vehicles=4, packages=6, locations=9, max_capacity=2, out_folder=testing_new/easy, instance_id=12, seed=2019

(define (problem transport-12)
 (:domain transport)
 (:objects 
    v1 v2 v3 v4 - vehicle
    p1 p2 p3 p4 p5 p6 - package
    l1 l2 l3 l4 l5 l6 l7 l8 l9 - location
    c0 c1 c2 - size
    )
 (:init (capacity v1 c1)
    (capacity v2 c1)
    (capacity v3 c2)
    (capacity v4 c1)
    (capacity-predecessor c0 c1)
    (capacity-predecessor c1 c2)
    (at p1 l4)
    (at p2 l4)
    (at p3 l6)
    (at p4 l5)
    (at p5 l7)
    (at p6 l4)
    (at v1 l7)
    (at v2 l8)
    (at v3 l2)
    (at v4 l6)
    (road l6 l2)
    (road l8 l4)
    (road l9 l1)
    (road l4 l9)
    (road l3 l2)
    (road l5 l7)
    (road l2 l3)
    (road l9 l5)
    (road l5 l2)
    (road l2 l6)
    (road l4 l8)
    (road l5 l9)
    (road l7 l5)
    (road l2 l5)
    (road l1 l9)
    (road l9 l4)
    (road l1 l8)
    (road l8 l1)
    (road l3 l8)
    (road l8 l3)
    (road l7 l8)
    (road l8 l7)
    (road l1 l2)
    (road l2 l1)
    (road l6 l8)
    (road l8 l6)
    )
 (:goal  (and 
    (at p1 l6)
    (at p2 l1)
    (at p3 l3)
    (at p4 l6)
    (at p5 l2)
    (at p6 l3))))
