;; passengers=3, floors=7, out_folder=testing_new/easy, instance_id=8, seed=2015

(define (problem miconic-08)
 (:domain miconic)
 (:objects 
    p1 p2 p3 - passenger
    f1 f2 f3 f4 f5 f6 f7 - floor
    )
 (:init 
    (lift-at f6)
    (origin p1 f1)
    (destin p1 f7)
    (origin p2 f6)
    (destin p2 f1)
    (origin p3 f2)
    (destin p3 f7)
    (above f1 f2)
    (above f1 f3)
    (above f1 f4)
    (above f1 f5)
    (above f1 f6)
    (above f1 f7)
    (above f2 f3)
    (above f2 f4)
    (above f2 f5)
    (above f2 f6)
    (above f2 f7)
    (above f3 f4)
    (above f3 f5)
    (above f3 f6)
    (above f3 f7)
    (above f4 f5)
    (above f4 f6)
    (above f4 f7)
    (above f5 f6)
    (above f5 f7)
    (above f6 f7)
)
 (:goal  (and (served p1)
   (served p2)
   (served p3))))
