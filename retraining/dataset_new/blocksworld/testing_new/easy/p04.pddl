;; blocks=7, out_folder=testing_new/easy, instance_id=4, seed=2011

(define (problem blocksworld-04)
 (:domain blocksworld)
 (:objects b1 b2 b3 b4 b5 b6 b7 - object)
 (:init 
    (arm-empty)
    (clear b4)
    (on-table b4)
    (clear b6)
    (on b6 b1)
    (on-table b1)
    (clear b3)
    (on b3 b2)
    (on b2 b7)
    (on b7 b5)
    (on-table b5))
 (:goal  (and 
    (clear b5)
    (on b5 b3)
    (on b3 b4)
    (on b4 b2)
    (on-table b2)
    (clear b7)
    (on b7 b1)
    (on b1 b6)
    (on-table b6))))
