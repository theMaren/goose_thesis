;; blocks=11, out_folder=testing_new/easy, instance_id=9, seed=2016

(define (problem blocksworld-09)
 (:domain blocksworld)
 (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 - object)
 (:init 
    (arm-empty)
    (clear b4)
    (on b4 b6)
    (on b6 b10)
    (on b10 b1)
    (on b1 b11)
    (on b11 b3)
    (on b3 b7)
    (on b7 b2)
    (on b2 b5)
    (on b5 b9)
    (on b9 b8)
    (on-table b8))
 (:goal  (and 
    (clear b3)
    (on b3 b1)
    (on b1 b4)
    (on b4 b10)
    (on b10 b5)
    (on b5 b7)
    (on-table b7)
    (clear b9)
    (on b9 b2)
    (on b2 b6)
    (on b6 b11)
    (on b11 b8)
    (on-table b8))))
