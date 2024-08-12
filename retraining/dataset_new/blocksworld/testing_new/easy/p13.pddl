;; blocks=15, out_folder=testing_new/easy, instance_id=13, seed=2020

(define (problem blocksworld-13)
 (:domain blocksworld)
 (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 - object)
 (:init 
    (arm-empty)
    (clear b13)
    (on b13 b7)
    (on b7 b9)
    (on b9 b1)
    (on b1 b2)
    (on b2 b5)
    (on b5 b4)
    (on b4 b6)
    (on b6 b14)
    (on b14 b12)
    (on b12 b8)
    (on b8 b11)
    (on b11 b3)
    (on b3 b15)
    (on-table b15)
    (clear b10)
    (on-table b10))
 (:goal  (and 
    (clear b12)
    (on b12 b15)
    (on b15 b13)
    (on b13 b3)
    (on b3 b14)
    (on b14 b10)
    (on-table b10)
    (clear b7)
    (on b7 b5)
    (on b5 b11)
    (on b11 b9)
    (on b9 b2)
    (on b2 b4)
    (on b4 b6)
    (on b6 b1)
    (on b1 b8)
    (on-table b8))))
