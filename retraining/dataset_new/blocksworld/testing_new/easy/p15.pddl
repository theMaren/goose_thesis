;; blocks=16, out_folder=testing_new/easy, instance_id=15, seed=2022

(define (problem blocksworld-15)
 (:domain blocksworld)
 (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 - object)
 (:init 
    (arm-empty)
    (clear b15)
    (on b15 b2)
    (on b2 b13)
    (on b13 b3)
    (on-table b3)
    (clear b4)
    (on b4 b11)
    (on-table b11)
    (clear b12)
    (on b12 b6)
    (on b6 b7)
    (on b7 b14)
    (on b14 b1)
    (on b1 b16)
    (on b16 b5)
    (on b5 b9)
    (on b9 b8)
    (on-table b8)
    (clear b10)
    (on-table b10))
 (:goal  (and 
    (clear b13)
    (on b13 b9)
    (on b9 b5)
    (on-table b5)
    (clear b1)
    (on b1 b3)
    (on b3 b15)
    (on-table b15)
    (clear b6)
    (on-table b6)
    (clear b12)
    (on b12 b10)
    (on b10 b14)
    (on b14 b16)
    (on b16 b7)
    (on b7 b8)
    (on b8 b11)
    (on b11 b4)
    (on b4 b2)
    (on-table b2))))
