;; blocks=6, out_folder=testing_new/easy, instance_id=3, seed=2010

(define (problem blocksworld-03)
 (:domain blocksworld)
 (:objects b1 b2 b3 b4 b5 b6 - object)
 (:init 
    (arm-empty)
    (clear b3)
    (on b3 b6)
    (on b6 b1)
    (on b1 b5)
    (on b5 b4)
    (on b4 b2)
    (on-table b2))
 (:goal  (and 
    (clear b5)
    (on b5 b1)
    (on b1 b6)
    (on b6 b4)
    (on b4 b3)
    (on b3 b2)
    (on-table b2))))
