;; blocks=5, out_folder=testing_new/easy, instance_id=2, seed=2009

(define (problem blocksworld-02)
 (:domain blocksworld)
 (:objects b1 b2 b3 b4 b5 - object)
 (:init 
    (arm-empty)
    (clear b5)
    (on b5 b1)
    (on b1 b4)
    (on b4 b2)
    (on b2 b3)
    (on-table b3))
 (:goal  (and 
    (clear b4)
    (on b4 b3)
    (on b3 b5)
    (on-table b5)
    (clear b2)
    (on b2 b1)
    (on-table b1))))
