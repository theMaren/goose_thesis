;; cars=3, locations=5, out_folder=testing_new/easy, instance_id=3, seed=2010

(define (problem ferry-03)
 (:domain ferry)
 (:objects 
    car1 car2 car3 - car
    loc1 loc2 loc3 loc4 loc5 - location
 )
 (:init 
    (empty-ferry)
    (at-ferry loc2)
    (at car1 loc4)
    (at car2 loc4)
    (at car3 loc2)
)
 (:goal  (and (at car1 loc5)
   (at car2 loc5)
   (at car3 loc4))))
