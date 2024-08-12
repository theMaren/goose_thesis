;; cars=3, locations=6, out_folder=testing_new/easy, instance_id=4, seed=2011

(define (problem ferry-04)
 (:domain ferry)
 (:objects 
    car1 car2 car3 - car
    loc1 loc2 loc3 loc4 loc5 loc6 - location
 )
 (:init 
    (empty-ferry)
    (at-ferry loc5)
    (at car1 loc5)
    (at car2 loc6)
    (at car3 loc2)
)
 (:goal  (and (at car1 loc3)
   (at car2 loc2)
   (at car3 loc4))))
