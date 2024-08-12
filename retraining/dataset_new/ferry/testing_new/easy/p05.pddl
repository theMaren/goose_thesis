;; cars=4, locations=6, out_folder=testing_new/easy, instance_id=5, seed=2012

(define (problem ferry-05)
 (:domain ferry)
 (:objects 
    car1 car2 car3 car4 - car
    loc1 loc2 loc3 loc4 loc5 loc6 - location
 )
 (:init 
    (empty-ferry)
    (at-ferry loc1)
    (at car1 loc4)
    (at car2 loc3)
    (at car3 loc2)
    (at car4 loc3)
)
 (:goal  (and (at car1 loc6)
   (at car2 loc1)
   (at car3 loc6)
   (at car4 loc6))))
