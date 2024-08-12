;; cars=5, locations=7, out_folder=testing_new/easy, instance_id=7, seed=2014

(define (problem ferry-07)
 (:domain ferry)
 (:objects 
    car1 car2 car3 car4 car5 - car
    loc1 loc2 loc3 loc4 loc5 loc6 loc7 - location
 )
 (:init 
    (empty-ferry)
    (at-ferry loc3)
    (at car1 loc5)
    (at car2 loc4)
    (at car3 loc2)
    (at car4 loc3)
    (at car5 loc2)
)
 (:goal  (and (at car1 loc4)
   (at car2 loc5)
   (at car3 loc7)
   (at car4 loc5)
   (at car5 loc1))))
