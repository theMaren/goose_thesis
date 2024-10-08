;; cars=6, locations=7, out_folder=testing_new/easy, instance_id=8, seed=2015

(define (problem ferry-08)
 (:domain ferry)
 (:objects 
    car1 car2 car3 car4 car5 car6 - car
    loc1 loc2 loc3 loc4 loc5 loc6 loc7 - location
 )
 (:init 
    (empty-ferry)
    (at-ferry loc6)
    (at car1 loc1)
    (at car2 loc6)
    (at car3 loc6)
    (at car4 loc1)
    (at car5 loc2)
    (at car6 loc6)
)
 (:goal  (and (at car1 loc7)
   (at car2 loc7)
   (at car3 loc1)
   (at car4 loc7)
   (at car5 loc6)
   (at car6 loc4))))
