;; cars=7, locations=8, out_folder=testing_new/easy, instance_id=10, seed=2017

(define (problem ferry-10)
 (:domain ferry)
 (:objects 
    car1 car2 car3 car4 car5 car6 car7 - car
    loc1 loc2 loc3 loc4 loc5 loc6 loc7 loc8 - location
 )
 (:init 
    (empty-ferry)
    (at-ferry loc4)
    (at car1 loc8)
    (at car2 loc6)
    (at car3 loc6)
    (at car4 loc4)
    (at car5 loc2)
    (at car6 loc8)
    (at car7 loc2)
)
 (:goal  (and (at car1 loc2)
   (at car2 loc5)
   (at car3 loc4)
   (at car4 loc7)
   (at car5 loc6)
   (at car6 loc6)
   (at car7 loc3))))
