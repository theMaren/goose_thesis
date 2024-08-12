;; cars=10, locations=9, out_folder=testing_new/easy, instance_id=14, seed=2021

(define (problem ferry-14)
 (:domain ferry)
 (:objects 
    car1 car2 car3 car4 car5 car6 car7 car8 car9 car10 - car
    loc1 loc2 loc3 loc4 loc5 loc6 loc7 loc8 loc9 - location
 )
 (:init 
    (empty-ferry)
    (at-ferry loc7)
    (at car1 loc9)
    (at car2 loc5)
    (at car3 loc4)
    (at car4 loc1)
    (at car5 loc8)
    (at car6 loc8)
    (at car7 loc2)
    (at car8 loc6)
    (at car9 loc5)
    (at car10 loc5)
)
 (:goal  (and (at car1 loc8)
   (at car2 loc2)
   (at car3 loc3)
   (at car4 loc4)
   (at car5 loc1)
   (at car6 loc2)
   (at car7 loc6)
   (at car8 loc1)
   (at car9 loc9)
   (at car10 loc2))))
