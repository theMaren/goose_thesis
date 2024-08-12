;; cars=10, locations=10, out_folder=testing_new/easy, instance_id=15, seed=2022

(define (problem ferry-15)
 (:domain ferry)
 (:objects 
    car1 car2 car3 car4 car5 car6 car7 car8 car9 car10 - car
    loc1 loc2 loc3 loc4 loc5 loc6 loc7 loc8 loc9 loc10 - location
 )
 (:init 
    (empty-ferry)
    (at-ferry loc9)
    (at car1 loc5)
    (at car2 loc8)
    (at car3 loc9)
    (at car4 loc5)
    (at car5 loc10)
    (at car6 loc1)
    (at car7 loc9)
    (at car8 loc7)
    (at car9 loc6)
    (at car10 loc1)
)
 (:goal  (and (at car1 loc8)
   (at car2 loc5)
   (at car3 loc8)
   (at car4 loc10)
   (at car5 loc9)
   (at car6 loc2)
   (at car7 loc5)
   (at car8 loc1)
   (at car9 loc7)
   (at car10 loc6))))
