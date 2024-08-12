;; cars=8, locations=9, out_folder=testing_new/easy, instance_id=12, seed=2019

(define (problem ferry-12)
 (:domain ferry)
 (:objects 
    car1 car2 car3 car4 car5 car6 car7 car8 - car
    loc1 loc2 loc3 loc4 loc5 loc6 loc7 loc8 loc9 - location
 )
 (:init 
    (empty-ferry)
    (at-ferry loc3)
    (at car1 loc4)
    (at car2 loc8)
    (at car3 loc3)
    (at car4 loc4)
    (at car5 loc4)
    (at car6 loc6)
    (at car7 loc5)
    (at car8 loc7)
)
 (:goal  (and (at car1 loc5)
   (at car2 loc7)
   (at car3 loc9)
   (at car4 loc2)
   (at car5 loc7)
   (at car6 loc1)
   (at car7 loc7)
   (at car8 loc6))))
