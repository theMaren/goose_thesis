;; cars=10, locations=20, out_folder=testing_new/medium, instance_id=1, seed=2008

(define (problem ferry-01)
 (:domain ferry)
 (:objects 
    car1 car2 car3 car4 car5 car6 car7 car8 car9 car10 - car
    loc1 loc2 loc3 loc4 loc5 loc6 loc7 loc8 loc9 loc10 loc11 loc12 loc13 loc14 loc15 loc16 loc17 loc18 loc19 loc20 - location
 )
 (:init 
    (empty-ferry)
    (at-ferry loc5)
    (at car1 loc18)
    (at car2 loc13)
    (at car3 loc4)
    (at car4 loc2)
    (at car5 loc13)
    (at car6 loc13)
    (at car7 loc12)
    (at car8 loc5)
    (at car9 loc11)
    (at car10 loc15)
)
 (:goal  (and (at car1 loc8)
   (at car2 loc8)
   (at car3 loc19)
   (at car4 loc13)
   (at car5 loc20)
   (at car6 loc19)
   (at car7 loc6)
   (at car8 loc10)
   (at car9 loc5)
   (at car10 loc14))))
