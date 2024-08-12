;; cars=14, locations=12, out_folder=testing_new/easy, instance_id=21, seed=2028

(define (problem ferry-21)
 (:domain ferry)
 (:objects 
    car1 car2 car3 car4 car5 car6 car7 car8 car9 car10 car11 car12 car13 car14 - car
    loc1 loc2 loc3 loc4 loc5 loc6 loc7 loc8 loc9 loc10 loc11 loc12 - location
 )
 (:init 
    (empty-ferry)
    (at-ferry loc6)
    (at car1 loc3)
    (at car2 loc12)
    (at car3 loc1)
    (at car4 loc6)
    (at car5 loc6)
    (at car6 loc5)
    (at car7 loc3)
    (at car8 loc11)
    (at car9 loc1)
    (at car10 loc4)
    (at car11 loc5)
    (at car12 loc10)
    (at car13 loc10)
    (at car14 loc6)
)
 (:goal  (and (at car1 loc10)
   (at car2 loc1)
   (at car3 loc11)
   (at car4 loc10)
   (at car5 loc11)
   (at car6 loc10)
   (at car7 loc1)
   (at car8 loc10)
   (at car9 loc5)
   (at car10 loc1)
   (at car11 loc1)
   (at car12 loc5)
   (at car13 loc8)
   (at car14 loc11))))
