(define (problem ferry-l8-c2)
(:domain ferry)
(:objects l0 l1 l2 l3 l4 l5 l6 l7 
          c0 c1 
)
(:init
(location l0)
(location l1)
(location l2)
(location l3)
(location l4)
(location l5)
(location l6)
(location l7)
(car c0)
(car c1)
(not-eq l0 l1)
(not-eq l1 l0)
(not-eq l0 l2)
(not-eq l2 l0)
(not-eq l0 l3)
(not-eq l3 l0)
(not-eq l0 l4)
(not-eq l4 l0)
(not-eq l0 l5)
(not-eq l5 l0)
(not-eq l0 l6)
(not-eq l6 l0)
(not-eq l0 l7)
(not-eq l7 l0)
(not-eq l1 l2)
(not-eq l2 l1)
(not-eq l1 l3)
(not-eq l3 l1)
(not-eq l1 l4)
(not-eq l4 l1)
(not-eq l1 l5)
(not-eq l5 l1)
(not-eq l1 l6)
(not-eq l6 l1)
(not-eq l1 l7)
(not-eq l7 l1)
(not-eq l2 l3)
(not-eq l3 l2)
(not-eq l2 l4)
(not-eq l4 l2)
(not-eq l2 l5)
(not-eq l5 l2)
(not-eq l2 l6)
(not-eq l6 l2)
(not-eq l2 l7)
(not-eq l7 l2)
(not-eq l3 l4)
(not-eq l4 l3)
(not-eq l3 l5)
(not-eq l5 l3)
(not-eq l3 l6)
(not-eq l6 l3)
(not-eq l3 l7)
(not-eq l7 l3)
(not-eq l4 l5)
(not-eq l5 l4)
(not-eq l4 l6)
(not-eq l6 l4)
(not-eq l4 l7)
(not-eq l7 l4)
(not-eq l5 l6)
(not-eq l6 l5)
(not-eq l5 l7)
(not-eq l7 l5)
(not-eq l6 l7)
(not-eq l7 l6)
(empty-ferry)
(at c0 l3)
(at c1 l5)
(at-ferry l4)
)
(:goal
(and
(at c0 l2)
(at c1 l0)
)
)
)
