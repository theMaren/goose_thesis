(define (problem ferry-l5-c6)
(:domain ferry)
(:objects l0 l1 l2 l3 l4 
          c0 c1 c2 c3 c4 c5 
)
(:init
(location l0)
(location l1)
(location l2)
(location l3)
(location l4)
(car c0)
(car c1)
(car c2)
(car c3)
(car c4)
(car c5)
(not-eq l0 l1)
(not-eq l1 l0)
(not-eq l0 l2)
(not-eq l2 l0)
(not-eq l0 l3)
(not-eq l3 l0)
(not-eq l0 l4)
(not-eq l4 l0)
(not-eq l1 l2)
(not-eq l2 l1)
(not-eq l1 l3)
(not-eq l3 l1)
(not-eq l1 l4)
(not-eq l4 l1)
(not-eq l2 l3)
(not-eq l3 l2)
(not-eq l2 l4)
(not-eq l4 l2)
(not-eq l3 l4)
(not-eq l4 l3)
(empty-ferry)
(at c0 l0)
(at c1 l4)
(at c2 l3)
(at c3 l0)
(at c4 l1)
(at c5 l3)
(at-ferry l4)
)
(:goal
(and
(at c0 l4)
(at c1 l2)
(at c2 l0)
(at c3 l2)
(at c4 l2)
(at c5 l4)
)
)
)
