(define (problem ferry-l16-c16)
(:domain ferry)
(:objects l0 l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 l11 l12 l13 l14 l15 
          c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 
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
(location l8)
(location l9)
(location l10)
(location l11)
(location l12)
(location l13)
(location l14)
(location l15)
(car c0)
(car c1)
(car c2)
(car c3)
(car c4)
(car c5)
(car c6)
(car c7)
(car c8)
(car c9)
(car c10)
(car c11)
(car c12)
(car c13)
(car c14)
(car c15)
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
(not-eq l0 l8)
(not-eq l8 l0)
(not-eq l0 l9)
(not-eq l9 l0)
(not-eq l0 l10)
(not-eq l10 l0)
(not-eq l0 l11)
(not-eq l11 l0)
(not-eq l0 l12)
(not-eq l12 l0)
(not-eq l0 l13)
(not-eq l13 l0)
(not-eq l0 l14)
(not-eq l14 l0)
(not-eq l0 l15)
(not-eq l15 l0)
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
(not-eq l1 l8)
(not-eq l8 l1)
(not-eq l1 l9)
(not-eq l9 l1)
(not-eq l1 l10)
(not-eq l10 l1)
(not-eq l1 l11)
(not-eq l11 l1)
(not-eq l1 l12)
(not-eq l12 l1)
(not-eq l1 l13)
(not-eq l13 l1)
(not-eq l1 l14)
(not-eq l14 l1)
(not-eq l1 l15)
(not-eq l15 l1)
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
(not-eq l2 l8)
(not-eq l8 l2)
(not-eq l2 l9)
(not-eq l9 l2)
(not-eq l2 l10)
(not-eq l10 l2)
(not-eq l2 l11)
(not-eq l11 l2)
(not-eq l2 l12)
(not-eq l12 l2)
(not-eq l2 l13)
(not-eq l13 l2)
(not-eq l2 l14)
(not-eq l14 l2)
(not-eq l2 l15)
(not-eq l15 l2)
(not-eq l3 l4)
(not-eq l4 l3)
(not-eq l3 l5)
(not-eq l5 l3)
(not-eq l3 l6)
(not-eq l6 l3)
(not-eq l3 l7)
(not-eq l7 l3)
(not-eq l3 l8)
(not-eq l8 l3)
(not-eq l3 l9)
(not-eq l9 l3)
(not-eq l3 l10)
(not-eq l10 l3)
(not-eq l3 l11)
(not-eq l11 l3)
(not-eq l3 l12)
(not-eq l12 l3)
(not-eq l3 l13)
(not-eq l13 l3)
(not-eq l3 l14)
(not-eq l14 l3)
(not-eq l3 l15)
(not-eq l15 l3)
(not-eq l4 l5)
(not-eq l5 l4)
(not-eq l4 l6)
(not-eq l6 l4)
(not-eq l4 l7)
(not-eq l7 l4)
(not-eq l4 l8)
(not-eq l8 l4)
(not-eq l4 l9)
(not-eq l9 l4)
(not-eq l4 l10)
(not-eq l10 l4)
(not-eq l4 l11)
(not-eq l11 l4)
(not-eq l4 l12)
(not-eq l12 l4)
(not-eq l4 l13)
(not-eq l13 l4)
(not-eq l4 l14)
(not-eq l14 l4)
(not-eq l4 l15)
(not-eq l15 l4)
(not-eq l5 l6)
(not-eq l6 l5)
(not-eq l5 l7)
(not-eq l7 l5)
(not-eq l5 l8)
(not-eq l8 l5)
(not-eq l5 l9)
(not-eq l9 l5)
(not-eq l5 l10)
(not-eq l10 l5)
(not-eq l5 l11)
(not-eq l11 l5)
(not-eq l5 l12)
(not-eq l12 l5)
(not-eq l5 l13)
(not-eq l13 l5)
(not-eq l5 l14)
(not-eq l14 l5)
(not-eq l5 l15)
(not-eq l15 l5)
(not-eq l6 l7)
(not-eq l7 l6)
(not-eq l6 l8)
(not-eq l8 l6)
(not-eq l6 l9)
(not-eq l9 l6)
(not-eq l6 l10)
(not-eq l10 l6)
(not-eq l6 l11)
(not-eq l11 l6)
(not-eq l6 l12)
(not-eq l12 l6)
(not-eq l6 l13)
(not-eq l13 l6)
(not-eq l6 l14)
(not-eq l14 l6)
(not-eq l6 l15)
(not-eq l15 l6)
(not-eq l7 l8)
(not-eq l8 l7)
(not-eq l7 l9)
(not-eq l9 l7)
(not-eq l7 l10)
(not-eq l10 l7)
(not-eq l7 l11)
(not-eq l11 l7)
(not-eq l7 l12)
(not-eq l12 l7)
(not-eq l7 l13)
(not-eq l13 l7)
(not-eq l7 l14)
(not-eq l14 l7)
(not-eq l7 l15)
(not-eq l15 l7)
(not-eq l8 l9)
(not-eq l9 l8)
(not-eq l8 l10)
(not-eq l10 l8)
(not-eq l8 l11)
(not-eq l11 l8)
(not-eq l8 l12)
(not-eq l12 l8)
(not-eq l8 l13)
(not-eq l13 l8)
(not-eq l8 l14)
(not-eq l14 l8)
(not-eq l8 l15)
(not-eq l15 l8)
(not-eq l9 l10)
(not-eq l10 l9)
(not-eq l9 l11)
(not-eq l11 l9)
(not-eq l9 l12)
(not-eq l12 l9)
(not-eq l9 l13)
(not-eq l13 l9)
(not-eq l9 l14)
(not-eq l14 l9)
(not-eq l9 l15)
(not-eq l15 l9)
(not-eq l10 l11)
(not-eq l11 l10)
(not-eq l10 l12)
(not-eq l12 l10)
(not-eq l10 l13)
(not-eq l13 l10)
(not-eq l10 l14)
(not-eq l14 l10)
(not-eq l10 l15)
(not-eq l15 l10)
(not-eq l11 l12)
(not-eq l12 l11)
(not-eq l11 l13)
(not-eq l13 l11)
(not-eq l11 l14)
(not-eq l14 l11)
(not-eq l11 l15)
(not-eq l15 l11)
(not-eq l12 l13)
(not-eq l13 l12)
(not-eq l12 l14)
(not-eq l14 l12)
(not-eq l12 l15)
(not-eq l15 l12)
(not-eq l13 l14)
(not-eq l14 l13)
(not-eq l13 l15)
(not-eq l15 l13)
(not-eq l14 l15)
(not-eq l15 l14)
(empty-ferry)
(at c0 l7)
(at c1 l6)
(at c2 l9)
(at c3 l3)
(at c4 l1)
(at c5 l15)
(at c6 l10)
(at c7 l12)
(at c8 l9)
(at c9 l13)
(at c10 l10)
(at c11 l11)
(at c12 l2)
(at c13 l11)
(at c14 l3)
(at c15 l6)
(at-ferry l6)
)
(:goal
(and
(at c0 l12)
(at c1 l2)
(at c2 l4)
(at c3 l8)
(at c4 l11)
(at c5 l8)
(at c6 l7)
(at c7 l13)
(at c8 l6)
(at c9 l10)
(at c10 l14)
(at c11 l3)
(at c12 l3)
(at c13 l15)
(at c14 l9)
(at c15 l10)
)
)
)
