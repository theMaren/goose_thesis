( define ( domain openstacks-sequencedstrips-nonADL-nonNegated ) ( :requirements :typing :action-costs ) ( :types order product count ) ( :constants p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31 p32 p33 p34 - product o1 o2 o3 o4 o5 o6 o7 o8 o9 o10 o11 o12 o13 o14 o15 o16 o17 o18 o19 o20 o21 o22 o23 o24 o25 o26 o27 o28 o29 o30 o31 o32 o33 o34 - order ) ( :predicates ( includes ?o - order ?p - product ) ( waiting ?o - order ) ( started ?o - order ) ( shipped ?o - order ) ( made ?p - product ) ( not-made ?p - product ) ( stacks-avail ?s - count ) ( next-count ?s ?ns - count ) ) ( :functions ( total-cost ) - number ) ( :action open-new-stack :parameters ( ?open ?new-open - count ) :precondition ( and ( stacks-avail ?open ) ( next-count ?open ?new-open ) ) :effect ( and ( stacks-avail ?new-open ) ( increase ( total-cost ) 1 ) ) ) ( :action start-order :parameters ( ?o - order ?avail ?new-avail - count ) :precondition ( and ( waiting ?o ) ( stacks-avail ?avail ) ( next-count ?new-avail ?avail ) ) :effect ( and ( started ?o ) ( stacks-avail ?new-avail ) ) ) ( :action make-product-p1 :parameters ( ) :precondition ( and ( not-made p1 ) ( started o18 ) ) :effect ( and ( made p1 ) ) ) ( :action make-product-p2 :parameters ( ) :precondition ( and ( not-made p2 ) ( started o4 ) ( started o18 ) ( started o27 ) ) :effect ( and ( made p2 ) ) ) ( :action make-product-p3 :parameters ( ) :precondition ( and ( not-made p3 ) ( started o2 ) ( started o15 ) ) :effect ( and ( made p3 ) ) ) ( :action make-product-p4 :parameters ( ) :precondition ( and ( not-made p4 ) ( started o6 ) ) :effect ( and ( made p4 ) ) ) ( :action make-product-p5 :parameters ( ) :precondition ( and ( not-made p5 ) ( started o17 ) ( started o22 ) ( started o30 ) ) :effect ( and ( made p5 ) ) ) ( :action make-product-p6 :parameters ( ) :precondition ( and ( not-made p6 ) ( started o15 ) ) :effect ( and ( made p6 ) ) ) ( :action make-product-p7 :parameters ( ) :precondition ( and ( not-made p7 ) ( started o33 ) ) :effect ( and ( made p7 ) ) ) ( :action make-product-p8 :parameters ( ) :precondition ( and ( not-made p8 ) ( started o6 ) ( started o30 ) ) :effect ( and ( made p8 ) ) ) ( :action make-product-p9 :parameters ( ) :precondition ( and ( not-made p9 ) ( started o25 ) ( started o30 ) ( started o31 ) ) :effect ( and ( made p9 ) ) ) ( :action make-product-p10 :parameters ( ) :precondition ( and ( not-made p10 ) ( started o19 ) ) :effect ( and ( made p10 ) ) ) ( :action make-product-p11 :parameters ( ) :precondition ( and ( not-made p11 ) ( started o11 ) ( started o15 ) ) :effect ( and ( made p11 ) ) ) ( :action make-product-p12 :parameters ( ) :precondition ( and ( not-made p12 ) ( started o14 ) ( started o30 ) ) :effect ( and ( made p12 ) ) ) ( :action make-product-p13 :parameters ( ) :precondition ( and ( not-made p13 ) ( started o10 ) ) :effect ( and ( made p13 ) ) ) ( :action make-product-p14 :parameters ( ) :precondition ( and ( not-made p14 ) ( started o5 ) ) :effect ( and ( made p14 ) ) ) ( :action make-product-p15 :parameters ( ) :precondition ( and ( not-made p15 ) ( started o31 ) ) :effect ( and ( made p15 ) ) ) ( :action make-product-p16 :parameters ( ) :precondition ( and ( not-made p16 ) ( started o1 ) ( started o17 ) ( started o34 ) ) :effect ( and ( made p16 ) ) ) ( :action make-product-p17 :parameters ( ) :precondition ( and ( not-made p17 ) ( started o3 ) ( started o10 ) ( started o29 ) ) :effect ( and ( made p17 ) ) ) ( :action make-product-p18 :parameters ( ) :precondition ( and ( not-made p18 ) ( started o23 ) ) :effect ( and ( made p18 ) ) ) ( :action make-product-p19 :parameters ( ) :precondition ( and ( not-made p19 ) ( started o28 ) ) :effect ( and ( made p19 ) ) ) ( :action make-product-p20 :parameters ( ) :precondition ( and ( not-made p20 ) ( started o29 ) ) :effect ( and ( made p20 ) ) ) ( :action make-product-p21 :parameters ( ) :precondition ( and ( not-made p21 ) ( started o8 ) ) :effect ( and ( made p21 ) ) ) ( :action make-product-p22 :parameters ( ) :precondition ( and ( not-made p22 ) ( started o10 ) ) :effect ( and ( made p22 ) ) ) ( :action make-product-p23 :parameters ( ) :precondition ( and ( not-made p23 ) ( started o5 ) ) :effect ( and ( made p23 ) ) ) ( :action make-product-p24 :parameters ( ) :precondition ( and ( not-made p24 ) ( started o7 ) ( started o29 ) ) :effect ( and ( made p24 ) ) ) ( :action make-product-p25 :parameters ( ) :precondition ( and ( not-made p25 ) ( started o17 ) ( started o28 ) ) :effect ( and ( made p25 ) ) ) ( :action make-product-p26 :parameters ( ) :precondition ( and ( not-made p26 ) ( started o8 ) ( started o9 ) ( started o26 ) ( started o32 ) ) :effect ( and ( made p26 ) ) ) ( :action make-product-p27 :parameters ( ) :precondition ( and ( not-made p27 ) ( started o13 ) ( started o32 ) ( started o33 ) ) :effect ( and ( made p27 ) ) ) ( :action make-product-p28 :parameters ( ) :precondition ( and ( not-made p28 ) ( started o8 ) ) :effect ( and ( made p28 ) ) ) ( :action make-product-p29 :parameters ( ) :precondition ( and ( not-made p29 ) ( started o20 ) ( started o29 ) ) :effect ( and ( made p29 ) ) ) ( :action make-product-p30 :parameters ( ) :precondition ( and ( not-made p30 ) ( started o12 ) ( started o21 ) ( started o24 ) ) :effect ( and ( made p30 ) ) ) ( :action make-product-p31 :parameters ( ) :precondition ( and ( not-made p31 ) ( started o3 ) ) :effect ( and ( made p31 ) ) ) ( :action make-product-p32 :parameters ( ) :precondition ( and ( not-made p32 ) ( started o25 ) ) :effect ( and ( made p32 ) ) ) ( :action make-product-p33 :parameters ( ) :precondition ( and ( not-made p33 ) ( started o23 ) ) :effect ( and ( made p33 ) ) ) ( :action make-product-p34 :parameters ( ) :precondition ( and ( not-made p34 ) ( started o16 ) ) :effect ( and ( made p34 ) ) ) ( :action ship-order-o1 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o1 ) ( made p16 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o1 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o2 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o2 ) ( made p3 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o2 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o3 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o3 ) ( made p17 ) ( made p31 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o3 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o4 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o4 ) ( made p2 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o4 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o5 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o5 ) ( made p14 ) ( made p23 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o5 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o6 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o6 ) ( made p4 ) ( made p8 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o6 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o7 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o7 ) ( made p24 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o7 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o8 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o8 ) ( made p21 ) ( made p26 ) ( made p28 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o8 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o9 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o9 ) ( made p26 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o9 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o10 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o10 ) ( made p13 ) ( made p17 ) ( made p22 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o10 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o11 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o11 ) ( made p11 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o11 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o12 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o12 ) ( made p30 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o12 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o13 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o13 ) ( made p27 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o13 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o14 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o14 ) ( made p12 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o14 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o15 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o15 ) ( made p3 ) ( made p6 ) ( made p11 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o15 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o16 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o16 ) ( made p34 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o16 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o17 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o17 ) ( made p5 ) ( made p16 ) ( made p25 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o17 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o18 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o18 ) ( made p1 ) ( made p2 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o18 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o19 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o19 ) ( made p10 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o19 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o20 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o20 ) ( made p29 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o20 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o21 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o21 ) ( made p30 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o21 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o22 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o22 ) ( made p5 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o22 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o23 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o23 ) ( made p18 ) ( made p33 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o23 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o24 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o24 ) ( made p30 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o24 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o25 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o25 ) ( made p9 ) ( made p32 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o25 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o26 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o26 ) ( made p26 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o26 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o27 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o27 ) ( made p2 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o27 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o28 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o28 ) ( made p19 ) ( made p25 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o28 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o29 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o29 ) ( made p17 ) ( made p20 ) ( made p24 ) ( made p29 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o29 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o30 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o30 ) ( made p5 ) ( made p8 ) ( made p9 ) ( made p12 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o30 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o31 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o31 ) ( made p9 ) ( made p15 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o31 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o32 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o32 ) ( made p26 ) ( made p27 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o32 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o33 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o33 ) ( made p7 ) ( made p27 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o33 ) ( stacks-avail ?new-avail ) ) ) ( :action ship-order-o34 :parameters ( ?avail ?new-avail - count ) :precondition ( and ( started o34 ) ( made p16 ) ( stacks-avail ?avail ) ( next-count ?avail ?new-avail ) ) :effect ( and ( shipped o34 ) ( stacks-avail ?new-avail ) ) ) )