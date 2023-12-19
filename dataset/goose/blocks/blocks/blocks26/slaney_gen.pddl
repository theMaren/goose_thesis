

(define (problem BW-26-1-1)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26)
    (:init
        (on b1 b22)
        (on-table b2)
        (on b3 b10)
        (on b4 b6)
        (on-table b5)
        (on-table b6)
        (on b7 b14)
        (on b8 b25)
        (on b9 b1)
        (on-table b10)
        (on b11 b24)
        (on b12 b9)
        (on-table b13)
        (on-table b14)
        (on b15 b23)
        (on b16 b13)
        (on b17 b7)
        (on-table b18)
        (on b19 b18)
        (on b20 b2)
        (on b21 b17)
        (on b22 b8)
        (on b23 b11)
        (on b24 b4)
        (on-table b25)
        (on b26 b12)
        (clear b3)
        (clear b5)
        (clear b15)
        (clear b16)
        (clear b19)
        (clear b20)
        (clear b21)
        (clear b26)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b17)
            (on b3 b9)
            (on b4 b23)
            (on b5 b6)
            (on b6 b10)
            (on b7 b25)
            (on b8 b3)
            (on b9 b24)
            (on-table b10)
            (on-table b11)
            (on b12 b21)
            (on b13 b19)
            (on b14 b12)
            (on b15 b26)
            (on b16 b7)
            (on b17 b11)
            (on b18 b1)
            (on b19 b18)
            (on b20 b5)
            (on b21 b20)
            (on b22 b2)
            (on b23 b8)
            (on b24 b15)
            (on b25 b22)
            (on b26 b16)
        )
    )
)


(define (problem BW-26-1-2)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26)
    (:init
        (on b1 b17)
        (on b2 b4)
        (on b3 b10)
        (on-table b4)
        (on-table b5)
        (on b6 b25)
        (on b7 b9)
        (on b8 b18)
        (on b9 b11)
        (on b10 b24)
        (on b11 b1)
        (on b12 b22)
        (on b13 b26)
        (on b14 b6)
        (on b15 b16)
        (on b16 b19)
        (on b17 b3)
        (on b18 b13)
        (on b19 b21)
        (on b20 b12)
        (on b21 b8)
        (on-table b22)
        (on b23 b2)
        (on b24 b23)
        (on b25 b7)
        (on b26 b5)
        (clear b14)
        (clear b15)
        (clear b20)
    )
    (:goal
        (and
            (on b1 b17)
            (on b2 b6)
            (on b3 b18)
            (on b4 b3)
            (on b5 b19)
            (on b6 b20)
            (on b7 b11)
            (on-table b8)
            (on b9 b15)
            (on-table b10)
            (on b11 b8)
            (on b12 b16)
            (on b13 b22)
            (on b14 b24)
            (on b15 b26)
            (on-table b16)
            (on b17 b10)
            (on b18 b2)
            (on b19 b25)
            (on b20 b21)
            (on b21 b14)
            (on b22 b7)
            (on b23 b4)
            (on b24 b13)
            (on-table b25)
            (on b26 b1)
        )
    )
)


(define (problem BW-26-1-3)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26)
    (:init
        (on b1 b15)
        (on b2 b12)
        (on-table b3)
        (on b4 b24)
        (on b5 b20)
        (on b6 b3)
        (on b7 b17)
        (on b8 b16)
        (on b9 b18)
        (on b10 b2)
        (on b11 b21)
        (on b12 b7)
        (on b13 b25)
        (on b14 b10)
        (on b15 b4)
        (on b16 b26)
        (on-table b17)
        (on b18 b13)
        (on b19 b14)
        (on b20 b6)
        (on b21 b22)
        (on b22 b9)
        (on b23 b5)
        (on-table b24)
        (on b25 b1)
        (on b26 b11)
        (clear b8)
        (clear b19)
        (clear b23)
    )
    (:goal
        (and
            (on b1 b23)
            (on b2 b15)
            (on b3 b17)
            (on b4 b2)
            (on b5 b26)
            (on b6 b9)
            (on b7 b21)
            (on b8 b1)
            (on b9 b4)
            (on b10 b24)
            (on b11 b25)
            (on-table b12)
            (on b13 b18)
            (on b14 b5)
            (on b15 b7)
            (on b16 b6)
            (on b17 b20)
            (on b18 b11)
            (on b19 b16)
            (on-table b20)
            (on b21 b13)
            (on b22 b14)
            (on b23 b19)
            (on b24 b22)
            (on b25 b3)
            (on b26 b12)
        )
    )
)


(define (problem BW-26-1-4)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26)
    (:init
        (on b1 b9)
        (on b2 b3)
        (on b3 b24)
        (on b4 b10)
        (on b5 b16)
        (on-table b6)
        (on b7 b1)
        (on b8 b21)
        (on b9 b5)
        (on b10 b6)
        (on-table b11)
        (on b12 b15)
        (on b13 b23)
        (on b14 b7)
        (on b15 b18)
        (on b16 b19)
        (on b17 b14)
        (on b18 b13)
        (on b19 b2)
        (on b20 b11)
        (on b21 b25)
        (on-table b22)
        (on-table b23)
        (on b24 b26)
        (on b25 b4)
        (on b26 b8)
        (clear b12)
        (clear b17)
        (clear b20)
        (clear b22)
    )
    (:goal
        (and
            (on b1 b20)
            (on b2 b13)
            (on-table b3)
            (on b4 b2)
            (on b5 b1)
            (on b6 b14)
            (on b7 b19)
            (on b8 b11)
            (on b9 b4)
            (on b10 b24)
            (on-table b11)
            (on-table b12)
            (on b13 b10)
            (on b14 b12)
            (on b15 b8)
            (on-table b16)
            (on-table b17)
            (on b18 b3)
            (on-table b19)
            (on b20 b22)
            (on b21 b25)
            (on b22 b26)
            (on b23 b21)
            (on-table b24)
            (on b25 b15)
            (on b26 b18)
        )
    )
)


(define (problem BW-26-1-5)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26)
    (:init
        (on b1 b5)
        (on b2 b22)
        (on-table b3)
        (on b4 b13)
        (on b5 b11)
        (on b6 b16)
        (on b7 b10)
        (on b8 b21)
        (on b9 b18)
        (on-table b10)
        (on-table b11)
        (on b12 b14)
        (on b13 b19)
        (on b14 b4)
        (on-table b15)
        (on b16 b24)
        (on b17 b2)
        (on b18 b25)
        (on b19 b23)
        (on b20 b26)
        (on b21 b7)
        (on-table b22)
        (on-table b23)
        (on b24 b9)
        (on b25 b12)
        (on b26 b8)
        (clear b1)
        (clear b3)
        (clear b6)
        (clear b15)
        (clear b17)
        (clear b20)
    )
    (:goal
        (and
            (on b1 b12)
            (on b2 b23)
            (on b3 b6)
            (on b4 b15)
            (on b5 b4)
            (on b6 b16)
            (on b7 b11)
            (on-table b8)
            (on b9 b20)
            (on b10 b21)
            (on b11 b1)
            (on b12 b2)
            (on b13 b24)
            (on-table b14)
            (on b15 b26)
            (on b16 b14)
            (on-table b17)
            (on b18 b5)
            (on b19 b13)
            (on b20 b10)
            (on b21 b3)
            (on b22 b17)
            (on b23 b18)
            (on-table b24)
            (on b25 b19)
            (on b26 b25)
        )
    )
)


(define (problem BW-26-1-6)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26)
    (:init
        (on-table b1)
        (on-table b2)
        (on b3 b24)
        (on-table b4)
        (on b5 b1)
        (on b6 b23)
        (on-table b7)
        (on b8 b14)
        (on-table b9)
        (on b10 b12)
        (on b11 b20)
        (on b12 b18)
        (on b13 b9)
        (on b14 b6)
        (on b15 b19)
        (on-table b16)
        (on b17 b26)
        (on b18 b16)
        (on b19 b10)
        (on b20 b4)
        (on b21 b22)
        (on b22 b2)
        (on b23 b25)
        (on b24 b15)
        (on b25 b7)
        (on b26 b21)
        (clear b3)
        (clear b5)
        (clear b8)
        (clear b11)
        (clear b13)
        (clear b17)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b15)
            (on b3 b9)
            (on-table b4)
            (on b5 b24)
            (on b6 b14)
            (on b7 b18)
            (on b8 b6)
            (on b9 b22)
            (on b10 b1)
            (on b11 b13)
            (on b12 b19)
            (on b13 b25)
            (on b14 b20)
            (on b15 b8)
            (on-table b16)
            (on b17 b5)
            (on b18 b4)
            (on b19 b21)
            (on b20 b11)
            (on b21 b16)
            (on b22 b17)
            (on-table b23)
            (on b24 b26)
            (on b25 b3)
            (on b26 b23)
        )
    )
)


(define (problem BW-26-1-7)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26)
    (:init
        (on b1 b12)
        (on-table b2)
        (on b3 b23)
        (on b4 b9)
        (on b5 b6)
        (on b6 b7)
        (on-table b7)
        (on b8 b26)
        (on b9 b17)
        (on-table b10)
        (on b11 b16)
        (on b12 b19)
        (on b13 b20)
        (on b14 b5)
        (on b15 b8)
        (on b16 b14)
        (on b17 b11)
        (on b18 b4)
        (on b19 b13)
        (on b20 b15)
        (on b21 b1)
        (on b22 b24)
        (on-table b23)
        (on b24 b18)
        (on b25 b10)
        (on-table b26)
        (clear b2)
        (clear b3)
        (clear b21)
        (clear b22)
        (clear b25)
    )
    (:goal
        (and
            (on-table b1)
            (on-table b2)
            (on b3 b12)
            (on b4 b18)
            (on b5 b13)
            (on-table b6)
            (on b7 b22)
            (on-table b8)
            (on b9 b19)
            (on b10 b5)
            (on b11 b9)
            (on b12 b24)
            (on b13 b2)
            (on b14 b7)
            (on b15 b16)
            (on b16 b10)
            (on-table b17)
            (on b18 b8)
            (on b19 b21)
            (on b20 b23)
            (on b21 b26)
            (on-table b22)
            (on b23 b6)
            (on b24 b15)
            (on b25 b4)
            (on b26 b17)
        )
    )
)


(define (problem BW-26-1-8)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26)
    (:init
        (on b1 b20)
        (on b2 b25)
        (on b3 b11)
        (on b4 b14)
        (on b5 b15)
        (on b6 b18)
        (on b7 b10)
        (on-table b8)
        (on b9 b19)
        (on b10 b5)
        (on b11 b24)
        (on-table b12)
        (on b13 b7)
        (on-table b14)
        (on b15 b17)
        (on b16 b2)
        (on-table b17)
        (on b18 b13)
        (on-table b19)
        (on b20 b16)
        (on-table b21)
        (on b22 b3)
        (on b23 b9)
        (on b24 b1)
        (on b25 b21)
        (on b26 b6)
        (clear b4)
        (clear b8)
        (clear b12)
        (clear b22)
        (clear b23)
        (clear b26)
    )
    (:goal
        (and
            (on b1 b8)
            (on b2 b6)
            (on b3 b12)
            (on b4 b9)
            (on b5 b20)
            (on b6 b14)
            (on-table b7)
            (on b8 b16)
            (on b9 b25)
            (on-table b10)
            (on b11 b3)
            (on b12 b4)
            (on b13 b11)
            (on b14 b22)
            (on b15 b18)
            (on b16 b23)
            (on-table b17)
            (on b18 b26)
            (on b19 b10)
            (on-table b20)
            (on b21 b1)
            (on-table b22)
            (on b23 b17)
            (on-table b24)
            (on b25 b24)
            (on b26 b19)
        )
    )
)


(define (problem BW-26-1-9)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26)
    (:init
        (on b1 b24)
        (on b2 b20)
        (on-table b3)
        (on-table b4)
        (on-table b5)
        (on b6 b7)
        (on b7 b9)
        (on b8 b1)
        (on b9 b13)
        (on b10 b5)
        (on b11 b21)
        (on b12 b25)
        (on-table b13)
        (on b14 b18)
        (on b15 b3)
        (on b16 b11)
        (on b17 b23)
        (on-table b18)
        (on b19 b26)
        (on b20 b6)
        (on b21 b15)
        (on b22 b16)
        (on-table b23)
        (on b24 b17)
        (on b25 b4)
        (on b26 b10)
        (clear b2)
        (clear b8)
        (clear b12)
        (clear b14)
        (clear b19)
        (clear b22)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b9)
            (on b3 b10)
            (on b4 b12)
            (on-table b5)
            (on b6 b5)
            (on b7 b19)
            (on-table b8)
            (on b9 b25)
            (on b10 b15)
            (on b11 b16)
            (on b12 b18)
            (on b13 b11)
            (on b14 b22)
            (on b15 b4)
            (on b16 b23)
            (on b17 b7)
            (on-table b18)
            (on b19 b6)
            (on-table b20)
            (on b21 b14)
            (on b22 b13)
            (on b23 b20)
            (on-table b24)
            (on b25 b3)
            (on b26 b1)
        )
    )
)


(define (problem BW-26-1-10)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26)
    (:init
        (on b1 b15)
        (on b2 b10)
        (on b3 b8)
        (on b4 b2)
        (on b5 b24)
        (on-table b6)
        (on b7 b25)
        (on b8 b7)
        (on-table b9)
        (on-table b10)
        (on-table b11)
        (on b12 b18)
        (on b13 b1)
        (on-table b14)
        (on-table b15)
        (on b16 b17)
        (on b17 b11)
        (on b18 b21)
        (on-table b19)
        (on b20 b16)
        (on b21 b23)
        (on b22 b14)
        (on b23 b6)
        (on b24 b3)
        (on b25 b19)
        (on b26 b13)
        (clear b4)
        (clear b5)
        (clear b9)
        (clear b12)
        (clear b20)
        (clear b22)
        (clear b26)
    )
    (:goal
        (and
            (on-table b1)
            (on-table b2)
            (on b3 b23)
            (on-table b4)
            (on b5 b17)
            (on b6 b16)
            (on b7 b26)
            (on-table b8)
            (on b9 b4)
            (on b10 b19)
            (on b11 b9)
            (on b12 b13)
            (on b13 b14)
            (on b14 b8)
            (on b15 b6)
            (on b16 b25)
            (on-table b17)
            (on-table b18)
            (on b19 b12)
            (on b20 b21)
            (on-table b21)
            (on b22 b10)
            (on b23 b7)
            (on-table b24)
            (on b25 b5)
            (on b26 b24)
        )
    )
)
