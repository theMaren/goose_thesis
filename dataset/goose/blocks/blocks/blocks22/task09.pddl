(define (problem BW-22-1-9)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 - block)
    (:init
        (handempty)
        (on-table b1)
        (on b2 b19)
        (on b3 b5)
        (on b4 b2)
        (on b5 b20)
        (on b6 b4)
        (on-table b7)
        (on b8 b16)
        (on b9 b12)
        (on-table b10)
        (on b11 b13)
        (on b12 b17)
        (on-table b13)
        (on-table b14)
        (on b15 b18)
        (on b16 b6)
        (on b17 b21)
        (on b18 b9)
        (on b19 b10)
        (on-table b20)
        (on b21 b7)
        (on b22 b11)
        (clear b1)
        (clear b3)
        (clear b8)
        (clear b14)
        (clear b15)
        (clear b22)
    )
    (:goal
        (and
            (on b1 b11)
            (on b2 b21)
            (on b3 b8)
            (on b4 b1)
            (on b5 b15)
            (on b6 b9)
            (on b7 b18)
            (on b8 b10)
            (on b9 b3)
            (on-table b10)
            (on-table b11)
            (on b12 b5)
            (on b13 b4)
            (on b14 b13)
            (on b15 b22)
            (on b16 b6)
            (on-table b17)
            (on b18 b20)
            (on b19 b14)
            (on-table b20)
            (on b21 b16)
            (on b22 b2)
        )
    )
)