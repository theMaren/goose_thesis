(define (problem BW-27-1-3)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26 b27 - block)
    (:init
        (handempty)
        (on b1 b5)
        (on b2 b27)
        (on b3 b23)
        (on b4 b11)
        (on b5 b12)
        (on b6 b26)
        (on-table b7)
        (on-table b8)
        (on b9 b20)
        (on b10 b24)
        (on b11 b6)
        (on b12 b22)
        (on b13 b19)
        (on b14 b13)
        (on b15 b10)
        (on b16 b9)
        (on-table b17)
        (on b18 b7)
        (on b19 b15)
        (on b20 b3)
        (on b21 b4)
        (on-table b22)
        (on b23 b25)
        (on b24 b2)
        (on b25 b8)
        (on b26 b14)
        (on b27 b17)
        (clear b1)
        (clear b16)
        (clear b18)
        (clear b21)
    )
    (:goal
        (and
            (on b1 b2)
            (on b2 b14)
            (on b3 b27)
            (on b4 b16)
            (on b5 b26)
            (on-table b6)
            (on-table b7)
            (on b8 b15)
            (on-table b9)
            (on b10 b4)
            (on b11 b23)
            (on-table b12)
            (on b13 b20)
            (on b14 b6)
            (on b15 b24)
            (on b16 b7)
            (on b17 b10)
            (on b18 b25)
            (on b19 b11)
            (on b20 b8)
            (on b21 b9)
            (on b22 b17)
            (on-table b23)
            (on b24 b22)
            (on b25 b13)
            (on-table b26)
            (on b27 b19)
        )
    )
)