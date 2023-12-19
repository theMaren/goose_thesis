(define (problem BW-17-1-7)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 - block)
    (:init
        (handempty)
        (on-table b1)
        (on b2 b4)
        (on b3 b12)
        (on b4 b17)
        (on b5 b9)
        (on b6 b8)
        (on b7 b10)
        (on b8 b15)
        (on b9 b1)
        (on b10 b3)
        (on b11 b13)
        (on-table b12)
        (on-table b13)
        (on-table b14)
        (on-table b15)
        (on b16 b11)
        (on b17 b14)
        (clear b2)
        (clear b5)
        (clear b6)
        (clear b7)
        (clear b16)
    )
    (:goal
        (and
            (on b1 b12)
            (on-table b2)
            (on b3 b5)
            (on-table b4)
            (on b5 b15)
            (on b6 b13)
            (on-table b7)
            (on b8 b14)
            (on b9 b8)
            (on b10 b7)
            (on b11 b3)
            (on-table b12)
            (on b13 b16)
            (on b14 b2)
            (on b15 b10)
            (on b16 b9)
            (on-table b17)
        )
    )
)