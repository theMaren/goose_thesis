(define (problem BW-15-4678-19)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 - block)
    (:init
        (handempty)
        (on-table b1)
        (on b2 b8)
        (on b3 b9)
        (on b4 b10)
        (on b5 b12)
        (on b6 b3)
        (on b7 b13)
        (on b8 b5)
        (on b9 b14)
        (on b10 b11)
        (on-table b11)
        (on b12 b4)
        (on b13 b2)
        (on b14 b7)
        (on-table b15)
        (clear b1)
        (clear b6)
        (clear b15)
    )
    (:goal
        (and
            (on b1 b10)
            (on-table b2)
            (on b3 b5)
            (on b4 b12)
            (on b5 b15)
            (on-table b6)
            (on b7 b14)
            (on b8 b7)
            (on b9 b3)
            (on-table b10)
            (on b11 b4)
            (on b12 b8)
            (on-table b13)
            (on b14 b9)
            (on b15 b6)
        )
    )
)