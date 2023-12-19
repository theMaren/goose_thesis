(define (problem BW-16-1-1)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 - block)
    (:init
        (handempty)
        (on b1 b8)
        (on b2 b13)
        (on b3 b5)
        (on b4 b15)
        (on b5 b12)
        (on-table b6)
        (on-table b7)
        (on-table b8)
        (on b9 b3)
        (on b10 b9)
        (on b11 b14)
        (on-table b12)
        (on-table b13)
        (on-table b14)
        (on b15 b7)
        (on-table b16)
        (clear b1)
        (clear b2)
        (clear b4)
        (clear b6)
        (clear b10)
        (clear b11)
        (clear b16)
    )
    (:goal
        (and
            (on b1 b11)
            (on-table b2)
            (on b3 b6)
            (on b4 b9)
            (on b5 b4)
            (on b6 b13)
            (on-table b7)
            (on b8 b15)
            (on-table b9)
            (on-table b10)
            (on b11 b8)
            (on b12 b2)
            (on b13 b7)
            (on-table b14)
            (on b15 b12)
            (on b16 b10)
        )
    )
)