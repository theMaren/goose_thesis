(define (problem BW-13-2654-20)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 - block)
    (:init
        (handempty)
        (on b1 b9)
        (on b2 b11)
        (on b3 b2)
        (on b4 b8)
        (on-table b5)
        (on-table b6)
        (on b7 b3)
        (on b8 b1)
        (on b9 b13)
        (on-table b10)
        (on b11 b6)
        (on b12 b7)
        (on-table b13)
        (clear b4)
        (clear b5)
        (clear b10)
        (clear b12)
    )
    (:goal
        (and
            (on b1 b10)
            (on b2 b8)
            (on-table b3)
            (on b4 b13)
            (on b5 b7)
            (on-table b6)
            (on b7 b12)
            (on-table b8)
            (on-table b9)
            (on-table b10)
            (on b11 b3)
            (on b12 b1)
            (on b13 b6)
        )
    )
)