(define (problem BW-12-9546-14)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 - block)
    (:init
        (handempty)
        (on b1 b12)
        (on-table b2)
        (on-table b3)
        (on b4 b10)
        (on b5 b8)
        (on-table b6)
        (on b7 b3)
        (on b8 b11)
        (on b9 b1)
        (on b10 b6)
        (on b11 b9)
        (on b12 b4)
        (clear b2)
        (clear b5)
        (clear b7)
    )
    (:goal
        (and
            (on b1 b11)
            (on b2 b9)
            (on b3 b8)
            (on b4 b1)
            (on b5 b10)
            (on-table b6)
            (on-table b7)
            (on b8 b7)
            (on b9 b6)
            (on b10 b3)
            (on-table b11)
            (on b12 b5)
        )
    )
)