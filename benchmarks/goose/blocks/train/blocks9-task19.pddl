(define (problem BW-9-7235-19)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 - block)
    (:init
        (handempty)
        (on b1 b3)
        (on b2 b4)
        (on b3 b7)
        (on-table b4)
        (on b5 b9)
        (on-table b6)
        (on-table b7)
        (on-table b8)
        (on b9 b1)
        (clear b2)
        (clear b5)
        (clear b6)
        (clear b8)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b5)
            (on b3 b6)
            (on b4 b3)
            (on b5 b1)
            (on b6 b2)
            (on-table b7)
            (on b8 b7)
            (on b9 b4)
        )
    )
)