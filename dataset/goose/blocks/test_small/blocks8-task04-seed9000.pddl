(define (problem BW-8-9000-4)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 - block)
    (:init
        (handempty)
        (on-table b1)
        (on b2 b8)
        (on b3 b5)
        (on-table b4)
        (on b5 b1)
        (on b6 b2)
        (on b7 b3)
        (on b8 b4)
        (clear b6)
        (clear b7)
    )
    (:goal
        (and
            (on b1 b2)
            (on-table b2)
            (on b3 b5)
            (on b4 b6)
            (on b5 b7)
            (on b6 b1)
            (on-table b7)
            (on b8 b4)
        )
    )
)