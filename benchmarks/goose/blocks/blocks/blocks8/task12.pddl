(define (problem BW-8-3326-12)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 - block)
    (:init
        (handempty)
        (on b1 b8)
        (on-table b2)
        (on-table b3)
        (on-table b4)
        (on b5 b4)
        (on-table b6)
        (on-table b7)
        (on b8 b7)
        (clear b1)
        (clear b2)
        (clear b3)
        (clear b5)
        (clear b6)
    )
    (:goal
        (and
            (on-table b1)
            (on-table b2)
            (on b3 b4)
            (on b4 b8)
            (on b5 b7)
            (on b6 b5)
            (on-table b7)
            (on b8 b2)
        )
    )
)