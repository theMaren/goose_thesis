(define (problem BW-6-4532-25)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 - block)
    (:init
        (handempty)
        (on b1 b6)
        (on b2 b4)
        (on b3 b5)
        (on-table b4)
        (on b5 b2)
        (on b6 b3)
        (clear b1)
    )
    (:goal
        (and
            (on b1 b5)
            (on b2 b1)
            (on b3 b6)
            (on b4 b3)
            (on b5 b4)
            (on-table b6)
        )
    )
)