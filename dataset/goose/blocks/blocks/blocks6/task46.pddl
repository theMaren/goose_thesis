(define (problem BW-6-4532-46)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 - block)
    (:init
        (handempty)
        (on b1 b5)
        (on-table b2)
        (on b3 b4)
        (on-table b4)
        (on b5 b6)
        (on b6 b2)
        (clear b1)
        (clear b3)
    )
    (:goal
        (and
            (on b1 b5)
            (on b2 b6)
            (on b3 b4)
            (on-table b4)
            (on b5 b2)
            (on b6 b3)
        )
    )
)