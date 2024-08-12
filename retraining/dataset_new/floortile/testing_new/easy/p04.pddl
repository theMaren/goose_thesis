;; rows=3, columns=5, robots=1, out_folder=testing_new/easy, instance_id=4, seed=2011

(define (problem floortile-04)
 (:domain floortile)
 (:objects 
    tile_0_1
    tile_0_2
    tile_0_3
    tile_0_4
    tile_0_5
    tile_1_1
    tile_1_2
    tile_1_3
    tile_1_4
    tile_1_5
    tile_2_1
    tile_2_2
    tile_2_3
    tile_2_4
    tile_2_5
    tile_3_1
    tile_3_2
    tile_3_3
    tile_3_4
    tile_3_5 - tile
    robot1 - robot
    white black - color
)
 (:init 
    (robot-at robot1 tile_2_3)
    (robot-has robot1 white)
    (available-color white)
    (available-color black)
    (clear tile_0_1)
    (clear tile_0_2)
    (clear tile_0_3)
    (clear tile_0_4)
    (clear tile_0_5)
    (clear tile_1_1)
    (clear tile_1_2)
    (clear tile_1_3)
    (clear tile_1_4)
    (clear tile_1_5)
    (clear tile_2_1)
    (clear tile_2_2)
    (clear tile_2_4)
    (clear tile_2_5)
    (clear tile_3_1)
    (clear tile_3_2)
    (clear tile_3_3)
    (clear tile_3_4)
    (clear tile_3_5)
    (up tile_1_1 tile_0_1 )
    (up tile_1_2 tile_0_2 )
    (up tile_1_3 tile_0_3 )
    (up tile_1_4 tile_0_4 )
    (up tile_1_5 tile_0_5 )
    (up tile_2_1 tile_1_1 )
    (up tile_2_2 tile_1_2 )
    (up tile_2_3 tile_1_3 )
    (up tile_2_4 tile_1_4 )
    (up tile_2_5 tile_1_5 )
    (up tile_3_1 tile_2_1 )
    (up tile_3_2 tile_2_2 )
    (up tile_3_3 tile_2_3 )
    (up tile_3_4 tile_2_4 )
    (up tile_3_5 tile_2_5 )
    (down tile_0_1 tile_1_1 )
    (down tile_0_2 tile_1_2 )
    (down tile_0_3 tile_1_3 )
    (down tile_0_4 tile_1_4 )
    (down tile_0_5 tile_1_5 )
    (down tile_1_1 tile_2_1 )
    (down tile_1_2 tile_2_2 )
    (down tile_1_3 tile_2_3 )
    (down tile_1_4 tile_2_4 )
    (down tile_1_5 tile_2_5 )
    (down tile_2_1 tile_3_1 )
    (down tile_2_2 tile_3_2 )
    (down tile_2_3 tile_3_3 )
    (down tile_2_4 tile_3_4 )
    (down tile_2_5 tile_3_5 )
    (left tile_0_1 tile_0_2 )
    (left tile_0_2 tile_0_3 )
    (left tile_0_3 tile_0_4 )
    (left tile_0_4 tile_0_5 )
    (left tile_1_1 tile_1_2 )
    (left tile_1_2 tile_1_3 )
    (left tile_1_3 tile_1_4 )
    (left tile_1_4 tile_1_5 )
    (left tile_2_1 tile_2_2 )
    (left tile_2_2 tile_2_3 )
    (left tile_2_3 tile_2_4 )
    (left tile_2_4 tile_2_5 )
    (left tile_3_1 tile_3_2 )
    (left tile_3_2 tile_3_3 )
    (left tile_3_3 tile_3_4 )
    (left tile_3_4 tile_3_5 )
    (right tile_0_2 tile_0_1 )
    (right tile_0_3 tile_0_2 )
    (right tile_0_4 tile_0_3 )
    (right tile_0_5 tile_0_4 )
    (right tile_1_2 tile_1_1 )
    (right tile_1_3 tile_1_2 )
    (right tile_1_4 tile_1_3 )
    (right tile_1_5 tile_1_4 )
    (right tile_2_2 tile_2_1 )
    (right tile_2_3 tile_2_2 )
    (right tile_2_4 tile_2_3 )
    (right tile_2_5 tile_2_4 )
    (right tile_3_2 tile_3_1 )
    (right tile_3_3 tile_3_2 )
    (right tile_3_4 tile_3_3 )
    (right tile_3_5 tile_3_4 ))
 (:goal  (and 
    (painted tile_1_1 white)
    (painted tile_1_2 black)
    (painted tile_1_3 white)
    (painted tile_1_4 black)
    (painted tile_1_5 white)
    (painted tile_2_1 black)
    (painted tile_2_2 white)
    (painted tile_2_3 black)
    (painted tile_2_4 white)
    (painted tile_2_5 black)
    (painted tile_3_1 white)
    (painted tile_3_2 black)
    (painted tile_3_3 white)
    (painted tile_3_4 black)
    (painted tile_3_5 white))))
