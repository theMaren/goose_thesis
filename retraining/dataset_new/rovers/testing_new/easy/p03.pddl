;; rovers=1, waypoints=4, cameras=1, objectives=1, out_folder=testing_new/easy, instance_id=3, seed=2010

(define (problem rover-03)
 (:domain rover)
 (:objects 
    general - lander
    colour high_res low_res - mode
    rover1 - rover
    rover1store - store
    waypoint1 waypoint2 waypoint3 waypoint4 - waypoint
    camera1 - camera
    objective1 - objective)
 (:init 
    (at_lander general waypoint2)
    (at rover1 waypoint4)
    (equipped_for_soil_analysis rover1)
    (equipped_for_rock_analysis rover1)
    (equipped_for_imaging rover1)
    (empty rover1store)
    (store_of rover1store rover1)
    (at_rock_sample waypoint3)
    (at_rock_sample waypoint4)
    (at_soil_sample waypoint1)
    (visible waypoint1 waypoint2)
    (visible waypoint2 waypoint1)
    (visible waypoint4 waypoint1)
    (visible waypoint3 waypoint1)
    (visible waypoint1 waypoint4)
    (visible waypoint1 waypoint3)
    (can_traverse rover1 waypoint1 waypoint2)
    (can_traverse rover1 waypoint2 waypoint1)
    (can_traverse rover1 waypoint4 waypoint1)
    (can_traverse rover1 waypoint3 waypoint1)
    (can_traverse rover1 waypoint1 waypoint4)
    (can_traverse rover1 waypoint1 waypoint3)
    (calibration_target camera1 objective1)
    (on_board camera1 rover1)
    (supports camera1 low_res)
    (supports camera1 high_res)
    (supports camera1 colour)
    (visible_from objective1 waypoint2))
 (:goal  (and 
    (communicated_rock_data waypoint4)
    
    (communicated_image_data objective1 high_res)
    (communicated_image_data objective1 colour))))
