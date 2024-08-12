;; satellites=3, instruments=3, modes=1, directions=4, out_folder=testing_new/easy, instance_id=2, seed=2009

(define (problem satellite-02)
 (:domain satellite)
 (:objects 
    sat1 sat2 sat3 - satellite
    ins1 ins2 ins3 - instrument
    mod1 - mode
    dir1 dir2 dir3 dir4 - direction
    )
 (:init 
    (pointing sat1 dir3)
    (pointing sat2 dir2)
    (pointing sat3 dir4)
    (power_avail sat1)
    (power_avail sat2)
    (power_avail sat3)
    (calibration_target ins1 dir1)
    (calibration_target ins2 dir3)
    (calibration_target ins3 dir2)
    (on_board ins1 sat3)
    (on_board ins2 sat1)
    (on_board ins3 sat2)
    (supports ins2 mod1)
    (supports ins3 mod1)
    (supports ins1 mod1))
 (:goal  (and (pointing sat1 dir2)
   (pointing sat2 dir1)
   (have_image dir2 mod1)
   (have_image dir1 mod1))))
