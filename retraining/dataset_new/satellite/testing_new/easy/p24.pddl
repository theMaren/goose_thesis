;; satellites=9, instruments=16, modes=3, directions=9, out_folder=testing_new/easy, instance_id=24, seed=2031

(define (problem satellite-24)
 (:domain satellite)
 (:objects 
    sat1 sat2 sat3 sat4 sat5 sat6 sat7 sat8 sat9 - satellite
    ins1 ins2 ins3 ins4 ins5 ins6 ins7 ins8 ins9 ins10 ins11 ins12 ins13 ins14 ins15 ins16 - instrument
    mod1 mod2 mod3 - mode
    dir1 dir2 dir3 dir4 dir5 dir6 dir7 dir8 dir9 - direction
    )
 (:init 
    (pointing sat1 dir3)
    (pointing sat2 dir6)
    (pointing sat3 dir6)
    (pointing sat4 dir5)
    (pointing sat5 dir8)
    (pointing sat6 dir8)
    (pointing sat7 dir4)
    (pointing sat8 dir4)
    (pointing sat9 dir8)
    (power_avail sat1)
    (power_avail sat2)
    (power_avail sat3)
    (power_avail sat4)
    (power_avail sat5)
    (power_avail sat6)
    (power_avail sat7)
    (power_avail sat8)
    (power_avail sat9)
    (calibration_target ins1 dir5)
    (calibration_target ins2 dir7)
    (calibration_target ins3 dir2)
    (calibration_target ins4 dir9)
    (calibration_target ins5 dir3)
    (calibration_target ins6 dir4)
    (calibration_target ins7 dir7)
    (calibration_target ins8 dir1)
    (calibration_target ins9 dir5)
    (calibration_target ins10 dir4)
    (calibration_target ins11 dir6)
    (calibration_target ins12 dir1)
    (calibration_target ins13 dir7)
    (calibration_target ins14 dir7)
    (calibration_target ins15 dir4)
    (calibration_target ins16 dir7)
    (on_board ins1 sat4)
    (on_board ins2 sat9)
    (on_board ins3 sat1)
    (on_board ins4 sat6)
    (on_board ins5 sat8)
    (on_board ins6 sat3)
    (on_board ins7 sat7)
    (on_board ins8 sat2)
    (on_board ins9 sat5)
    (on_board ins10 sat6)
    (on_board ins11 sat7)
    (on_board ins12 sat4)
    (on_board ins13 sat1)
    (on_board ins14 sat2)
    (on_board ins15 sat1)
    (on_board ins16 sat3)
    (supports ins5 mod1)
    (supports ins4 mod1)
    (supports ins6 mod3)
    (supports ins10 mod3)
    (supports ins7 mod1)
    (supports ins14 mod1)
    (supports ins8 mod3)
    (supports ins6 mod2)
    (supports ins2 mod3)
    (supports ins4 mod2)
    (supports ins14 mod2)
    (supports ins11 mod2)
    (supports ins8 mod2)
    (supports ins7 mod3)
    (supports ins2 mod1)
    (supports ins5 mod2)
    (supports ins9 mod3)
    (supports ins2 mod2)
    (supports ins3 mod2)
    (supports ins1 mod2)
    (supports ins13 mod3)
    (supports ins4 mod3)
    (supports ins8 mod1)
    (supports ins15 mod1)
    (supports ins9 mod2)
    (supports ins16 mod3)
    (supports ins10 mod2)
    (supports ins3 mod1)
    (supports ins10 mod1)
    (supports ins3 mod3)
    (supports ins11 mod1)
    (supports ins15 mod3)
    (supports ins14 mod3)
    (supports ins12 mod2))
 (:goal  (and (pointing sat1 dir1)
   (pointing sat2 dir1)
   (pointing sat3 dir1)
   (pointing sat4 dir9)
   (pointing sat5 dir8)
   (pointing sat7 dir5)
   (have_image dir3 mod3)
   (have_image dir3 mod2)
   (have_image dir8 mod2)
   (have_image dir6 mod2)
   (have_image dir8 mod1)
   (have_image dir3 mod1)
   (have_image dir7 mod2)
   (have_image dir5 mod3)
   (have_image dir6 mod3)
   (have_image dir7 mod3)
   (have_image dir9 mod2))))
