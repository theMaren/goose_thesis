;; satellites=10, instruments=20, modes=3, directions=10, out_folder=testing_new/easy, instance_id=30, seed=2037

(define (problem satellite-30)
 (:domain satellite)
 (:objects 
    sat1 sat2 sat3 sat4 sat5 sat6 sat7 sat8 sat9 sat10 - satellite
    ins1 ins2 ins3 ins4 ins5 ins6 ins7 ins8 ins9 ins10 ins11 ins12 ins13 ins14 ins15 ins16 ins17 ins18 ins19 ins20 - instrument
    mod1 mod2 mod3 - mode
    dir1 dir2 dir3 dir4 dir5 dir6 dir7 dir8 dir9 dir10 - direction
    )
 (:init 
    (pointing sat1 dir5)
    (pointing sat2 dir9)
    (pointing sat3 dir1)
    (pointing sat4 dir5)
    (pointing sat5 dir8)
    (pointing sat6 dir3)
    (pointing sat7 dir1)
    (pointing sat8 dir7)
    (pointing sat9 dir2)
    (pointing sat10 dir5)
    (power_avail sat1)
    (power_avail sat2)
    (power_avail sat3)
    (power_avail sat4)
    (power_avail sat5)
    (power_avail sat6)
    (power_avail sat7)
    (power_avail sat8)
    (power_avail sat9)
    (power_avail sat10)
    (calibration_target ins1 dir5)
    (calibration_target ins2 dir7)
    (calibration_target ins3 dir5)
    (calibration_target ins4 dir3)
    (calibration_target ins5 dir5)
    (calibration_target ins6 dir9)
    (calibration_target ins7 dir7)
    (calibration_target ins8 dir8)
    (calibration_target ins9 dir10)
    (calibration_target ins10 dir9)
    (calibration_target ins11 dir2)
    (calibration_target ins12 dir5)
    (calibration_target ins13 dir4)
    (calibration_target ins14 dir8)
    (calibration_target ins15 dir6)
    (calibration_target ins16 dir7)
    (calibration_target ins17 dir7)
    (calibration_target ins18 dir3)
    (calibration_target ins19 dir4)
    (calibration_target ins20 dir9)
    (on_board ins1 sat3)
    (on_board ins2 sat7)
    (on_board ins3 sat1)
    (on_board ins4 sat10)
    (on_board ins5 sat4)
    (on_board ins6 sat5)
    (on_board ins7 sat9)
    (on_board ins8 sat8)
    (on_board ins9 sat6)
    (on_board ins10 sat2)
    (on_board ins11 sat2)
    (on_board ins12 sat6)
    (on_board ins13 sat6)
    (on_board ins14 sat8)
    (on_board ins15 sat10)
    (on_board ins16 sat10)
    (on_board ins17 sat2)
    (on_board ins18 sat9)
    (on_board ins19 sat1)
    (on_board ins20 sat4)
    (supports ins1 mod1)
    (supports ins7 mod2)
    (supports ins9 mod1)
    (supports ins11 mod3)
    (supports ins2 mod3)
    (supports ins6 mod2)
    (supports ins13 mod2)
    (supports ins19 mod3)
    (supports ins14 mod1)
    (supports ins9 mod2)
    (supports ins9 mod3)
    (supports ins8 mod1)
    (supports ins10 mod3)
    (supports ins16 mod3)
    (supports ins19 mod2)
    (supports ins3 mod3)
    (supports ins8 mod3)
    (supports ins18 mod1)
    (supports ins20 mod2)
    (supports ins6 mod1)
    (supports ins14 mod3)
    (supports ins13 mod3)
    (supports ins12 mod2)
    (supports ins6 mod3)
    (supports ins1 mod3)
    (supports ins16 mod2)
    (supports ins18 mod2)
    (supports ins7 mod3)
    (supports ins8 mod2)
    (supports ins5 mod1)
    (supports ins19 mod1)
    (supports ins17 mod3)
    (supports ins17 mod2)
    (supports ins11 mod1)
    (supports ins20 mod3)
    (supports ins15 mod3)
    (supports ins5 mod3)
    (supports ins2 mod1)
    (supports ins17 mod1)
    (supports ins1 mod2)
    (supports ins4 mod1))
 (:goal  (and (pointing sat1 dir1)
   (pointing sat2 dir10)
   (pointing sat4 dir9)
   (pointing sat5 dir7)
   (pointing sat6 dir3)
   (pointing sat7 dir4)
   (pointing sat8 dir1)
   (pointing sat10 dir6)
   (have_image dir8 mod2)
   (have_image dir5 mod1)
   (have_image dir6 mod1)
   (have_image dir4 mod3)
   (have_image dir7 mod1)
   (have_image dir2 mod3)
   (have_image dir10 mod1)
   (have_image dir10 mod2)
   (have_image dir1 mod3)
   (have_image dir6 mod3)
   (have_image dir7 mod2)
   (have_image dir4 mod2)
   (have_image dir3 mod3)
   (have_image dir1 mod1)
   (have_image dir5 mod2)
   (have_image dir4 mod1))))
