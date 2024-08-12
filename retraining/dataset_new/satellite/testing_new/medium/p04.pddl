;; satellites=17, instruments=21, modes=3, directions=16, out_folder=testing_new/medium, instance_id=4, seed=2011

(define (problem satellite-04)
 (:domain satellite)
 (:objects 
    sat1 sat2 sat3 sat4 sat5 sat6 sat7 sat8 sat9 sat10 sat11 sat12 sat13 sat14 sat15 sat16 sat17 - satellite
    ins1 ins2 ins3 ins4 ins5 ins6 ins7 ins8 ins9 ins10 ins11 ins12 ins13 ins14 ins15 ins16 ins17 ins18 ins19 ins20 ins21 - instrument
    mod1 mod2 mod3 - mode
    dir1 dir2 dir3 dir4 dir5 dir6 dir7 dir8 dir9 dir10 dir11 dir12 dir13 dir14 dir15 dir16 - direction
    )
 (:init 
    (pointing sat1 dir8)
    (pointing sat2 dir10)
    (pointing sat3 dir6)
    (pointing sat4 dir10)
    (pointing sat5 dir2)
    (pointing sat6 dir12)
    (pointing sat7 dir1)
    (pointing sat8 dir10)
    (pointing sat9 dir7)
    (pointing sat10 dir8)
    (pointing sat11 dir2)
    (pointing sat12 dir2)
    (pointing sat13 dir6)
    (pointing sat14 dir11)
    (pointing sat15 dir13)
    (pointing sat16 dir15)
    (pointing sat17 dir1)
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
    (power_avail sat11)
    (power_avail sat12)
    (power_avail sat13)
    (power_avail sat14)
    (power_avail sat15)
    (power_avail sat16)
    (power_avail sat17)
    (calibration_target ins1 dir4)
    (calibration_target ins2 dir10)
    (calibration_target ins3 dir13)
    (calibration_target ins4 dir6)
    (calibration_target ins5 dir13)
    (calibration_target ins6 dir10)
    (calibration_target ins7 dir8)
    (calibration_target ins8 dir8)
    (calibration_target ins9 dir2)
    (calibration_target ins10 dir1)
    (calibration_target ins11 dir6)
    (calibration_target ins12 dir4)
    (calibration_target ins13 dir15)
    (calibration_target ins14 dir15)
    (calibration_target ins15 dir13)
    (calibration_target ins16 dir16)
    (calibration_target ins17 dir13)
    (calibration_target ins18 dir8)
    (calibration_target ins19 dir1)
    (calibration_target ins20 dir16)
    (calibration_target ins21 dir6)
    (on_board ins1 sat8)
    (on_board ins2 sat10)
    (on_board ins3 sat2)
    (on_board ins4 sat3)
    (on_board ins5 sat5)
    (on_board ins6 sat15)
    (on_board ins7 sat14)
    (on_board ins8 sat11)
    (on_board ins9 sat9)
    (on_board ins10 sat6)
    (on_board ins11 sat4)
    (on_board ins12 sat17)
    (on_board ins13 sat1)
    (on_board ins14 sat13)
    (on_board ins15 sat12)
    (on_board ins16 sat16)
    (on_board ins17 sat7)
    (on_board ins18 sat9)
    (on_board ins19 sat2)
    (on_board ins20 sat17)
    (on_board ins21 sat7)
    (supports ins13 mod2)
    (supports ins15 mod1)
    (supports ins21 mod1)
    (supports ins13 mod3)
    (supports ins3 mod1)
    (supports ins5 mod3)
    (supports ins19 mod3)
    (supports ins1 mod3)
    (supports ins19 mod2)
    (supports ins4 mod3)
    (supports ins11 mod3)
    (supports ins2 mod1)
    (supports ins9 mod2)
    (supports ins4 mod2)
    (supports ins6 mod3)
    (supports ins3 mod2)
    (supports ins16 mod2)
    (supports ins8 mod3)
    (supports ins20 mod3)
    (supports ins12 mod1)
    (supports ins1 mod2)
    (supports ins12 mod3)
    (supports ins7 mod1)
    (supports ins17 mod2)
    (supports ins7 mod3)
    (supports ins20 mod2)
    (supports ins2 mod2)
    (supports ins21 mod3)
    (supports ins16 mod3)
    (supports ins5 mod2)
    (supports ins10 mod2)
    (supports ins15 mod3)
    (supports ins17 mod1)
    (supports ins3 mod3)
    (supports ins6 mod1)
    (supports ins5 mod1)
    (supports ins6 mod2)
    (supports ins21 mod2)
    (supports ins18 mod1)
    (supports ins10 mod3)
    (supports ins1 mod1)
    (supports ins8 mod1)
    (supports ins4 mod1)
    (supports ins11 mod2)
    (supports ins10 mod1)
    (supports ins19 mod1)
    (supports ins11 mod1)
    (supports ins15 mod2)
    (supports ins14 mod1))
 (:goal  (and (pointing sat1 dir3)
   (pointing sat2 dir1)
   (pointing sat4 dir16)
   (pointing sat8 dir8)
   (pointing sat10 dir4)
   (pointing sat11 dir12)
   (pointing sat13 dir10)
   (pointing sat15 dir16)
   (have_image dir11 mod2)
   (have_image dir15 mod1)
   (have_image dir16 mod1)
   (have_image dir14 mod2)
   (have_image dir5 mod3)
   (have_image dir5 mod1)
   (have_image dir4 mod3)
   (have_image dir7 mod3)
   (have_image dir15 mod3)
   (have_image dir8 mod1)
   (have_image dir1 mod3)
   (have_image dir2 mod3)
   (have_image dir6 mod1)
   (have_image dir15 mod2)
   (have_image dir13 mod1)
   (have_image dir8 mod3)
   (have_image dir10 mod2)
   (have_image dir13 mod2)
   (have_image dir4 mod1)
   (have_image dir16 mod2)
   (have_image dir11 mod3)
   (have_image dir11 mod1)
   (have_image dir9 mod2)
   (have_image dir10 mod3)
   (have_image dir5 mod2)
   (have_image dir3 mod2)
   (have_image dir14 mod1)
   (have_image dir1 mod2)
   (have_image dir2 mod1)
   (have_image dir1 mod1)
   (have_image dir7 mod2)
   (have_image dir6 mod2))))
