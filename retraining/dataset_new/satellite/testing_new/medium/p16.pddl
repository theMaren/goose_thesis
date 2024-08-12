;; satellites=28, instruments=48, modes=4, directions=23, out_folder=testing_new/medium, instance_id=16, seed=2023

(define (problem satellite-16)
 (:domain satellite)
 (:objects 
    sat1 sat2 sat3 sat4 sat5 sat6 sat7 sat8 sat9 sat10 sat11 sat12 sat13 sat14 sat15 sat16 sat17 sat18 sat19 sat20 sat21 sat22 sat23 sat24 sat25 sat26 sat27 sat28 - satellite
    ins1 ins2 ins3 ins4 ins5 ins6 ins7 ins8 ins9 ins10 ins11 ins12 ins13 ins14 ins15 ins16 ins17 ins18 ins19 ins20 ins21 ins22 ins23 ins24 ins25 ins26 ins27 ins28 ins29 ins30 ins31 ins32 ins33 ins34 ins35 ins36 ins37 ins38 ins39 ins40 ins41 ins42 ins43 ins44 ins45 ins46 ins47 ins48 - instrument
    mod1 mod2 mod3 mod4 - mode
    dir1 dir2 dir3 dir4 dir5 dir6 dir7 dir8 dir9 dir10 dir11 dir12 dir13 dir14 dir15 dir16 dir17 dir18 dir19 dir20 dir21 dir22 dir23 - direction
    )
 (:init 
    (pointing sat1 dir13)
    (pointing sat2 dir23)
    (pointing sat3 dir15)
    (pointing sat4 dir13)
    (pointing sat5 dir11)
    (pointing sat6 dir20)
    (pointing sat7 dir19)
    (pointing sat8 dir11)
    (pointing sat9 dir21)
    (pointing sat10 dir4)
    (pointing sat11 dir4)
    (pointing sat12 dir21)
    (pointing sat13 dir10)
    (pointing sat14 dir23)
    (pointing sat15 dir18)
    (pointing sat16 dir23)
    (pointing sat17 dir17)
    (pointing sat18 dir10)
    (pointing sat19 dir6)
    (pointing sat20 dir8)
    (pointing sat21 dir6)
    (pointing sat22 dir21)
    (pointing sat23 dir1)
    (pointing sat24 dir7)
    (pointing sat25 dir12)
    (pointing sat26 dir3)
    (pointing sat27 dir20)
    (pointing sat28 dir5)
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
    (power_avail sat18)
    (power_avail sat19)
    (power_avail sat20)
    (power_avail sat21)
    (power_avail sat22)
    (power_avail sat23)
    (power_avail sat24)
    (power_avail sat25)
    (power_avail sat26)
    (power_avail sat27)
    (power_avail sat28)
    (calibration_target ins1 dir23)
    (calibration_target ins2 dir2)
    (calibration_target ins3 dir7)
    (calibration_target ins4 dir5)
    (calibration_target ins5 dir16)
    (calibration_target ins6 dir15)
    (calibration_target ins7 dir18)
    (calibration_target ins8 dir11)
    (calibration_target ins9 dir17)
    (calibration_target ins10 dir3)
    (calibration_target ins11 dir3)
    (calibration_target ins12 dir18)
    (calibration_target ins13 dir15)
    (calibration_target ins14 dir16)
    (calibration_target ins15 dir1)
    (calibration_target ins16 dir4)
    (calibration_target ins17 dir19)
    (calibration_target ins18 dir8)
    (calibration_target ins19 dir20)
    (calibration_target ins20 dir21)
    (calibration_target ins21 dir4)
    (calibration_target ins22 dir19)
    (calibration_target ins23 dir3)
    (calibration_target ins24 dir3)
    (calibration_target ins25 dir5)
    (calibration_target ins26 dir13)
    (calibration_target ins27 dir16)
    (calibration_target ins28 dir21)
    (calibration_target ins29 dir7)
    (calibration_target ins30 dir20)
    (calibration_target ins31 dir21)
    (calibration_target ins32 dir6)
    (calibration_target ins33 dir18)
    (calibration_target ins34 dir7)
    (calibration_target ins35 dir10)
    (calibration_target ins36 dir9)
    (calibration_target ins37 dir21)
    (calibration_target ins38 dir23)
    (calibration_target ins39 dir4)
    (calibration_target ins40 dir19)
    (calibration_target ins41 dir2)
    (calibration_target ins42 dir12)
    (calibration_target ins43 dir16)
    (calibration_target ins44 dir5)
    (calibration_target ins45 dir10)
    (calibration_target ins46 dir3)
    (calibration_target ins47 dir17)
    (calibration_target ins48 dir13)
    (on_board ins1 sat4)
    (on_board ins2 sat14)
    (on_board ins3 sat13)
    (on_board ins4 sat18)
    (on_board ins5 sat17)
    (on_board ins6 sat23)
    (on_board ins7 sat7)
    (on_board ins8 sat28)
    (on_board ins9 sat16)
    (on_board ins10 sat1)
    (on_board ins11 sat2)
    (on_board ins12 sat24)
    (on_board ins13 sat20)
    (on_board ins14 sat21)
    (on_board ins15 sat25)
    (on_board ins16 sat27)
    (on_board ins17 sat8)
    (on_board ins18 sat12)
    (on_board ins19 sat3)
    (on_board ins20 sat15)
    (on_board ins21 sat6)
    (on_board ins22 sat5)
    (on_board ins23 sat10)
    (on_board ins24 sat26)
    (on_board ins25 sat9)
    (on_board ins26 sat11)
    (on_board ins27 sat22)
    (on_board ins28 sat19)
    (on_board ins29 sat6)
    (on_board ins30 sat21)
    (on_board ins31 sat26)
    (on_board ins32 sat9)
    (on_board ins33 sat14)
    (on_board ins34 sat17)
    (on_board ins35 sat23)
    (on_board ins36 sat4)
    (on_board ins37 sat14)
    (on_board ins38 sat25)
    (on_board ins39 sat24)
    (on_board ins40 sat16)
    (on_board ins41 sat23)
    (on_board ins42 sat14)
    (on_board ins43 sat3)
    (on_board ins44 sat26)
    (on_board ins45 sat24)
    (on_board ins46 sat21)
    (on_board ins47 sat28)
    (on_board ins48 sat28)
    (supports ins24 mod4)
    (supports ins40 mod2)
    (supports ins30 mod1)
    (supports ins29 mod1)
    (supports ins30 mod3)
    (supports ins14 mod1)
    (supports ins21 mod3)
    (supports ins1 mod3)
    (supports ins12 mod1)
    (supports ins21 mod4)
    (supports ins14 mod4)
    (supports ins33 mod1)
    (supports ins2 mod3)
    (supports ins35 mod4)
    (supports ins40 mod3)
    (supports ins16 mod3)
    (supports ins46 mod2)
    (supports ins1 mod2)
    (supports ins3 mod4)
    (supports ins6 mod4)
    (supports ins31 mod2)
    (supports ins32 mod4)
    (supports ins35 mod3)
    (supports ins47 mod3)
    (supports ins25 mod1)
    (supports ins41 mod2)
    (supports ins44 mod4)
    (supports ins47 mod2)
    (supports ins32 mod3)
    (supports ins20 mod2)
    (supports ins48 mod2)
    (supports ins40 mod4)
    (supports ins11 mod4)
    (supports ins4 mod3)
    (supports ins27 mod3)
    (supports ins10 mod3)
    (supports ins5 mod3)
    (supports ins27 mod4)
    (supports ins1 mod1)
    (supports ins30 mod2)
    (supports ins29 mod3)
    (supports ins45 mod1)
    (supports ins17 mod3)
    (supports ins23 mod3)
    (supports ins31 mod1)
    (supports ins35 mod1)
    (supports ins17 mod2)
    (supports ins46 mod4)
    (supports ins5 mod1)
    (supports ins8 mod1)
    (supports ins23 mod4)
    (supports ins42 mod4)
    (supports ins9 mod1)
    (supports ins12 mod3)
    (supports ins39 mod4)
    (supports ins36 mod3)
    (supports ins11 mod3)
    (supports ins44 mod1)
    (supports ins47 mod1)
    (supports ins34 mod4)
    (supports ins10 mod1)
    (supports ins37 mod4)
    (supports ins3 mod2)
    (supports ins9 mod2)
    (supports ins10 mod2)
    (supports ins43 mod4)
    (supports ins19 mod3)
    (supports ins29 mod2)
    (supports ins22 mod3)
    (supports ins21 mod2)
    (supports ins42 mod2)
    (supports ins40 mod1)
    (supports ins17 mod4)
    (supports ins26 mod4)
    (supports ins46 mod1)
    (supports ins42 mod3)
    (supports ins26 mod2)
    (supports ins1 mod4)
    (supports ins28 mod1)
    (supports ins33 mod3)
    (supports ins33 mod2)
    (supports ins7 mod4)
    (supports ins8 mod3)
    (supports ins35 mod2)
    (supports ins20 mod3)
    (supports ins34 mod2)
    (supports ins44 mod3)
    (supports ins28 mod3)
    (supports ins13 mod1)
    (supports ins41 mod4)
    (supports ins2 mod1)
    (supports ins4 mod4)
    (supports ins48 mod3)
    (supports ins7 mod1)
    (supports ins20 mod4)
    (supports ins36 mod4)
    (supports ins25 mod3)
    (supports ins12 mod4)
    (supports ins22 mod4)
    (supports ins24 mod2)
    (supports ins31 mod4)
    (supports ins11 mod2)
    (supports ins37 mod3)
    (supports ins47 mod4)
    (supports ins43 mod2)
    (supports ins30 mod4)
    (supports ins26 mod1)
    (supports ins27 mod2)
    (supports ins29 mod4)
    (supports ins45 mod3)
    (supports ins37 mod2)
    (supports ins39 mod3)
    (supports ins6 mod2)
    (supports ins13 mod3)
    (supports ins18 mod2)
    (supports ins2 mod2)
    (supports ins7 mod3)
    (supports ins36 mod1)
    (supports ins4 mod2)
    (supports ins16 mod2)
    (supports ins44 mod2)
    (supports ins9 mod3)
    (supports ins27 mod1)
    (supports ins36 mod2)
    (supports ins14 mod3)
    (supports ins16 mod1)
    (supports ins28 mod4)
    (supports ins48 mod4)
    (supports ins34 mod3)
    (supports ins41 mod1)
    (supports ins15 mod3)
    (supports ins20 mod1)
    (supports ins46 mod3)
    (supports ins15 mod2)
    (supports ins9 mod4)
    (supports ins15 mod4)
    (supports ins6 mod1)
    (supports ins6 mod3)
    (supports ins11 mod1)
    (supports ins10 mod4)
    (supports ins19 mod2)
    (supports ins24 mod1)
    (supports ins42 mod1)
    (supports ins34 mod1)
    (supports ins15 mod1)
    (supports ins41 mod3)
    (supports ins19 mod1)
    (supports ins18 mod4)
    (supports ins45 mod2)
    (supports ins33 mod4)
    (supports ins38 mod2))
 (:goal  (and (pointing sat1 dir3)
   (pointing sat3 dir2)
   (pointing sat4 dir22)
   (pointing sat5 dir9)
   (pointing sat11 dir23)
   (pointing sat15 dir18)
   (pointing sat16 dir7)
   (pointing sat22 dir23)
   (pointing sat24 dir17)
   (pointing sat27 dir3)
   (pointing sat28 dir3)
   (have_image dir6 mod2)
   (have_image dir16 mod2)
   (have_image dir12 mod2)
   (have_image dir21 mod4)
   (have_image dir23 mod3))))
