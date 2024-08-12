;; blocks=88, out_folder=testing_new/medium, instance_id=15, seed=2022

(define (problem blocksworld-15)
 (:domain blocksworld)
 (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26 b27 b28 b29 b30 b31 b32 b33 b34 b35 b36 b37 b38 b39 b40 b41 b42 b43 b44 b45 b46 b47 b48 b49 b50 b51 b52 b53 b54 b55 b56 b57 b58 b59 b60 b61 b62 b63 b64 b65 b66 b67 b68 b69 b70 b71 b72 b73 b74 b75 b76 b77 b78 b79 b80 b81 b82 b83 b84 b85 b86 b87 b88 - object)
 (:init 
    (arm-empty)
    (clear b86)
    (on b86 b76)
    (on b76 b46)
    (on b46 b39)
    (on b39 b16)
    (on b16 b20)
    (on b20 b62)
    (on b62 b12)
    (on b12 b51)
    (on b51 b22)
    (on b22 b80)
    (on b80 b33)
    (on b33 b30)
    (on b30 b24)
    (on b24 b73)
    (on b73 b26)
    (on b26 b23)
    (on b23 b68)
    (on b68 b55)
    (on b55 b64)
    (on b64 b83)
    (on b83 b13)
    (on b13 b14)
    (on b14 b32)
    (on b32 b47)
    (on b47 b85)
    (on b85 b19)
    (on b19 b60)
    (on b60 b49)
    (on b49 b45)
    (on b45 b10)
    (on b10 b38)
    (on b38 b42)
    (on b42 b77)
    (on b77 b50)
    (on b50 b78)
    (on b78 b66)
    (on b66 b43)
    (on b43 b61)
    (on b61 b17)
    (on-table b17)
    (clear b28)
    (on b28 b21)
    (on b21 b88)
    (on-table b88)
    (clear b35)
    (on b35 b3)
    (on b3 b18)
    (on-table b18)
    (clear b63)
    (on b63 b79)
    (on b79 b25)
    (on-table b25)
    (clear b36)
    (on b36 b7)
    (on b7 b65)
    (on b65 b82)
    (on b82 b58)
    (on b58 b72)
    (on b72 b27)
    (on b27 b31)
    (on b31 b44)
    (on b44 b15)
    (on b15 b4)
    (on b4 b1)
    (on-table b1)
    (clear b11)
    (on b11 b29)
    (on b29 b74)
    (on b74 b54)
    (on b54 b9)
    (on b9 b52)
    (on b52 b87)
    (on b87 b48)
    (on b48 b6)
    (on-table b6)
    (clear b34)
    (on-table b34)
    (clear b5)
    (on-table b5)
    (clear b81)
    (on b81 b71)
    (on b71 b59)
    (on b59 b84)
    (on b84 b56)
    (on b56 b2)
    (on b2 b41)
    (on b41 b53)
    (on b53 b67)
    (on b67 b8)
    (on b8 b75)
    (on b75 b40)
    (on b40 b70)
    (on b70 b57)
    (on b57 b37)
    (on b37 b69)
    (on-table b69))
 (:goal  (and 
    (clear b58)
    (on b58 b49)
    (on b49 b79)
    (on b79 b13)
    (on b13 b86)
    (on b86 b10)
    (on b10 b33)
    (on b33 b5)
    (on b5 b31)
    (on b31 b70)
    (on b70 b11)
    (on-table b11)
    (clear b75)
    (on-table b75)
    (clear b36)
    (on b36 b32)
    (on b32 b60)
    (on b60 b68)
    (on b68 b21)
    (on b21 b45)
    (on b45 b23)
    (on b23 b72)
    (on b72 b41)
    (on b41 b37)
    (on b37 b56)
    (on b56 b3)
    (on b3 b42)
    (on b42 b85)
    (on b85 b35)
    (on-table b35)
    (clear b71)
    (on b71 b88)
    (on b88 b40)
    (on b40 b24)
    (on b24 b74)
    (on b74 b62)
    (on b62 b20)
    (on b20 b38)
    (on b38 b66)
    (on b66 b55)
    (on-table b55)
    (clear b51)
    (on b51 b52)
    (on b52 b81)
    (on b81 b22)
    (on b22 b44)
    (on b44 b57)
    (on b57 b19)
    (on b19 b87)
    (on b87 b9)
    (on b9 b1)
    (on b1 b15)
    (on-table b15)
    (clear b26)
    (on b26 b84)
    (on b84 b64)
    (on b64 b28)
    (on b28 b76)
    (on b76 b17)
    (on b17 b12)
    (on b12 b53)
    (on b53 b65)
    (on b65 b43)
    (on b43 b6)
    (on b6 b59)
    (on b59 b46)
    (on b46 b30)
    (on b30 b47)
    (on b47 b4)
    (on b4 b80)
    (on b80 b50)
    (on-table b50)
    (clear b16)
    (on b16 b69)
    (on b69 b77)
    (on b77 b61)
    (on b61 b25)
    (on b25 b29)
    (on b29 b63)
    (on b63 b27)
    (on b27 b8)
    (on b8 b48)
    (on b48 b2)
    (on b2 b54)
    (on b54 b83)
    (on b83 b14)
    (on-table b14)
    (clear b82)
    (on-table b82)
    (clear b67)
    (on b67 b7)
    (on b7 b78)
    (on b78 b73)
    (on b73 b39)
    (on b39 b18)
    (on b18 b34)
    (on-table b34))))
