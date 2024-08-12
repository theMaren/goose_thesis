;; blocks=96, out_folder=testing_new/medium, instance_id=17, seed=2024

(define (problem blocksworld-17)
 (:domain blocksworld)
 (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26 b27 b28 b29 b30 b31 b32 b33 b34 b35 b36 b37 b38 b39 b40 b41 b42 b43 b44 b45 b46 b47 b48 b49 b50 b51 b52 b53 b54 b55 b56 b57 b58 b59 b60 b61 b62 b63 b64 b65 b66 b67 b68 b69 b70 b71 b72 b73 b74 b75 b76 b77 b78 b79 b80 b81 b82 b83 b84 b85 b86 b87 b88 b89 b90 b91 b92 b93 b94 b95 b96 - object)
 (:init 
    (arm-empty)
    (clear b38)
    (on b38 b18)
    (on b18 b36)
    (on b36 b56)
    (on b56 b29)
    (on b29 b76)
    (on b76 b35)
    (on b35 b1)
    (on b1 b44)
    (on b44 b52)
    (on b52 b42)
    (on b42 b72)
    (on b72 b93)
    (on b93 b4)
    (on b4 b41)
    (on b41 b95)
    (on b95 b6)
    (on b6 b89)
    (on b89 b58)
    (on b58 b11)
    (on b11 b91)
    (on b91 b66)
    (on b66 b96)
    (on b96 b87)
    (on b87 b92)
    (on b92 b20)
    (on b20 b81)
    (on b81 b31)
    (on b31 b7)
    (on b7 b5)
    (on-table b5)
    (clear b12)
    (on b12 b30)
    (on b30 b85)
    (on b85 b77)
    (on b77 b57)
    (on b57 b86)
    (on b86 b63)
    (on-table b63)
    (clear b65)
    (on b65 b50)
    (on b50 b33)
    (on b33 b17)
    (on b17 b2)
    (on b2 b84)
    (on b84 b3)
    (on b3 b62)
    (on b62 b15)
    (on b15 b55)
    (on b55 b14)
    (on b14 b37)
    (on b37 b71)
    (on b71 b74)
    (on b74 b78)
    (on b78 b59)
    (on b59 b13)
    (on b13 b51)
    (on b51 b23)
    (on b23 b22)
    (on-table b22)
    (clear b25)
    (on b25 b21)
    (on b21 b49)
    (on b49 b9)
    (on b9 b48)
    (on b48 b47)
    (on b47 b16)
    (on-table b16)
    (clear b73)
    (on b73 b83)
    (on b83 b45)
    (on-table b45)
    (clear b8)
    (on b8 b90)
    (on-table b90)
    (clear b80)
    (on b80 b88)
    (on b88 b19)
    (on b19 b60)
    (on b60 b27)
    (on b27 b10)
    (on b10 b67)
    (on b67 b43)
    (on b43 b70)
    (on b70 b40)
    (on b40 b28)
    (on b28 b79)
    (on b79 b68)
    (on b68 b54)
    (on b54 b46)
    (on b46 b64)
    (on b64 b82)
    (on b82 b32)
    (on b32 b69)
    (on b69 b34)
    (on b34 b53)
    (on b53 b26)
    (on b26 b39)
    (on b39 b75)
    (on b75 b94)
    (on b94 b24)
    (on b24 b61)
    (on-table b61))
 (:goal  (and 
    (clear b34)
    (on b34 b42)
    (on b42 b10)
    (on b10 b47)
    (on b47 b58)
    (on b58 b13)
    (on b13 b53)
    (on b53 b91)
    (on b91 b48)
    (on b48 b25)
    (on b25 b27)
    (on b27 b81)
    (on b81 b36)
    (on b36 b51)
    (on b51 b93)
    (on b93 b50)
    (on b50 b6)
    (on b6 b31)
    (on b31 b96)
    (on b96 b86)
    (on b86 b94)
    (on-table b94)
    (clear b72)
    (on b72 b67)
    (on b67 b38)
    (on b38 b26)
    (on b26 b68)
    (on b68 b39)
    (on b39 b84)
    (on b84 b19)
    (on b19 b37)
    (on b37 b35)
    (on b35 b49)
    (on b49 b45)
    (on b45 b15)
    (on b15 b79)
    (on b79 b62)
    (on b62 b75)
    (on b75 b4)
    (on b4 b77)
    (on b77 b70)
    (on b70 b63)
    (on b63 b41)
    (on b41 b52)
    (on b52 b55)
    (on b55 b95)
    (on b95 b8)
    (on b8 b3)
    (on-table b3)
    (clear b61)
    (on b61 b69)
    (on b69 b40)
    (on b40 b57)
    (on b57 b83)
    (on b83 b30)
    (on b30 b85)
    (on b85 b24)
    (on b24 b11)
    (on b11 b32)
    (on b32 b17)
    (on b17 b20)
    (on b20 b22)
    (on-table b22)
    (clear b73)
    (on b73 b74)
    (on b74 b12)
    (on b12 b18)
    (on b18 b9)
    (on b9 b76)
    (on b76 b14)
    (on b14 b89)
    (on b89 b21)
    (on b21 b33)
    (on b33 b82)
    (on b82 b44)
    (on b44 b66)
    (on b66 b92)
    (on b92 b90)
    (on b90 b56)
    (on b56 b7)
    (on b7 b80)
    (on b80 b64)
    (on b64 b71)
    (on b71 b60)
    (on b60 b78)
    (on b78 b28)
    (on b28 b5)
    (on b5 b54)
    (on b54 b65)
    (on b65 b29)
    (on b29 b1)
    (on b1 b87)
    (on b87 b43)
    (on b43 b16)
    (on b16 b2)
    (on b2 b88)
    (on b88 b59)
    (on-table b59)
    (clear b46)
    (on b46 b23)
    (on-table b23))))
