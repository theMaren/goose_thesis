;; blocks=127, out_folder=testing_new/medium, instance_id=25, seed=2032

(define (problem blocksworld-25)
 (:domain blocksworld)
 (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26 b27 b28 b29 b30 b31 b32 b33 b34 b35 b36 b37 b38 b39 b40 b41 b42 b43 b44 b45 b46 b47 b48 b49 b50 b51 b52 b53 b54 b55 b56 b57 b58 b59 b60 b61 b62 b63 b64 b65 b66 b67 b68 b69 b70 b71 b72 b73 b74 b75 b76 b77 b78 b79 b80 b81 b82 b83 b84 b85 b86 b87 b88 b89 b90 b91 b92 b93 b94 b95 b96 b97 b98 b99 b100 b101 b102 b103 b104 b105 b106 b107 b108 b109 b110 b111 b112 b113 b114 b115 b116 b117 b118 b119 b120 b121 b122 b123 b124 b125 b126 b127 - object)
 (:init 
    (arm-empty)
    (clear b53)
    (on-table b53)
    (clear b111)
    (on b111 b13)
    (on b13 b3)
    (on b3 b96)
    (on b96 b72)
    (on b72 b52)
    (on b52 b63)
    (on b63 b7)
    (on b7 b57)
    (on b57 b112)
    (on b112 b103)
    (on b103 b24)
    (on b24 b92)
    (on b92 b15)
    (on b15 b28)
    (on b28 b91)
    (on b91 b77)
    (on b77 b40)
    (on b40 b37)
    (on b37 b110)
    (on b110 b45)
    (on b45 b47)
    (on b47 b74)
    (on b74 b42)
    (on b42 b82)
    (on b82 b22)
    (on b22 b127)
    (on b127 b25)
    (on b25 b100)
    (on b100 b64)
    (on b64 b109)
    (on b109 b32)
    (on b32 b88)
    (on b88 b66)
    (on-table b66)
    (clear b80)
    (on b80 b97)
    (on b97 b34)
    (on b34 b68)
    (on b68 b58)
    (on b58 b115)
    (on b115 b29)
    (on b29 b99)
    (on b99 b94)
    (on b94 b61)
    (on b61 b10)
    (on b10 b41)
    (on-table b41)
    (clear b48)
    (on b48 b98)
    (on b98 b116)
    (on b116 b62)
    (on b62 b9)
    (on b9 b38)
    (on b38 b56)
    (on b56 b50)
    (on b50 b14)
    (on b14 b4)
    (on b4 b8)
    (on b8 b49)
    (on b49 b104)
    (on b104 b12)
    (on b12 b2)
    (on b2 b69)
    (on b69 b65)
    (on b65 b71)
    (on b71 b55)
    (on b55 b1)
    (on b1 b33)
    (on b33 b79)
    (on b79 b19)
    (on b19 b67)
    (on-table b67)
    (clear b126)
    (on b126 b17)
    (on b17 b51)
    (on-table b51)
    (clear b26)
    (on-table b26)
    (clear b43)
    (on-table b43)
    (clear b124)
    (on b124 b105)
    (on-table b105)
    (clear b44)
    (on b44 b107)
    (on-table b107)
    (clear b118)
    (on b118 b89)
    (on-table b89)
    (clear b81)
    (on b81 b36)
    (on b36 b70)
    (on b70 b18)
    (on b18 b78)
    (on b78 b11)
    (on b11 b102)
    (on b102 b90)
    (on b90 b108)
    (on b108 b125)
    (on b125 b87)
    (on b87 b35)
    (on b35 b54)
    (on b54 b31)
    (on b31 b39)
    (on b39 b83)
    (on b83 b114)
    (on b114 b20)
    (on b20 b59)
    (on b59 b27)
    (on b27 b93)
    (on b93 b95)
    (on b95 b122)
    (on b122 b6)
    (on b6 b46)
    (on b46 b84)
    (on b84 b113)
    (on b113 b76)
    (on-table b76)
    (clear b73)
    (on b73 b21)
    (on b21 b121)
    (on-table b121)
    (clear b23)
    (on b23 b30)
    (on b30 b120)
    (on b120 b123)
    (on b123 b16)
    (on-table b16)
    (clear b60)
    (on b60 b119)
    (on-table b119)
    (clear b85)
    (on b85 b117)
    (on b117 b101)
    (on b101 b86)
    (on b86 b75)
    (on-table b75)
    (clear b5)
    (on b5 b106)
    (on-table b106))
 (:goal  (and 
    (clear b23)
    (on b23 b101)
    (on-table b101)
    (clear b9)
    (on b9 b35)
    (on b35 b10)
    (on b10 b70)
    (on b70 b57)
    (on b57 b6)
    (on b6 b73)
    (on b73 b112)
    (on b112 b111)
    (on b111 b127)
    (on b127 b78)
    (on b78 b19)
    (on-table b19)
    (clear b88)
    (on-table b88)
    (clear b104)
    (on b104 b48)
    (on b48 b81)
    (on b81 b123)
    (on b123 b37)
    (on-table b37)
    (clear b52)
    (on b52 b49)
    (on b49 b34)
    (on b34 b83)
    (on b83 b63)
    (on b63 b55)
    (on b55 b126)
    (on b126 b74)
    (on-table b74)
    (clear b87)
    (on b87 b120)
    (on b120 b89)
    (on b89 b113)
    (on b113 b56)
    (on b56 b100)
    (on b100 b12)
    (on b12 b33)
    (on-table b33)
    (clear b24)
    (on b24 b2)
    (on b2 b62)
    (on-table b62)
    (clear b13)
    (on b13 b60)
    (on b60 b17)
    (on b17 b117)
    (on b117 b121)
    (on b121 b90)
    (on b90 b22)
    (on b22 b36)
    (on b36 b29)
    (on b29 b53)
    (on b53 b66)
    (on b66 b79)
    (on b79 b114)
    (on b114 b115)
    (on b115 b118)
    (on b118 b124)
    (on b124 b43)
    (on b43 b14)
    (on b14 b32)
    (on b32 b28)
    (on b28 b67)
    (on b67 b58)
    (on b58 b16)
    (on b16 b3)
    (on b3 b5)
    (on b5 b11)
    (on b11 b45)
    (on b45 b91)
    (on b91 b51)
    (on b51 b103)
    (on-table b103)
    (clear b26)
    (on b26 b1)
    (on b1 b96)
    (on b96 b7)
    (on b7 b125)
    (on-table b125)
    (clear b110)
    (on-table b110)
    (clear b84)
    (on-table b84)
    (clear b77)
    (on b77 b65)
    (on b65 b8)
    (on b8 b109)
    (on b109 b4)
    (on b4 b42)
    (on b42 b92)
    (on b92 b76)
    (on b76 b47)
    (on b47 b116)
    (on b116 b108)
    (on b108 b38)
    (on b38 b61)
    (on-table b61)
    (clear b21)
    (on b21 b97)
    (on b97 b64)
    (on b64 b75)
    (on b75 b119)
    (on b119 b20)
    (on b20 b105)
    (on b105 b80)
    (on b80 b102)
    (on b102 b30)
    (on b30 b71)
    (on b71 b41)
    (on b41 b85)
    (on b85 b31)
    (on b31 b25)
    (on b25 b93)
    (on b93 b59)
    (on b59 b94)
    (on b94 b44)
    (on b44 b40)
    (on b40 b69)
    (on b69 b46)
    (on b46 b86)
    (on b86 b68)
    (on b68 b95)
    (on b95 b39)
    (on b39 b50)
    (on b50 b98)
    (on b98 b15)
    (on b15 b82)
    (on-table b82)
    (clear b107)
    (on-table b107)
    (clear b27)
    (on b27 b18)
    (on b18 b99)
    (on b99 b72)
    (on b72 b54)
    (on b54 b106)
    (on b106 b122)
    (on-table b122))))
