(define (problem BW-88-1-6)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26 b27 b28 b29 b30 b31 b32 b33 b34 b35 b36 b37 b38 b39 b40 b41 b42 b43 b44 b45 b46 b47 b48 b49 b50 b51 b52 b53 b54 b55 b56 b57 b58 b59 b60 b61 b62 b63 b64 b65 b66 b67 b68 b69 b70 b71 b72 b73 b74 b75 b76 b77 b78 b79 b80 b81 b82 b83 b84 b85 b86 b87 b88 - block)
    (:init
        (handempty)
        (on b1 b35)
        (on-table b2)
        (on b3 b29)
        (on-table b4)
        (on b5 b23)
        (on b6 b67)
        (on b7 b58)
        (on b8 b39)
        (on b9 b3)
        (on b10 b12)
        (on b11 b49)
        (on b12 b45)
        (on b13 b18)
        (on b14 b28)
        (on b15 b25)
        (on-table b16)
        (on b17 b86)
        (on b18 b11)
        (on b19 b55)
        (on b20 b4)
        (on b21 b50)
        (on b22 b63)
        (on b23 b82)
        (on-table b24)
        (on b25 b87)
        (on b26 b69)
        (on b27 b51)
        (on b28 b7)
        (on b29 b77)
        (on-table b30)
        (on b31 b14)
        (on b32 b62)
        (on-table b33)
        (on b34 b73)
        (on b35 b59)
        (on b36 b34)
        (on b37 b16)
        (on b38 b30)
        (on b39 b41)
        (on b40 b56)
        (on b41 b60)
        (on b42 b52)
        (on b43 b88)
        (on b44 b6)
        (on b45 b40)
        (on b46 b31)
        (on b47 b19)
        (on b48 b37)
        (on b49 b2)
        (on-table b50)
        (on b51 b61)
        (on-table b52)
        (on b53 b84)
        (on b54 b79)
        (on b55 b70)
        (on b56 b36)
        (on b57 b10)
        (on b58 b74)
        (on-table b59)
        (on b60 b26)
        (on b61 b48)
        (on b62 b13)
        (on b63 b15)
        (on b64 b65)
        (on b65 b44)
        (on b66 b78)
        (on b67 b80)
        (on b68 b54)
        (on b69 b32)
        (on b70 b64)
        (on b71 b21)
        (on b72 b71)
        (on b73 b43)
        (on b74 b33)
        (on b75 b8)
        (on b76 b1)
        (on b77 b47)
        (on b78 b24)
        (on b79 b20)
        (on b80 b81)
        (on b81 b38)
        (on b82 b27)
        (on b83 b66)
        (on b84 b85)
        (on b85 b83)
        (on b86 b76)
        (on b87 b42)
        (on b88 b46)
        (clear b5)
        (clear b9)
        (clear b17)
        (clear b22)
        (clear b53)
        (clear b57)
        (clear b68)
        (clear b72)
        (clear b75)
    )
    (:goal
        (and
            (on b1 b59)
            (on b2 b26)
            (on b3 b36)
            (on b4 b11)
            (on b5 b43)
            (on-table b6)
            (on b7 b23)
            (on b8 b35)
            (on b9 b25)
            (on b10 b18)
            (on b11 b52)
            (on b12 b63)
            (on b13 b6)
            (on b14 b32)
            (on-table b15)
            (on b16 b68)
            (on b17 b80)
            (on b18 b54)
            (on b19 b15)
            (on b20 b65)
            (on b21 b53)
            (on b22 b74)
            (on b23 b84)
            (on b24 b10)
            (on b25 b46)
            (on b26 b19)
            (on b27 b38)
            (on b28 b30)
            (on b29 b77)
            (on b30 b31)
            (on b31 b64)
            (on b32 b86)
            (on b33 b56)
            (on b34 b20)
            (on b35 b62)
            (on b36 b72)
            (on b37 b76)
            (on b38 b45)
            (on b39 b14)
            (on b40 b61)
            (on-table b41)
            (on b42 b44)
            (on b43 b66)
            (on b44 b88)
            (on b45 b82)
            (on b46 b48)
            (on-table b47)
            (on b48 b39)
            (on b49 b13)
            (on b50 b3)
            (on b51 b7)
            (on b52 b47)
            (on b53 b37)
            (on b54 b16)
            (on-table b55)
            (on b56 b8)
            (on b57 b69)
            (on b58 b21)
            (on b59 b29)
            (on b60 b27)
            (on b61 b22)
            (on b62 b55)
            (on b63 b1)
            (on-table b64)
            (on b65 b4)
            (on b66 b42)
            (on b67 b2)
            (on b68 b41)
            (on b69 b70)
            (on b70 b85)
            (on b71 b67)
            (on b72 b83)
            (on b73 b33)
            (on b74 b75)
            (on b75 b81)
            (on b76 b87)
            (on b77 b79)
            (on b78 b58)
            (on b79 b9)
            (on b80 b49)
            (on b81 b51)
            (on b82 b28)
            (on b83 b12)
            (on-table b84)
            (on b85 b78)
            (on b86 b71)
            (on b87 b17)
            (on b88 b50)
        )
    )
)