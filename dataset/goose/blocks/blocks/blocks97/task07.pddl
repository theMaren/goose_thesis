(define (problem BW-97-1-7)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26 b27 b28 b29 b30 b31 b32 b33 b34 b35 b36 b37 b38 b39 b40 b41 b42 b43 b44 b45 b46 b47 b48 b49 b50 b51 b52 b53 b54 b55 b56 b57 b58 b59 b60 b61 b62 b63 b64 b65 b66 b67 b68 b69 b70 b71 b72 b73 b74 b75 b76 b77 b78 b79 b80 b81 b82 b83 b84 b85 b86 b87 b88 b89 b90 b91 b92 b93 b94 b95 b96 b97 - block)
    (:init
        (handempty)
        (on b1 b73)
        (on b2 b51)
        (on b3 b52)
        (on b4 b49)
        (on b5 b3)
        (on b6 b17)
        (on-table b7)
        (on b8 b38)
        (on b9 b15)
        (on b10 b1)
        (on b11 b44)
        (on b12 b20)
        (on b13 b31)
        (on b14 b48)
        (on b15 b84)
        (on b16 b68)
        (on b17 b55)
        (on b18 b76)
        (on-table b19)
        (on b20 b91)
        (on b21 b71)
        (on b22 b18)
        (on b23 b25)
        (on b24 b77)
        (on b25 b75)
        (on b26 b88)
        (on b27 b87)
        (on-table b28)
        (on b29 b19)
        (on b30 b2)
        (on-table b31)
        (on b32 b11)
        (on b33 b82)
        (on b34 b61)
        (on b35 b29)
        (on b36 b21)
        (on b37 b43)
        (on b38 b39)
        (on b39 b28)
        (on b40 b64)
        (on b41 b86)
        (on b42 b12)
        (on b43 b65)
        (on b44 b23)
        (on b45 b26)
        (on b46 b7)
        (on-table b47)
        (on b48 b90)
        (on b49 b9)
        (on b50 b45)
        (on b51 b57)
        (on b52 b35)
        (on-table b53)
        (on b54 b74)
        (on b55 b8)
        (on b56 b36)
        (on b57 b62)
        (on-table b58)
        (on b59 b85)
        (on b60 b67)
        (on b61 b14)
        (on b62 b32)
        (on b63 b30)
        (on b64 b50)
        (on b65 b27)
        (on b66 b54)
        (on b67 b42)
        (on b68 b60)
        (on b69 b78)
        (on b70 b33)
        (on b71 b16)
        (on b72 b46)
        (on b73 b69)
        (on b74 b41)
        (on b75 b22)
        (on b76 b70)
        (on b77 b13)
        (on b78 b89)
        (on b79 b53)
        (on b80 b58)
        (on b81 b83)
        (on b82 b5)
        (on b83 b56)
        (on b84 b72)
        (on b85 b93)
        (on-table b86)
        (on b87 b94)
        (on b88 b4)
        (on b89 b37)
        (on b90 b66)
        (on b91 b63)
        (on b92 b97)
        (on b93 b92)
        (on b94 b96)
        (on b95 b81)
        (on b96 b34)
        (on b97 b10)
        (clear b6)
        (clear b24)
        (clear b40)
        (clear b47)
        (clear b59)
        (clear b79)
        (clear b80)
        (clear b95)
    )
    (:goal
        (and
            (on b1 b86)
            (on b2 b10)
            (on b3 b64)
            (on b4 b96)
            (on-table b5)
            (on b6 b75)
            (on b7 b97)
            (on b8 b69)
            (on b9 b72)
            (on b10 b49)
            (on b11 b60)
            (on b12 b47)
            (on b13 b40)
            (on b14 b16)
            (on-table b15)
            (on b16 b23)
            (on b17 b88)
            (on b18 b85)
            (on b19 b53)
            (on b20 b55)
            (on b21 b95)
            (on b22 b17)
            (on-table b23)
            (on b24 b3)
            (on b25 b26)
            (on b26 b71)
            (on b27 b80)
            (on b28 b74)
            (on b29 b92)
            (on-table b30)
            (on b31 b79)
            (on b32 b25)
            (on b33 b12)
            (on b34 b4)
            (on b35 b22)
            (on b36 b87)
            (on b37 b58)
            (on b38 b21)
            (on b39 b36)
            (on b40 b73)
            (on b41 b81)
            (on b42 b20)
            (on-table b43)
            (on b44 b29)
            (on b45 b48)
            (on-table b46)
            (on-table b47)
            (on b48 b2)
            (on b49 b9)
            (on b50 b54)
            (on b51 b18)
            (on b52 b19)
            (on b53 b83)
            (on b54 b34)
            (on b55 b51)
            (on b56 b77)
            (on b57 b7)
            (on b58 b31)
            (on b59 b50)
            (on b60 b14)
            (on b61 b66)
            (on b62 b32)
            (on b63 b42)
            (on b64 b91)
            (on-table b65)
            (on b66 b27)
            (on b67 b11)
            (on b68 b24)
            (on b69 b59)
            (on b70 b15)
            (on b71 b41)
            (on b72 b76)
            (on b73 b46)
            (on b74 b30)
            (on b75 b89)
            (on b76 b8)
            (on b77 b52)
            (on b78 b43)
            (on b79 b44)
            (on b80 b63)
            (on b81 b94)
            (on b82 b39)
            (on b83 b90)
            (on-table b84)
            (on b85 b68)
            (on b86 b13)
            (on b87 b1)
            (on b88 b62)
            (on b89 b84)
            (on b90 b70)
            (on b91 b5)
            (on-table b92)
            (on b93 b38)
            (on b94 b93)
            (on b95 b67)
            (on b96 b6)
            (on b97 b78)
        )
    )
)