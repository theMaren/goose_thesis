
;In guarantee_solution():
;Planning path: 0
;In plan_path():
;In put_box: 
;length of path: 33
;Planning path: 1
;In plan_path():
;In put_box: 
;length of path: 32
;In set_wall_mask(): 
;In clear_box_neighborhood(): 
;In enforce_direction_constraints(): 
;In setup_walls(): 
;Grid: 

; 8 8 8 8 8 8 8 8 8 8 8 
; 8 8 8 8 8 8 8 8 8 8 8 
; 8 8 8 8 8 8 8 8 8 8 8 
; 0 8 8 8 8 8 8 8 8 8 8 
; 8 3 8 3 8 8 8 8 2 8 8 
; 8 8 3 3 8 8 8 8 8 8 8 
; 8 8 8 8 8 8 8 3 8 8 8 
; 8 8 8 8 2 8 8 8 8 8 8 
; 8 8 8 8 8 8 8 8 8 8 8 
; 8 8 8 8 8 8 8 8 8 3 1 
; 8 8 8 8 8 8 8 8 8 3 1 

;Wall mask: 

; 0 1 1 1 1 0 0 1 1 0 0 
; 0 0 0 0 0 0 0 0 0 0 0 
; 0 1 1 0 0 0 0 0 0 0 0 
; 0 0 0 0 0 1 0 1 0 0 0 
; 0 1 1 1 0 1 0 0 0 0 0 
; 0 1 1 1 0 1 0 1 0 0 0 
; 0 1 1 1 0 1 0 1 0 0 0 
; 0 1 1 0 0 0 0 1 0 0 0 
; 0 1 1 0 0 0 0 1 0 0 0 
; 0 1 1 0 0 0 0 0 0 0 0 
; 0 1 1 1 1 0 0 0 0 0 0 




(define (problem typed-sokoban-grid11-boxes2-walls4)
(:domain typed-sokoban)
(:objects 
        up down left right - DIR
        box0 box1 - BOX
        f0-0f f0-1f f0-2f f0-3f f0-4f f0-5f f0-6f f0-7f f0-8f f0-9f f0-10f 
        f1-0f f1-1f f1-2f f1-3f f1-4f f1-5f f1-6f f1-7f f1-8f f1-9f f1-10f 
        f2-0f f2-1f f2-2f f2-3f f2-4f f2-5f f2-6f f2-7f f2-8f f2-9f f2-10f 
        f3-0f f3-1f f3-2f f3-3f f3-4f f3-5f f3-6f f3-7f f3-8f f3-9f f3-10f 
        f4-0f f4-1f f4-2f f4-3f f4-4f f4-5f f4-6f f4-7f f4-8f f4-9f f4-10f 
        f5-0f f5-1f f5-2f f5-3f f5-4f f5-5f f5-6f f5-7f f5-8f f5-9f f5-10f 
        f6-0f f6-1f f6-2f f6-3f f6-4f f6-5f f6-6f f6-7f f6-8f f6-9f f6-10f 
        f7-0f f7-1f f7-2f f7-3f f7-4f f7-5f f7-6f f7-7f f7-8f f7-9f f7-10f 
        f8-0f f8-1f f8-2f f8-3f f8-4f f8-5f f8-6f f8-7f f8-8f f8-9f f8-10f 
        f9-0f f9-1f f9-2f f9-3f f9-4f f9-5f f9-6f f9-7f f9-8f f9-9f f9-10f 
        f10-0f f10-1f f10-2f f10-3f f10-4f f10-5f f10-6f f10-7f f10-8f f10-9f f10-10f  - LOC
)
(:init
(adjacent f0-0f f0-1f right)
(adjacent f0-0f f1-0f down)
(adjacent f0-1f f0-0f left)
(adjacent f0-1f f0-2f right)
(adjacent f0-1f f1-1f down)
(adjacent f0-2f f0-1f left)
(adjacent f0-2f f0-3f right)
(adjacent f0-2f f1-2f down)
(adjacent f0-3f f0-2f left)
(adjacent f0-3f f0-4f right)
(adjacent f0-3f f1-3f down)
(adjacent f0-4f f0-3f left)
(adjacent f0-4f f0-5f right)
(adjacent f0-4f f1-4f down)
(adjacent f0-5f f0-4f left)
(adjacent f0-5f f0-6f right)
(adjacent f0-5f f1-5f down)
(adjacent f0-6f f0-5f left)
(adjacent f0-6f f0-7f right)
(adjacent f0-6f f1-6f down)
(adjacent f0-7f f0-6f left)
(adjacent f0-7f f0-8f right)
(adjacent f0-7f f1-7f down)
(adjacent f0-8f f0-7f left)
(adjacent f0-8f f0-9f right)
(adjacent f0-8f f1-8f down)
(adjacent f0-9f f0-8f left)
(adjacent f0-9f f0-10f right)
(adjacent f0-9f f1-9f down)
(adjacent f0-10f f0-9f left)
(adjacent f0-10f f1-10f down)
(adjacent f1-0f f1-1f right)
(adjacent f1-0f f0-0f up)
(adjacent f1-0f f2-0f down)
(adjacent f1-1f f1-0f left)
(adjacent f1-1f f1-2f right)
(adjacent f1-1f f0-1f up)
(adjacent f1-1f f2-1f down)
(adjacent f1-2f f1-1f left)
(adjacent f1-2f f1-3f right)
(adjacent f1-2f f0-2f up)
(adjacent f1-2f f2-2f down)
(adjacent f1-3f f1-2f left)
(adjacent f1-3f f1-4f right)
(adjacent f1-3f f0-3f up)
(adjacent f1-3f f2-3f down)
(adjacent f1-4f f1-3f left)
(adjacent f1-4f f1-5f right)
(adjacent f1-4f f0-4f up)
(adjacent f1-4f f2-4f down)
(adjacent f1-5f f1-4f left)
(adjacent f1-5f f1-6f right)
(adjacent f1-5f f0-5f up)
(adjacent f1-5f f2-5f down)
(adjacent f1-6f f1-5f left)
(adjacent f1-6f f1-7f right)
(adjacent f1-6f f0-6f up)
(adjacent f1-6f f2-6f down)
(adjacent f1-7f f1-6f left)
(adjacent f1-7f f1-8f right)
(adjacent f1-7f f0-7f up)
(adjacent f1-7f f2-7f down)
(adjacent f1-8f f1-7f left)
(adjacent f1-8f f1-9f right)
(adjacent f1-8f f0-8f up)
(adjacent f1-8f f2-8f down)
(adjacent f1-9f f1-8f left)
(adjacent f1-9f f1-10f right)
(adjacent f1-9f f0-9f up)
(adjacent f1-9f f2-9f down)
(adjacent f1-10f f1-9f left)
(adjacent f1-10f f0-10f up)
(adjacent f1-10f f2-10f down)
(adjacent f2-0f f2-1f right)
(adjacent f2-0f f1-0f up)
(adjacent f2-0f f3-0f down)
(adjacent f2-1f f2-0f left)
(adjacent f2-1f f2-2f right)
(adjacent f2-1f f1-1f up)
(adjacent f2-1f f3-1f down)
(adjacent f2-2f f2-1f left)
(adjacent f2-2f f2-3f right)
(adjacent f2-2f f1-2f up)
(adjacent f2-2f f3-2f down)
(adjacent f2-3f f2-2f left)
(adjacent f2-3f f2-4f right)
(adjacent f2-3f f1-3f up)
(adjacent f2-3f f3-3f down)
(adjacent f2-4f f2-3f left)
(adjacent f2-4f f2-5f right)
(adjacent f2-4f f1-4f up)
(adjacent f2-4f f3-4f down)
(adjacent f2-5f f2-4f left)
(adjacent f2-5f f2-6f right)
(adjacent f2-5f f1-5f up)
(adjacent f2-5f f3-5f down)
(adjacent f2-6f f2-5f left)
(adjacent f2-6f f2-7f right)
(adjacent f2-6f f1-6f up)
(adjacent f2-6f f3-6f down)
(adjacent f2-7f f2-6f left)
(adjacent f2-7f f2-8f right)
(adjacent f2-7f f1-7f up)
(adjacent f2-7f f3-7f down)
(adjacent f2-8f f2-7f left)
(adjacent f2-8f f2-9f right)
(adjacent f2-8f f1-8f up)
(adjacent f2-8f f3-8f down)
(adjacent f2-9f f2-8f left)
(adjacent f2-9f f2-10f right)
(adjacent f2-9f f1-9f up)
(adjacent f2-9f f3-9f down)
(adjacent f2-10f f2-9f left)
(adjacent f2-10f f1-10f up)
(adjacent f2-10f f3-10f down)
(adjacent f3-0f f3-1f right)
(adjacent f3-0f f2-0f up)
(adjacent f3-0f f4-0f down)
(adjacent f3-1f f3-0f left)
(adjacent f3-1f f3-2f right)
(adjacent f3-1f f2-1f up)
(adjacent f3-1f f4-1f down)
(adjacent f3-2f f3-1f left)
(adjacent f3-2f f3-3f right)
(adjacent f3-2f f2-2f up)
(adjacent f3-2f f4-2f down)
(adjacent f3-3f f3-2f left)
(adjacent f3-3f f3-4f right)
(adjacent f3-3f f2-3f up)
(adjacent f3-3f f4-3f down)
(adjacent f3-4f f3-3f left)
(adjacent f3-4f f3-5f right)
(adjacent f3-4f f2-4f up)
(adjacent f3-4f f4-4f down)
(adjacent f3-5f f3-4f left)
(adjacent f3-5f f3-6f right)
(adjacent f3-5f f2-5f up)
(adjacent f3-5f f4-5f down)
(adjacent f3-6f f3-5f left)
(adjacent f3-6f f3-7f right)
(adjacent f3-6f f2-6f up)
(adjacent f3-6f f4-6f down)
(adjacent f3-7f f3-6f left)
(adjacent f3-7f f3-8f right)
(adjacent f3-7f f2-7f up)
(adjacent f3-7f f4-7f down)
(adjacent f3-8f f3-7f left)
(adjacent f3-8f f3-9f right)
(adjacent f3-8f f2-8f up)
(adjacent f3-8f f4-8f down)
(adjacent f3-9f f3-8f left)
(adjacent f3-9f f3-10f right)
(adjacent f3-9f f2-9f up)
(adjacent f3-9f f4-9f down)
(adjacent f3-10f f3-9f left)
(adjacent f3-10f f2-10f up)
(adjacent f3-10f f4-10f down)
(adjacent f4-0f f4-1f right)
(adjacent f4-0f f3-0f up)
(adjacent f4-0f f5-0f down)
(adjacent f4-1f f4-0f left)
(adjacent f4-1f f4-2f right)
(adjacent f4-1f f3-1f up)
(adjacent f4-1f f5-1f down)
(adjacent f4-2f f4-1f left)
(adjacent f4-2f f4-3f right)
(adjacent f4-2f f3-2f up)
(adjacent f4-2f f5-2f down)
(adjacent f4-3f f4-2f left)
(adjacent f4-3f f4-4f right)
(adjacent f4-3f f3-3f up)
(adjacent f4-3f f5-3f down)
(adjacent f4-4f f4-3f left)
(adjacent f4-4f f4-5f right)
(adjacent f4-4f f3-4f up)
(adjacent f4-4f f5-4f down)
(adjacent f4-5f f4-4f left)
(adjacent f4-5f f4-6f right)
(adjacent f4-5f f3-5f up)
(adjacent f4-5f f5-5f down)
(adjacent f4-6f f4-5f left)
(adjacent f4-6f f4-7f right)
(adjacent f4-6f f3-6f up)
(adjacent f4-6f f5-6f down)
(adjacent f4-7f f4-6f left)
(adjacent f4-7f f4-8f right)
(adjacent f4-7f f3-7f up)
(adjacent f4-7f f5-7f down)
(adjacent f4-8f f4-7f left)
(adjacent f4-8f f4-9f right)
(adjacent f4-8f f3-8f up)
(adjacent f4-8f f5-8f down)
(adjacent f4-9f f4-8f left)
(adjacent f4-9f f4-10f right)
(adjacent f4-9f f3-9f up)
(adjacent f4-9f f5-9f down)
(adjacent f4-10f f4-9f left)
(adjacent f4-10f f3-10f up)
(adjacent f4-10f f5-10f down)
(adjacent f5-0f f5-1f right)
(adjacent f5-0f f4-0f up)
(adjacent f5-0f f6-0f down)
(adjacent f5-1f f5-0f left)
(adjacent f5-1f f5-2f right)
(adjacent f5-1f f4-1f up)
(adjacent f5-1f f6-1f down)
(adjacent f5-2f f5-1f left)
(adjacent f5-2f f5-3f right)
(adjacent f5-2f f4-2f up)
(adjacent f5-2f f6-2f down)
(adjacent f5-3f f5-2f left)
(adjacent f5-3f f5-4f right)
(adjacent f5-3f f4-3f up)
(adjacent f5-3f f6-3f down)
(adjacent f5-4f f5-3f left)
(adjacent f5-4f f5-5f right)
(adjacent f5-4f f4-4f up)
(adjacent f5-4f f6-4f down)
(adjacent f5-5f f5-4f left)
(adjacent f5-5f f5-6f right)
(adjacent f5-5f f4-5f up)
(adjacent f5-5f f6-5f down)
(adjacent f5-6f f5-5f left)
(adjacent f5-6f f5-7f right)
(adjacent f5-6f f4-6f up)
(adjacent f5-6f f6-6f down)
(adjacent f5-7f f5-6f left)
(adjacent f5-7f f5-8f right)
(adjacent f5-7f f4-7f up)
(adjacent f5-7f f6-7f down)
(adjacent f5-8f f5-7f left)
(adjacent f5-8f f5-9f right)
(adjacent f5-8f f4-8f up)
(adjacent f5-8f f6-8f down)
(adjacent f5-9f f5-8f left)
(adjacent f5-9f f5-10f right)
(adjacent f5-9f f4-9f up)
(adjacent f5-9f f6-9f down)
(adjacent f5-10f f5-9f left)
(adjacent f5-10f f4-10f up)
(adjacent f5-10f f6-10f down)
(adjacent f6-0f f6-1f right)
(adjacent f6-0f f5-0f up)
(adjacent f6-0f f7-0f down)
(adjacent f6-1f f6-0f left)
(adjacent f6-1f f6-2f right)
(adjacent f6-1f f5-1f up)
(adjacent f6-1f f7-1f down)
(adjacent f6-2f f6-1f left)
(adjacent f6-2f f6-3f right)
(adjacent f6-2f f5-2f up)
(adjacent f6-2f f7-2f down)
(adjacent f6-3f f6-2f left)
(adjacent f6-3f f6-4f right)
(adjacent f6-3f f5-3f up)
(adjacent f6-3f f7-3f down)
(adjacent f6-4f f6-3f left)
(adjacent f6-4f f6-5f right)
(adjacent f6-4f f5-4f up)
(adjacent f6-4f f7-4f down)
(adjacent f6-5f f6-4f left)
(adjacent f6-5f f6-6f right)
(adjacent f6-5f f5-5f up)
(adjacent f6-5f f7-5f down)
(adjacent f6-6f f6-5f left)
(adjacent f6-6f f6-7f right)
(adjacent f6-6f f5-6f up)
(adjacent f6-6f f7-6f down)
(adjacent f6-7f f6-6f left)
(adjacent f6-7f f6-8f right)
(adjacent f6-7f f5-7f up)
(adjacent f6-7f f7-7f down)
(adjacent f6-8f f6-7f left)
(adjacent f6-8f f6-9f right)
(adjacent f6-8f f5-8f up)
(adjacent f6-8f f7-8f down)
(adjacent f6-9f f6-8f left)
(adjacent f6-9f f6-10f right)
(adjacent f6-9f f5-9f up)
(adjacent f6-9f f7-9f down)
(adjacent f6-10f f6-9f left)
(adjacent f6-10f f5-10f up)
(adjacent f6-10f f7-10f down)
(adjacent f7-0f f7-1f right)
(adjacent f7-0f f6-0f up)
(adjacent f7-0f f8-0f down)
(adjacent f7-1f f7-0f left)
(adjacent f7-1f f7-2f right)
(adjacent f7-1f f6-1f up)
(adjacent f7-1f f8-1f down)
(adjacent f7-2f f7-1f left)
(adjacent f7-2f f7-3f right)
(adjacent f7-2f f6-2f up)
(adjacent f7-2f f8-2f down)
(adjacent f7-3f f7-2f left)
(adjacent f7-3f f7-4f right)
(adjacent f7-3f f6-3f up)
(adjacent f7-3f f8-3f down)
(adjacent f7-4f f7-3f left)
(adjacent f7-4f f7-5f right)
(adjacent f7-4f f6-4f up)
(adjacent f7-4f f8-4f down)
(adjacent f7-5f f7-4f left)
(adjacent f7-5f f7-6f right)
(adjacent f7-5f f6-5f up)
(adjacent f7-5f f8-5f down)
(adjacent f7-6f f7-5f left)
(adjacent f7-6f f7-7f right)
(adjacent f7-6f f6-6f up)
(adjacent f7-6f f8-6f down)
(adjacent f7-7f f7-6f left)
(adjacent f7-7f f7-8f right)
(adjacent f7-7f f6-7f up)
(adjacent f7-7f f8-7f down)
(adjacent f7-8f f7-7f left)
(adjacent f7-8f f7-9f right)
(adjacent f7-8f f6-8f up)
(adjacent f7-8f f8-8f down)
(adjacent f7-9f f7-8f left)
(adjacent f7-9f f7-10f right)
(adjacent f7-9f f6-9f up)
(adjacent f7-9f f8-9f down)
(adjacent f7-10f f7-9f left)
(adjacent f7-10f f6-10f up)
(adjacent f7-10f f8-10f down)
(adjacent f8-0f f8-1f right)
(adjacent f8-0f f7-0f up)
(adjacent f8-0f f9-0f down)
(adjacent f8-1f f8-0f left)
(adjacent f8-1f f8-2f right)
(adjacent f8-1f f7-1f up)
(adjacent f8-1f f9-1f down)
(adjacent f8-2f f8-1f left)
(adjacent f8-2f f8-3f right)
(adjacent f8-2f f7-2f up)
(adjacent f8-2f f9-2f down)
(adjacent f8-3f f8-2f left)
(adjacent f8-3f f8-4f right)
(adjacent f8-3f f7-3f up)
(adjacent f8-3f f9-3f down)
(adjacent f8-4f f8-3f left)
(adjacent f8-4f f8-5f right)
(adjacent f8-4f f7-4f up)
(adjacent f8-4f f9-4f down)
(adjacent f8-5f f8-4f left)
(adjacent f8-5f f8-6f right)
(adjacent f8-5f f7-5f up)
(adjacent f8-5f f9-5f down)
(adjacent f8-6f f8-5f left)
(adjacent f8-6f f8-7f right)
(adjacent f8-6f f7-6f up)
(adjacent f8-6f f9-6f down)
(adjacent f8-7f f8-6f left)
(adjacent f8-7f f8-8f right)
(adjacent f8-7f f7-7f up)
(adjacent f8-7f f9-7f down)
(adjacent f8-8f f8-7f left)
(adjacent f8-8f f8-9f right)
(adjacent f8-8f f7-8f up)
(adjacent f8-8f f9-8f down)
(adjacent f8-9f f8-8f left)
(adjacent f8-9f f8-10f right)
(adjacent f8-9f f7-9f up)
(adjacent f8-9f f9-9f down)
(adjacent f8-10f f8-9f left)
(adjacent f8-10f f7-10f up)
(adjacent f8-10f f9-10f down)
(adjacent f9-0f f9-1f right)
(adjacent f9-0f f8-0f up)
(adjacent f9-0f f10-0f down)
(adjacent f9-1f f9-0f left)
(adjacent f9-1f f9-2f right)
(adjacent f9-1f f8-1f up)
(adjacent f9-1f f10-1f down)
(adjacent f9-2f f9-1f left)
(adjacent f9-2f f9-3f right)
(adjacent f9-2f f8-2f up)
(adjacent f9-2f f10-2f down)
(adjacent f9-3f f9-2f left)
(adjacent f9-3f f9-4f right)
(adjacent f9-3f f8-3f up)
(adjacent f9-3f f10-3f down)
(adjacent f9-4f f9-3f left)
(adjacent f9-4f f9-5f right)
(adjacent f9-4f f8-4f up)
(adjacent f9-4f f10-4f down)
(adjacent f9-5f f9-4f left)
(adjacent f9-5f f9-6f right)
(adjacent f9-5f f8-5f up)
(adjacent f9-5f f10-5f down)
(adjacent f9-6f f9-5f left)
(adjacent f9-6f f9-7f right)
(adjacent f9-6f f8-6f up)
(adjacent f9-6f f10-6f down)
(adjacent f9-7f f9-6f left)
(adjacent f9-7f f9-8f right)
(adjacent f9-7f f8-7f up)
(adjacent f9-7f f10-7f down)
(adjacent f9-8f f9-7f left)
(adjacent f9-8f f9-9f right)
(adjacent f9-8f f8-8f up)
(adjacent f9-8f f10-8f down)
(adjacent f9-9f f9-8f left)
(adjacent f9-9f f9-10f right)
(adjacent f9-9f f8-9f up)
(adjacent f9-9f f10-9f down)
(adjacent f9-10f f9-9f left)
(adjacent f9-10f f8-10f up)
(adjacent f9-10f f10-10f down)
(adjacent f10-0f f10-1f right)
(adjacent f10-0f f9-0f up)
(adjacent f10-1f f10-0f left)
(adjacent f10-1f f10-2f right)
(adjacent f10-1f f9-1f up)
(adjacent f10-2f f10-1f left)
(adjacent f10-2f f10-3f right)
(adjacent f10-2f f9-2f up)
(adjacent f10-3f f10-2f left)
(adjacent f10-3f f10-4f right)
(adjacent f10-3f f9-3f up)
(adjacent f10-4f f10-3f left)
(adjacent f10-4f f10-5f right)
(adjacent f10-4f f9-4f up)
(adjacent f10-5f f10-4f left)
(adjacent f10-5f f10-6f right)
(adjacent f10-5f f9-5f up)
(adjacent f10-6f f10-5f left)
(adjacent f10-6f f10-7f right)
(adjacent f10-6f f9-6f up)
(adjacent f10-7f f10-6f left)
(adjacent f10-7f f10-8f right)
(adjacent f10-7f f9-7f up)
(adjacent f10-8f f10-7f left)
(adjacent f10-8f f10-9f right)
(adjacent f10-8f f9-8f up)
(adjacent f10-9f f10-8f left)
(adjacent f10-9f f10-10f right)
(adjacent f10-9f f9-9f up)
(adjacent f10-10f f10-9f left)
(adjacent f10-10f f9-10f up)
(at box0 f4-8f) 
(at box1 f7-4f) 
(clear f0-0f) 
(clear f0-1f) 
(clear f0-2f) 
(clear f0-3f) 
(clear f0-4f) 
(clear f0-5f) 
(clear f0-6f) 
(clear f0-7f) 
(clear f0-8f) 
(clear f0-9f) 
(clear f0-10f) 
(clear f1-0f) 
(clear f1-1f) 
(clear f1-2f) 
(clear f1-3f) 
(clear f1-4f) 
(clear f1-5f) 
(clear f1-6f) 
(clear f1-7f) 
(clear f1-8f) 
(clear f1-9f) 
(clear f1-10f) 
(clear f2-0f) 
(clear f2-1f) 
(clear f2-2f) 
(clear f2-3f) 
(clear f2-4f) 
(clear f2-5f) 
(clear f2-6f) 
(clear f2-7f) 
(clear f2-8f) 
(clear f2-9f) 
(clear f2-10f) 
(at-robot f3-0f) 
(clear f3-0f) 
(clear f3-1f) 
(clear f3-2f) 
(clear f3-3f) 
(clear f3-4f) 
(clear f3-5f) 
(clear f3-6f) 
(clear f3-7f) 
(clear f3-8f) 
(clear f3-9f) 
(clear f3-10f) 
(clear f4-0f) 
(clear f4-2f) 
(clear f4-4f) 
(clear f4-5f) 
(clear f4-6f) 
(clear f4-7f) 
(clear f4-9f) 
(clear f4-10f) 
(clear f5-0f) 
(clear f5-1f) 
(clear f5-4f) 
(clear f5-5f) 
(clear f5-6f) 
(clear f5-7f) 
(clear f5-8f) 
(clear f5-9f) 
(clear f5-10f) 
(clear f6-0f) 
(clear f6-1f) 
(clear f6-2f) 
(clear f6-3f) 
(clear f6-4f) 
(clear f6-5f) 
(clear f6-6f) 
(clear f6-8f) 
(clear f6-9f) 
(clear f6-10f) 
(clear f7-0f) 
(clear f7-1f) 
(clear f7-2f) 
(clear f7-3f) 
(clear f7-5f) 
(clear f7-6f) 
(clear f7-7f) 
(clear f7-8f) 
(clear f7-9f) 
(clear f7-10f) 
(clear f8-0f) 
(clear f8-1f) 
(clear f8-2f) 
(clear f8-3f) 
(clear f8-4f) 
(clear f8-5f) 
(clear f8-6f) 
(clear f8-7f) 
(clear f8-8f) 
(clear f8-9f) 
(clear f8-10f) 
(clear f9-0f) 
(clear f9-1f) 
(clear f9-2f) 
(clear f9-3f) 
(clear f9-4f) 
(clear f9-5f) 
(clear f9-6f) 
(clear f9-7f) 
(clear f9-8f) 
(clear f9-10f) 
(clear f10-0f) 
(clear f10-1f) 
(clear f10-2f) 
(clear f10-3f) 
(clear f10-4f) 
(clear f10-5f) 
(clear f10-6f) 
(clear f10-7f) 
(clear f10-8f) 
(clear f10-10f) 
)
(:goal
(and
(at box0 f9-10f) 
(at box1 f10-10f) 
)
)
)


;clearing memory: 
