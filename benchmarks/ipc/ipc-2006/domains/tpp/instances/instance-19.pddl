(define (problem grounded-STRIPS-TPP)
(:domain grounded-STRIPS-TPP-PROPOSITIONAL)
(:init
(AT-TRUCK5-DEPOT2)
(AT-TRUCK4-DEPOT1)
(AT-TRUCK3-DEPOT1)
(AT-TRUCK2-DEPOT1)
(AT-TRUCK1-DEPOT1)
(ON-SALE-GOODS14-MARKET5-LEVEL1)
(ON-SALE-GOODS14-MARKET4-LEVEL1)
(ON-SALE-GOODS13-MARKET4-LEVEL1)
(ON-SALE-GOODS12-MARKET4-LEVEL1)
(ON-SALE-GOODS11-MARKET4-LEVEL2)
(ON-SALE-GOODS9-MARKET4-LEVEL2)
(ON-SALE-GOODS8-MARKET4-LEVEL2)
(ON-SALE-GOODS7-MARKET4-LEVEL1)
(ON-SALE-GOODS6-MARKET4-LEVEL1)
(ON-SALE-GOODS4-MARKET4-LEVEL1)
(ON-SALE-GOODS3-MARKET4-LEVEL2)
(ON-SALE-GOODS2-MARKET4-LEVEL2)
(ON-SALE-GOODS1-MARKET4-LEVEL1)
(ON-SALE-GOODS14-MARKET3-LEVEL1)
(ON-SALE-GOODS13-MARKET3-LEVEL2)
(ON-SALE-GOODS10-MARKET3-LEVEL2)
(ON-SALE-GOODS9-MARKET3-LEVEL1)
(ON-SALE-GOODS7-MARKET3-LEVEL2)
(ON-SALE-GOODS6-MARKET3-LEVEL2)
(ON-SALE-GOODS12-MARKET2-LEVEL1)
(ON-SALE-GOODS8-MARKET2-LEVEL2)
(ON-SALE-GOODS6-MARKET2-LEVEL1)
(ON-SALE-GOODS5-MARKET2-LEVEL2)
(ON-SALE-GOODS3-MARKET2-LEVEL1)
(ON-SALE-GOODS2-MARKET2-LEVEL2)
(ON-SALE-GOODS1-MARKET2-LEVEL2)
(ON-SALE-GOODS14-MARKET1-LEVEL1)
(ON-SALE-GOODS13-MARKET1-LEVEL1)
(ON-SALE-GOODS12-MARKET1-LEVEL2)
(ON-SALE-GOODS11-MARKET1-LEVEL2)
(ON-SALE-GOODS10-MARKET1-LEVEL2)
(ON-SALE-GOODS9-MARKET1-LEVEL1)
(ON-SALE-GOODS7-MARKET1-LEVEL1)
(ON-SALE-GOODS5-MARKET1-LEVEL2)
(ON-SALE-GOODS4-MARKET1-LEVEL2)
(ON-SALE-GOODS1-MARKET1-LEVEL1)
(LOADED-GOODS14-TRUCK5-LEVEL0)
(LOADED-GOODS14-TRUCK4-LEVEL0)
(LOADED-GOODS14-TRUCK3-LEVEL0)
(LOADED-GOODS14-TRUCK2-LEVEL0)
(LOADED-GOODS14-TRUCK1-LEVEL0)
(LOADED-GOODS13-TRUCK5-LEVEL0)
(LOADED-GOODS13-TRUCK4-LEVEL0)
(LOADED-GOODS13-TRUCK3-LEVEL0)
(LOADED-GOODS13-TRUCK2-LEVEL0)
(LOADED-GOODS13-TRUCK1-LEVEL0)
(LOADED-GOODS12-TRUCK5-LEVEL0)
(LOADED-GOODS12-TRUCK4-LEVEL0)
(LOADED-GOODS12-TRUCK3-LEVEL0)
(LOADED-GOODS12-TRUCK2-LEVEL0)
(LOADED-GOODS12-TRUCK1-LEVEL0)
(LOADED-GOODS11-TRUCK5-LEVEL0)
(LOADED-GOODS11-TRUCK4-LEVEL0)
(LOADED-GOODS11-TRUCK3-LEVEL0)
(LOADED-GOODS11-TRUCK2-LEVEL0)
(LOADED-GOODS11-TRUCK1-LEVEL0)
(LOADED-GOODS10-TRUCK5-LEVEL0)
(LOADED-GOODS10-TRUCK4-LEVEL0)
(LOADED-GOODS10-TRUCK3-LEVEL0)
(LOADED-GOODS10-TRUCK2-LEVEL0)
(LOADED-GOODS10-TRUCK1-LEVEL0)
(LOADED-GOODS9-TRUCK5-LEVEL0)
(LOADED-GOODS9-TRUCK4-LEVEL0)
(LOADED-GOODS9-TRUCK3-LEVEL0)
(LOADED-GOODS9-TRUCK2-LEVEL0)
(LOADED-GOODS9-TRUCK1-LEVEL0)
(LOADED-GOODS8-TRUCK5-LEVEL0)
(LOADED-GOODS8-TRUCK4-LEVEL0)
(LOADED-GOODS8-TRUCK3-LEVEL0)
(LOADED-GOODS8-TRUCK2-LEVEL0)
(LOADED-GOODS8-TRUCK1-LEVEL0)
(LOADED-GOODS7-TRUCK5-LEVEL0)
(LOADED-GOODS7-TRUCK4-LEVEL0)
(LOADED-GOODS7-TRUCK3-LEVEL0)
(LOADED-GOODS7-TRUCK2-LEVEL0)
(LOADED-GOODS7-TRUCK1-LEVEL0)
(LOADED-GOODS6-TRUCK5-LEVEL0)
(LOADED-GOODS6-TRUCK4-LEVEL0)
(LOADED-GOODS6-TRUCK3-LEVEL0)
(LOADED-GOODS6-TRUCK2-LEVEL0)
(LOADED-GOODS6-TRUCK1-LEVEL0)
(LOADED-GOODS5-TRUCK5-LEVEL0)
(LOADED-GOODS5-TRUCK4-LEVEL0)
(LOADED-GOODS5-TRUCK3-LEVEL0)
(LOADED-GOODS5-TRUCK2-LEVEL0)
(LOADED-GOODS5-TRUCK1-LEVEL0)
(LOADED-GOODS4-TRUCK5-LEVEL0)
(LOADED-GOODS4-TRUCK4-LEVEL0)
(LOADED-GOODS4-TRUCK3-LEVEL0)
(LOADED-GOODS4-TRUCK2-LEVEL0)
(LOADED-GOODS4-TRUCK1-LEVEL0)
(LOADED-GOODS3-TRUCK5-LEVEL0)
(LOADED-GOODS3-TRUCK4-LEVEL0)
(LOADED-GOODS3-TRUCK3-LEVEL0)
(LOADED-GOODS3-TRUCK2-LEVEL0)
(LOADED-GOODS3-TRUCK1-LEVEL0)
(LOADED-GOODS2-TRUCK5-LEVEL0)
(LOADED-GOODS2-TRUCK4-LEVEL0)
(LOADED-GOODS2-TRUCK3-LEVEL0)
(LOADED-GOODS2-TRUCK2-LEVEL0)
(LOADED-GOODS2-TRUCK1-LEVEL0)
(LOADED-GOODS1-TRUCK5-LEVEL0)
(LOADED-GOODS1-TRUCK4-LEVEL0)
(LOADED-GOODS1-TRUCK3-LEVEL0)
(LOADED-GOODS1-TRUCK2-LEVEL0)
(LOADED-GOODS1-TRUCK1-LEVEL0)
(STORED-GOODS14-LEVEL0)
(STORED-GOODS13-LEVEL0)
(STORED-GOODS12-LEVEL0)
(STORED-GOODS11-LEVEL0)
(STORED-GOODS10-LEVEL0)
(STORED-GOODS9-LEVEL0)
(STORED-GOODS8-LEVEL0)
(STORED-GOODS7-LEVEL0)
(STORED-GOODS6-LEVEL0)
(STORED-GOODS5-LEVEL0)
(STORED-GOODS4-LEVEL0)
(STORED-GOODS3-LEVEL0)
(STORED-GOODS2-LEVEL0)
(STORED-GOODS1-LEVEL0)
(READY-TO-LOAD-GOODS14-MARKET5-LEVEL0)
(READY-TO-LOAD-GOODS14-MARKET4-LEVEL0)
(READY-TO-LOAD-GOODS14-MARKET3-LEVEL0)
(READY-TO-LOAD-GOODS14-MARKET1-LEVEL0)
(READY-TO-LOAD-GOODS13-MARKET4-LEVEL0)
(READY-TO-LOAD-GOODS13-MARKET3-LEVEL0)
(READY-TO-LOAD-GOODS13-MARKET1-LEVEL0)
(READY-TO-LOAD-GOODS12-MARKET4-LEVEL0)
(READY-TO-LOAD-GOODS12-MARKET2-LEVEL0)
(READY-TO-LOAD-GOODS12-MARKET1-LEVEL0)
(READY-TO-LOAD-GOODS11-MARKET4-LEVEL0)
(READY-TO-LOAD-GOODS11-MARKET1-LEVEL0)
(READY-TO-LOAD-GOODS10-MARKET3-LEVEL0)
(READY-TO-LOAD-GOODS10-MARKET1-LEVEL0)
(READY-TO-LOAD-GOODS9-MARKET4-LEVEL0)
(READY-TO-LOAD-GOODS9-MARKET3-LEVEL0)
(READY-TO-LOAD-GOODS9-MARKET1-LEVEL0)
(READY-TO-LOAD-GOODS8-MARKET4-LEVEL0)
(READY-TO-LOAD-GOODS8-MARKET2-LEVEL0)
(READY-TO-LOAD-GOODS7-MARKET4-LEVEL0)
(READY-TO-LOAD-GOODS7-MARKET3-LEVEL0)
(READY-TO-LOAD-GOODS7-MARKET1-LEVEL0)
(READY-TO-LOAD-GOODS6-MARKET4-LEVEL0)
(READY-TO-LOAD-GOODS6-MARKET3-LEVEL0)
(READY-TO-LOAD-GOODS6-MARKET2-LEVEL0)
(READY-TO-LOAD-GOODS5-MARKET2-LEVEL0)
(READY-TO-LOAD-GOODS5-MARKET1-LEVEL0)
(READY-TO-LOAD-GOODS4-MARKET4-LEVEL0)
(READY-TO-LOAD-GOODS4-MARKET1-LEVEL0)
(READY-TO-LOAD-GOODS3-MARKET4-LEVEL0)
(READY-TO-LOAD-GOODS3-MARKET2-LEVEL0)
(READY-TO-LOAD-GOODS2-MARKET4-LEVEL0)
(READY-TO-LOAD-GOODS2-MARKET2-LEVEL0)
(READY-TO-LOAD-GOODS1-MARKET4-LEVEL0)
(READY-TO-LOAD-GOODS1-MARKET2-LEVEL0)
(READY-TO-LOAD-GOODS1-MARKET1-LEVEL0)
)
(:goal
(and
(STORED-GOODS14-LEVEL2)
(STORED-GOODS13-LEVEL4)
(STORED-GOODS12-LEVEL2)
(STORED-GOODS11-LEVEL2)
(STORED-GOODS10-LEVEL1)
(STORED-GOODS9-LEVEL4)
(STORED-GOODS8-LEVEL3)
(STORED-GOODS7-LEVEL2)
(STORED-GOODS6-LEVEL4)
(STORED-GOODS5-LEVEL1)
(STORED-GOODS4-LEVEL2)
(STORED-GOODS3-LEVEL1)
(STORED-GOODS2-LEVEL1)
(STORED-GOODS1-LEVEL2)
)
)
)
