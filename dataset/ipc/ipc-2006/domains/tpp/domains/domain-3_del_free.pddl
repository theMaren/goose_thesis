( define ( domain grounded-STRIPS-TPP-PROPOSITIONAL ) ( :requirements :strips ) ( :predicates ( AT-TRUCK1-MARKET1 ) ( ON-SALE-GOODS1-MARKET1-LEVEL0 ) ( READY-TO-LOAD-GOODS1-MARKET1-LEVEL1 ) ( ON-SALE-GOODS2-MARKET1-LEVEL0 ) ( READY-TO-LOAD-GOODS2-MARKET1-LEVEL1 ) ( ON-SALE-GOODS3-MARKET1-LEVEL0 ) ( READY-TO-LOAD-GOODS3-MARKET1-LEVEL1 ) ( LOADED-GOODS1-TRUCK1-LEVEL1 ) ( LOADED-GOODS2-TRUCK1-LEVEL1 ) ( LOADED-GOODS3-TRUCK1-LEVEL1 ) ( STORED-GOODS1-LEVEL1 ) ( STORED-GOODS2-LEVEL1 ) ( STORED-GOODS3-LEVEL1 ) ( STORED-GOODS3-LEVEL0 ) ( STORED-GOODS2-LEVEL0 ) ( STORED-GOODS1-LEVEL0 ) ( LOADED-GOODS3-TRUCK1-LEVEL0 ) ( LOADED-GOODS2-TRUCK1-LEVEL0 ) ( LOADED-GOODS1-TRUCK1-LEVEL0 ) ( ON-SALE-GOODS3-MARKET1-LEVEL1 ) ( READY-TO-LOAD-GOODS3-MARKET1-LEVEL0 ) ( ON-SALE-GOODS2-MARKET1-LEVEL1 ) ( READY-TO-LOAD-GOODS2-MARKET1-LEVEL0 ) ( ON-SALE-GOODS1-MARKET1-LEVEL1 ) ( READY-TO-LOAD-GOODS1-MARKET1-LEVEL0 ) ( AT-TRUCK1-DEPOT1 ) ) ( :action UNLOAD-GOODS3-TRUCK1-DEPOT1-LEVEL0-LEVEL1-LEVEL0-LEVEL1 :parameters ( ) :precondition ( and ( STORED-GOODS3-LEVEL0 ) ( LOADED-GOODS3-TRUCK1-LEVEL1 ) ( AT-TRUCK1-DEPOT1 ) ) :effect ( and ( LOADED-GOODS3-TRUCK1-LEVEL0 ) ( STORED-GOODS3-LEVEL1 ) ) ) ( :action UNLOAD-GOODS2-TRUCK1-DEPOT1-LEVEL0-LEVEL1-LEVEL0-LEVEL1 :parameters ( ) :precondition ( and ( STORED-GOODS2-LEVEL0 ) ( LOADED-GOODS2-TRUCK1-LEVEL1 ) ( AT-TRUCK1-DEPOT1 ) ) :effect ( and ( LOADED-GOODS2-TRUCK1-LEVEL0 ) ( STORED-GOODS2-LEVEL1 ) ) ) ( :action UNLOAD-GOODS1-TRUCK1-DEPOT1-LEVEL0-LEVEL1-LEVEL0-LEVEL1 :parameters ( ) :precondition ( and ( STORED-GOODS1-LEVEL0 ) ( LOADED-GOODS1-TRUCK1-LEVEL1 ) ( AT-TRUCK1-DEPOT1 ) ) :effect ( and ( LOADED-GOODS1-TRUCK1-LEVEL0 ) ( STORED-GOODS1-LEVEL1 ) ) ) ( :action LOAD-GOODS3-TRUCK1-MARKET1-LEVEL0-LEVEL1-LEVEL0-LEVEL1 :parameters ( ) :precondition ( and ( READY-TO-LOAD-GOODS3-MARKET1-LEVEL1 ) ( LOADED-GOODS3-TRUCK1-LEVEL0 ) ( AT-TRUCK1-MARKET1 ) ) :effect ( and ( LOADED-GOODS3-TRUCK1-LEVEL1 ) ( READY-TO-LOAD-GOODS3-MARKET1-LEVEL0 ) ) ) ( :action LOAD-GOODS2-TRUCK1-MARKET1-LEVEL0-LEVEL1-LEVEL0-LEVEL1 :parameters ( ) :precondition ( and ( READY-TO-LOAD-GOODS2-MARKET1-LEVEL1 ) ( LOADED-GOODS2-TRUCK1-LEVEL0 ) ( AT-TRUCK1-MARKET1 ) ) :effect ( and ( LOADED-GOODS2-TRUCK1-LEVEL1 ) ( READY-TO-LOAD-GOODS2-MARKET1-LEVEL0 ) ) ) ( :action LOAD-GOODS1-TRUCK1-MARKET1-LEVEL0-LEVEL1-LEVEL0-LEVEL1 :parameters ( ) :precondition ( and ( READY-TO-LOAD-GOODS1-MARKET1-LEVEL1 ) ( LOADED-GOODS1-TRUCK1-LEVEL0 ) ( AT-TRUCK1-MARKET1 ) ) :effect ( and ( LOADED-GOODS1-TRUCK1-LEVEL1 ) ( READY-TO-LOAD-GOODS1-MARKET1-LEVEL0 ) ) ) ( :action DRIVE-TRUCK1-MARKET1-DEPOT1 :parameters ( ) :precondition ( and ( AT-TRUCK1-MARKET1 ) ) :effect ( and ( AT-TRUCK1-DEPOT1 ) ) ) ( :action BUY-TRUCK1-GOODS3-MARKET1-LEVEL0-LEVEL1-LEVEL0-LEVEL1 :parameters ( ) :precondition ( and ( READY-TO-LOAD-GOODS3-MARKET1-LEVEL0 ) ( ON-SALE-GOODS3-MARKET1-LEVEL1 ) ( AT-TRUCK1-MARKET1 ) ) :effect ( and ( ON-SALE-GOODS3-MARKET1-LEVEL0 ) ( READY-TO-LOAD-GOODS3-MARKET1-LEVEL1 ) ) ) ( :action BUY-TRUCK1-GOODS2-MARKET1-LEVEL0-LEVEL1-LEVEL0-LEVEL1 :parameters ( ) :precondition ( and ( READY-TO-LOAD-GOODS2-MARKET1-LEVEL0 ) ( ON-SALE-GOODS2-MARKET1-LEVEL1 ) ( AT-TRUCK1-MARKET1 ) ) :effect ( and ( ON-SALE-GOODS2-MARKET1-LEVEL0 ) ( READY-TO-LOAD-GOODS2-MARKET1-LEVEL1 ) ) ) ( :action BUY-TRUCK1-GOODS1-MARKET1-LEVEL0-LEVEL1-LEVEL0-LEVEL1 :parameters ( ) :precondition ( and ( READY-TO-LOAD-GOODS1-MARKET1-LEVEL0 ) ( ON-SALE-GOODS1-MARKET1-LEVEL1 ) ( AT-TRUCK1-MARKET1 ) ) :effect ( and ( ON-SALE-GOODS1-MARKET1-LEVEL0 ) ( READY-TO-LOAD-GOODS1-MARKET1-LEVEL1 ) ) ) ( :action DRIVE-TRUCK1-DEPOT1-MARKET1 :parameters ( ) :precondition ( and ( AT-TRUCK1-DEPOT1 ) ) :effect ( and ( AT-TRUCK1-MARKET1 ) ) ) )