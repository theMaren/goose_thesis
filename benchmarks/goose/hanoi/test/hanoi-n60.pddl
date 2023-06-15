(define (problem hanoi-60)
  (:domain hanoi-domain)
  (:objects peg1 peg2 peg3 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15 d16 d17 d18 d19 d20 d21 d22 d23 d24 d25 d26 d27 d28 d29 d30 d31 d32 d33 d34 d35 d36 d37 d38 d39 d40 d41 d42 d43 d44 d45 d46 d47 d48 d49 d50 d51 d52 d53 d54 d55 d56 d57 d58 d59 d60 )
  (:init 
    (smaller d1 peg1)(smaller d1 peg2)(smaller d1 peg3)
    (smaller d2 peg1)(smaller d2 peg2)(smaller d2 peg3)
    (smaller d3 peg1)(smaller d3 peg2)(smaller d3 peg3)
    (smaller d4 peg1)(smaller d4 peg2)(smaller d4 peg3)
    (smaller d5 peg1)(smaller d5 peg2)(smaller d5 peg3)
    (smaller d6 peg1)(smaller d6 peg2)(smaller d6 peg3)
    (smaller d7 peg1)(smaller d7 peg2)(smaller d7 peg3)
    (smaller d8 peg1)(smaller d8 peg2)(smaller d8 peg3)
    (smaller d9 peg1)(smaller d9 peg2)(smaller d9 peg3)
    (smaller d10 peg1)(smaller d10 peg2)(smaller d10 peg3)
    (smaller d11 peg1)(smaller d11 peg2)(smaller d11 peg3)
    (smaller d12 peg1)(smaller d12 peg2)(smaller d12 peg3)
    (smaller d13 peg1)(smaller d13 peg2)(smaller d13 peg3)
    (smaller d14 peg1)(smaller d14 peg2)(smaller d14 peg3)
    (smaller d15 peg1)(smaller d15 peg2)(smaller d15 peg3)
    (smaller d16 peg1)(smaller d16 peg2)(smaller d16 peg3)
    (smaller d17 peg1)(smaller d17 peg2)(smaller d17 peg3)
    (smaller d18 peg1)(smaller d18 peg2)(smaller d18 peg3)
    (smaller d19 peg1)(smaller d19 peg2)(smaller d19 peg3)
    (smaller d20 peg1)(smaller d20 peg2)(smaller d20 peg3)
    (smaller d21 peg1)(smaller d21 peg2)(smaller d21 peg3)
    (smaller d22 peg1)(smaller d22 peg2)(smaller d22 peg3)
    (smaller d23 peg1)(smaller d23 peg2)(smaller d23 peg3)
    (smaller d24 peg1)(smaller d24 peg2)(smaller d24 peg3)
    (smaller d25 peg1)(smaller d25 peg2)(smaller d25 peg3)
    (smaller d26 peg1)(smaller d26 peg2)(smaller d26 peg3)
    (smaller d27 peg1)(smaller d27 peg2)(smaller d27 peg3)
    (smaller d28 peg1)(smaller d28 peg2)(smaller d28 peg3)
    (smaller d29 peg1)(smaller d29 peg2)(smaller d29 peg3)
    (smaller d30 peg1)(smaller d30 peg2)(smaller d30 peg3)
    (smaller d31 peg1)(smaller d31 peg2)(smaller d31 peg3)
    (smaller d32 peg1)(smaller d32 peg2)(smaller d32 peg3)
    (smaller d33 peg1)(smaller d33 peg2)(smaller d33 peg3)
    (smaller d34 peg1)(smaller d34 peg2)(smaller d34 peg3)
    (smaller d35 peg1)(smaller d35 peg2)(smaller d35 peg3)
    (smaller d36 peg1)(smaller d36 peg2)(smaller d36 peg3)
    (smaller d37 peg1)(smaller d37 peg2)(smaller d37 peg3)
    (smaller d38 peg1)(smaller d38 peg2)(smaller d38 peg3)
    (smaller d39 peg1)(smaller d39 peg2)(smaller d39 peg3)
    (smaller d40 peg1)(smaller d40 peg2)(smaller d40 peg3)
    (smaller d41 peg1)(smaller d41 peg2)(smaller d41 peg3)
    (smaller d42 peg1)(smaller d42 peg2)(smaller d42 peg3)
    (smaller d43 peg1)(smaller d43 peg2)(smaller d43 peg3)
    (smaller d44 peg1)(smaller d44 peg2)(smaller d44 peg3)
    (smaller d45 peg1)(smaller d45 peg2)(smaller d45 peg3)
    (smaller d46 peg1)(smaller d46 peg2)(smaller d46 peg3)
    (smaller d47 peg1)(smaller d47 peg2)(smaller d47 peg3)
    (smaller d48 peg1)(smaller d48 peg2)(smaller d48 peg3)
    (smaller d49 peg1)(smaller d49 peg2)(smaller d49 peg3)
    (smaller d50 peg1)(smaller d50 peg2)(smaller d50 peg3)
    (smaller d51 peg1)(smaller d51 peg2)(smaller d51 peg3)
    (smaller d52 peg1)(smaller d52 peg2)(smaller d52 peg3)
    (smaller d53 peg1)(smaller d53 peg2)(smaller d53 peg3)
    (smaller d54 peg1)(smaller d54 peg2)(smaller d54 peg3)
    (smaller d55 peg1)(smaller d55 peg2)(smaller d55 peg3)
    (smaller d56 peg1)(smaller d56 peg2)(smaller d56 peg3)
    (smaller d57 peg1)(smaller d57 peg2)(smaller d57 peg3)
    (smaller d58 peg1)(smaller d58 peg2)(smaller d58 peg3)
    (smaller d59 peg1)(smaller d59 peg2)(smaller d59 peg3)
    (smaller d60 peg1)(smaller d60 peg2)(smaller d60 peg3)

    (smaller d1 d2)(smaller d1 d3)(smaller d1 d4)(smaller d1 d5)(smaller d1 d6)(smaller d1 d7)(smaller d1 d8)(smaller d1 d9)(smaller d1 d10)(smaller d1 d11)(smaller d1 d12)(smaller d1 d13)(smaller d1 d14)(smaller d1 d15)(smaller d1 d16)(smaller d1 d17)(smaller d1 d18)(smaller d1 d19)(smaller d1 d20)(smaller d1 d21)(smaller d1 d22)(smaller d1 d23)(smaller d1 d24)(smaller d1 d25)(smaller d1 d26)(smaller d1 d27)(smaller d1 d28)(smaller d1 d29)(smaller d1 d30)(smaller d1 d31)(smaller d1 d32)(smaller d1 d33)(smaller d1 d34)(smaller d1 d35)(smaller d1 d36)(smaller d1 d37)(smaller d1 d38)(smaller d1 d39)(smaller d1 d40)(smaller d1 d41)(smaller d1 d42)(smaller d1 d43)(smaller d1 d44)(smaller d1 d45)(smaller d1 d46)(smaller d1 d47)(smaller d1 d48)(smaller d1 d49)(smaller d1 d50)(smaller d1 d51)(smaller d1 d52)(smaller d1 d53)(smaller d1 d54)(smaller d1 d55)(smaller d1 d56)(smaller d1 d57)(smaller d1 d58)(smaller d1 d59)(smaller d1 d60)
    (smaller d2 d3)(smaller d2 d4)(smaller d2 d5)(smaller d2 d6)(smaller d2 d7)(smaller d2 d8)(smaller d2 d9)(smaller d2 d10)(smaller d2 d11)(smaller d2 d12)(smaller d2 d13)(smaller d2 d14)(smaller d2 d15)(smaller d2 d16)(smaller d2 d17)(smaller d2 d18)(smaller d2 d19)(smaller d2 d20)(smaller d2 d21)(smaller d2 d22)(smaller d2 d23)(smaller d2 d24)(smaller d2 d25)(smaller d2 d26)(smaller d2 d27)(smaller d2 d28)(smaller d2 d29)(smaller d2 d30)(smaller d2 d31)(smaller d2 d32)(smaller d2 d33)(smaller d2 d34)(smaller d2 d35)(smaller d2 d36)(smaller d2 d37)(smaller d2 d38)(smaller d2 d39)(smaller d2 d40)(smaller d2 d41)(smaller d2 d42)(smaller d2 d43)(smaller d2 d44)(smaller d2 d45)(smaller d2 d46)(smaller d2 d47)(smaller d2 d48)(smaller d2 d49)(smaller d2 d50)(smaller d2 d51)(smaller d2 d52)(smaller d2 d53)(smaller d2 d54)(smaller d2 d55)(smaller d2 d56)(smaller d2 d57)(smaller d2 d58)(smaller d2 d59)(smaller d2 d60)
    (smaller d3 d4)(smaller d3 d5)(smaller d3 d6)(smaller d3 d7)(smaller d3 d8)(smaller d3 d9)(smaller d3 d10)(smaller d3 d11)(smaller d3 d12)(smaller d3 d13)(smaller d3 d14)(smaller d3 d15)(smaller d3 d16)(smaller d3 d17)(smaller d3 d18)(smaller d3 d19)(smaller d3 d20)(smaller d3 d21)(smaller d3 d22)(smaller d3 d23)(smaller d3 d24)(smaller d3 d25)(smaller d3 d26)(smaller d3 d27)(smaller d3 d28)(smaller d3 d29)(smaller d3 d30)(smaller d3 d31)(smaller d3 d32)(smaller d3 d33)(smaller d3 d34)(smaller d3 d35)(smaller d3 d36)(smaller d3 d37)(smaller d3 d38)(smaller d3 d39)(smaller d3 d40)(smaller d3 d41)(smaller d3 d42)(smaller d3 d43)(smaller d3 d44)(smaller d3 d45)(smaller d3 d46)(smaller d3 d47)(smaller d3 d48)(smaller d3 d49)(smaller d3 d50)(smaller d3 d51)(smaller d3 d52)(smaller d3 d53)(smaller d3 d54)(smaller d3 d55)(smaller d3 d56)(smaller d3 d57)(smaller d3 d58)(smaller d3 d59)(smaller d3 d60)
    (smaller d4 d5)(smaller d4 d6)(smaller d4 d7)(smaller d4 d8)(smaller d4 d9)(smaller d4 d10)(smaller d4 d11)(smaller d4 d12)(smaller d4 d13)(smaller d4 d14)(smaller d4 d15)(smaller d4 d16)(smaller d4 d17)(smaller d4 d18)(smaller d4 d19)(smaller d4 d20)(smaller d4 d21)(smaller d4 d22)(smaller d4 d23)(smaller d4 d24)(smaller d4 d25)(smaller d4 d26)(smaller d4 d27)(smaller d4 d28)(smaller d4 d29)(smaller d4 d30)(smaller d4 d31)(smaller d4 d32)(smaller d4 d33)(smaller d4 d34)(smaller d4 d35)(smaller d4 d36)(smaller d4 d37)(smaller d4 d38)(smaller d4 d39)(smaller d4 d40)(smaller d4 d41)(smaller d4 d42)(smaller d4 d43)(smaller d4 d44)(smaller d4 d45)(smaller d4 d46)(smaller d4 d47)(smaller d4 d48)(smaller d4 d49)(smaller d4 d50)(smaller d4 d51)(smaller d4 d52)(smaller d4 d53)(smaller d4 d54)(smaller d4 d55)(smaller d4 d56)(smaller d4 d57)(smaller d4 d58)(smaller d4 d59)(smaller d4 d60)
    (smaller d5 d6)(smaller d5 d7)(smaller d5 d8)(smaller d5 d9)(smaller d5 d10)(smaller d5 d11)(smaller d5 d12)(smaller d5 d13)(smaller d5 d14)(smaller d5 d15)(smaller d5 d16)(smaller d5 d17)(smaller d5 d18)(smaller d5 d19)(smaller d5 d20)(smaller d5 d21)(smaller d5 d22)(smaller d5 d23)(smaller d5 d24)(smaller d5 d25)(smaller d5 d26)(smaller d5 d27)(smaller d5 d28)(smaller d5 d29)(smaller d5 d30)(smaller d5 d31)(smaller d5 d32)(smaller d5 d33)(smaller d5 d34)(smaller d5 d35)(smaller d5 d36)(smaller d5 d37)(smaller d5 d38)(smaller d5 d39)(smaller d5 d40)(smaller d5 d41)(smaller d5 d42)(smaller d5 d43)(smaller d5 d44)(smaller d5 d45)(smaller d5 d46)(smaller d5 d47)(smaller d5 d48)(smaller d5 d49)(smaller d5 d50)(smaller d5 d51)(smaller d5 d52)(smaller d5 d53)(smaller d5 d54)(smaller d5 d55)(smaller d5 d56)(smaller d5 d57)(smaller d5 d58)(smaller d5 d59)(smaller d5 d60)
    (smaller d6 d7)(smaller d6 d8)(smaller d6 d9)(smaller d6 d10)(smaller d6 d11)(smaller d6 d12)(smaller d6 d13)(smaller d6 d14)(smaller d6 d15)(smaller d6 d16)(smaller d6 d17)(smaller d6 d18)(smaller d6 d19)(smaller d6 d20)(smaller d6 d21)(smaller d6 d22)(smaller d6 d23)(smaller d6 d24)(smaller d6 d25)(smaller d6 d26)(smaller d6 d27)(smaller d6 d28)(smaller d6 d29)(smaller d6 d30)(smaller d6 d31)(smaller d6 d32)(smaller d6 d33)(smaller d6 d34)(smaller d6 d35)(smaller d6 d36)(smaller d6 d37)(smaller d6 d38)(smaller d6 d39)(smaller d6 d40)(smaller d6 d41)(smaller d6 d42)(smaller d6 d43)(smaller d6 d44)(smaller d6 d45)(smaller d6 d46)(smaller d6 d47)(smaller d6 d48)(smaller d6 d49)(smaller d6 d50)(smaller d6 d51)(smaller d6 d52)(smaller d6 d53)(smaller d6 d54)(smaller d6 d55)(smaller d6 d56)(smaller d6 d57)(smaller d6 d58)(smaller d6 d59)(smaller d6 d60)
    (smaller d7 d8)(smaller d7 d9)(smaller d7 d10)(smaller d7 d11)(smaller d7 d12)(smaller d7 d13)(smaller d7 d14)(smaller d7 d15)(smaller d7 d16)(smaller d7 d17)(smaller d7 d18)(smaller d7 d19)(smaller d7 d20)(smaller d7 d21)(smaller d7 d22)(smaller d7 d23)(smaller d7 d24)(smaller d7 d25)(smaller d7 d26)(smaller d7 d27)(smaller d7 d28)(smaller d7 d29)(smaller d7 d30)(smaller d7 d31)(smaller d7 d32)(smaller d7 d33)(smaller d7 d34)(smaller d7 d35)(smaller d7 d36)(smaller d7 d37)(smaller d7 d38)(smaller d7 d39)(smaller d7 d40)(smaller d7 d41)(smaller d7 d42)(smaller d7 d43)(smaller d7 d44)(smaller d7 d45)(smaller d7 d46)(smaller d7 d47)(smaller d7 d48)(smaller d7 d49)(smaller d7 d50)(smaller d7 d51)(smaller d7 d52)(smaller d7 d53)(smaller d7 d54)(smaller d7 d55)(smaller d7 d56)(smaller d7 d57)(smaller d7 d58)(smaller d7 d59)(smaller d7 d60)
    (smaller d8 d9)(smaller d8 d10)(smaller d8 d11)(smaller d8 d12)(smaller d8 d13)(smaller d8 d14)(smaller d8 d15)(smaller d8 d16)(smaller d8 d17)(smaller d8 d18)(smaller d8 d19)(smaller d8 d20)(smaller d8 d21)(smaller d8 d22)(smaller d8 d23)(smaller d8 d24)(smaller d8 d25)(smaller d8 d26)(smaller d8 d27)(smaller d8 d28)(smaller d8 d29)(smaller d8 d30)(smaller d8 d31)(smaller d8 d32)(smaller d8 d33)(smaller d8 d34)(smaller d8 d35)(smaller d8 d36)(smaller d8 d37)(smaller d8 d38)(smaller d8 d39)(smaller d8 d40)(smaller d8 d41)(smaller d8 d42)(smaller d8 d43)(smaller d8 d44)(smaller d8 d45)(smaller d8 d46)(smaller d8 d47)(smaller d8 d48)(smaller d8 d49)(smaller d8 d50)(smaller d8 d51)(smaller d8 d52)(smaller d8 d53)(smaller d8 d54)(smaller d8 d55)(smaller d8 d56)(smaller d8 d57)(smaller d8 d58)(smaller d8 d59)(smaller d8 d60)
    (smaller d9 d10)(smaller d9 d11)(smaller d9 d12)(smaller d9 d13)(smaller d9 d14)(smaller d9 d15)(smaller d9 d16)(smaller d9 d17)(smaller d9 d18)(smaller d9 d19)(smaller d9 d20)(smaller d9 d21)(smaller d9 d22)(smaller d9 d23)(smaller d9 d24)(smaller d9 d25)(smaller d9 d26)(smaller d9 d27)(smaller d9 d28)(smaller d9 d29)(smaller d9 d30)(smaller d9 d31)(smaller d9 d32)(smaller d9 d33)(smaller d9 d34)(smaller d9 d35)(smaller d9 d36)(smaller d9 d37)(smaller d9 d38)(smaller d9 d39)(smaller d9 d40)(smaller d9 d41)(smaller d9 d42)(smaller d9 d43)(smaller d9 d44)(smaller d9 d45)(smaller d9 d46)(smaller d9 d47)(smaller d9 d48)(smaller d9 d49)(smaller d9 d50)(smaller d9 d51)(smaller d9 d52)(smaller d9 d53)(smaller d9 d54)(smaller d9 d55)(smaller d9 d56)(smaller d9 d57)(smaller d9 d58)(smaller d9 d59)(smaller d9 d60)
    (smaller d10 d11)(smaller d10 d12)(smaller d10 d13)(smaller d10 d14)(smaller d10 d15)(smaller d10 d16)(smaller d10 d17)(smaller d10 d18)(smaller d10 d19)(smaller d10 d20)(smaller d10 d21)(smaller d10 d22)(smaller d10 d23)(smaller d10 d24)(smaller d10 d25)(smaller d10 d26)(smaller d10 d27)(smaller d10 d28)(smaller d10 d29)(smaller d10 d30)(smaller d10 d31)(smaller d10 d32)(smaller d10 d33)(smaller d10 d34)(smaller d10 d35)(smaller d10 d36)(smaller d10 d37)(smaller d10 d38)(smaller d10 d39)(smaller d10 d40)(smaller d10 d41)(smaller d10 d42)(smaller d10 d43)(smaller d10 d44)(smaller d10 d45)(smaller d10 d46)(smaller d10 d47)(smaller d10 d48)(smaller d10 d49)(smaller d10 d50)(smaller d10 d51)(smaller d10 d52)(smaller d10 d53)(smaller d10 d54)(smaller d10 d55)(smaller d10 d56)(smaller d10 d57)(smaller d10 d58)(smaller d10 d59)(smaller d10 d60)
    (smaller d11 d12)(smaller d11 d13)(smaller d11 d14)(smaller d11 d15)(smaller d11 d16)(smaller d11 d17)(smaller d11 d18)(smaller d11 d19)(smaller d11 d20)(smaller d11 d21)(smaller d11 d22)(smaller d11 d23)(smaller d11 d24)(smaller d11 d25)(smaller d11 d26)(smaller d11 d27)(smaller d11 d28)(smaller d11 d29)(smaller d11 d30)(smaller d11 d31)(smaller d11 d32)(smaller d11 d33)(smaller d11 d34)(smaller d11 d35)(smaller d11 d36)(smaller d11 d37)(smaller d11 d38)(smaller d11 d39)(smaller d11 d40)(smaller d11 d41)(smaller d11 d42)(smaller d11 d43)(smaller d11 d44)(smaller d11 d45)(smaller d11 d46)(smaller d11 d47)(smaller d11 d48)(smaller d11 d49)(smaller d11 d50)(smaller d11 d51)(smaller d11 d52)(smaller d11 d53)(smaller d11 d54)(smaller d11 d55)(smaller d11 d56)(smaller d11 d57)(smaller d11 d58)(smaller d11 d59)(smaller d11 d60)
    (smaller d12 d13)(smaller d12 d14)(smaller d12 d15)(smaller d12 d16)(smaller d12 d17)(smaller d12 d18)(smaller d12 d19)(smaller d12 d20)(smaller d12 d21)(smaller d12 d22)(smaller d12 d23)(smaller d12 d24)(smaller d12 d25)(smaller d12 d26)(smaller d12 d27)(smaller d12 d28)(smaller d12 d29)(smaller d12 d30)(smaller d12 d31)(smaller d12 d32)(smaller d12 d33)(smaller d12 d34)(smaller d12 d35)(smaller d12 d36)(smaller d12 d37)(smaller d12 d38)(smaller d12 d39)(smaller d12 d40)(smaller d12 d41)(smaller d12 d42)(smaller d12 d43)(smaller d12 d44)(smaller d12 d45)(smaller d12 d46)(smaller d12 d47)(smaller d12 d48)(smaller d12 d49)(smaller d12 d50)(smaller d12 d51)(smaller d12 d52)(smaller d12 d53)(smaller d12 d54)(smaller d12 d55)(smaller d12 d56)(smaller d12 d57)(smaller d12 d58)(smaller d12 d59)(smaller d12 d60)
    (smaller d13 d14)(smaller d13 d15)(smaller d13 d16)(smaller d13 d17)(smaller d13 d18)(smaller d13 d19)(smaller d13 d20)(smaller d13 d21)(smaller d13 d22)(smaller d13 d23)(smaller d13 d24)(smaller d13 d25)(smaller d13 d26)(smaller d13 d27)(smaller d13 d28)(smaller d13 d29)(smaller d13 d30)(smaller d13 d31)(smaller d13 d32)(smaller d13 d33)(smaller d13 d34)(smaller d13 d35)(smaller d13 d36)(smaller d13 d37)(smaller d13 d38)(smaller d13 d39)(smaller d13 d40)(smaller d13 d41)(smaller d13 d42)(smaller d13 d43)(smaller d13 d44)(smaller d13 d45)(smaller d13 d46)(smaller d13 d47)(smaller d13 d48)(smaller d13 d49)(smaller d13 d50)(smaller d13 d51)(smaller d13 d52)(smaller d13 d53)(smaller d13 d54)(smaller d13 d55)(smaller d13 d56)(smaller d13 d57)(smaller d13 d58)(smaller d13 d59)(smaller d13 d60)
    (smaller d14 d15)(smaller d14 d16)(smaller d14 d17)(smaller d14 d18)(smaller d14 d19)(smaller d14 d20)(smaller d14 d21)(smaller d14 d22)(smaller d14 d23)(smaller d14 d24)(smaller d14 d25)(smaller d14 d26)(smaller d14 d27)(smaller d14 d28)(smaller d14 d29)(smaller d14 d30)(smaller d14 d31)(smaller d14 d32)(smaller d14 d33)(smaller d14 d34)(smaller d14 d35)(smaller d14 d36)(smaller d14 d37)(smaller d14 d38)(smaller d14 d39)(smaller d14 d40)(smaller d14 d41)(smaller d14 d42)(smaller d14 d43)(smaller d14 d44)(smaller d14 d45)(smaller d14 d46)(smaller d14 d47)(smaller d14 d48)(smaller d14 d49)(smaller d14 d50)(smaller d14 d51)(smaller d14 d52)(smaller d14 d53)(smaller d14 d54)(smaller d14 d55)(smaller d14 d56)(smaller d14 d57)(smaller d14 d58)(smaller d14 d59)(smaller d14 d60)
    (smaller d15 d16)(smaller d15 d17)(smaller d15 d18)(smaller d15 d19)(smaller d15 d20)(smaller d15 d21)(smaller d15 d22)(smaller d15 d23)(smaller d15 d24)(smaller d15 d25)(smaller d15 d26)(smaller d15 d27)(smaller d15 d28)(smaller d15 d29)(smaller d15 d30)(smaller d15 d31)(smaller d15 d32)(smaller d15 d33)(smaller d15 d34)(smaller d15 d35)(smaller d15 d36)(smaller d15 d37)(smaller d15 d38)(smaller d15 d39)(smaller d15 d40)(smaller d15 d41)(smaller d15 d42)(smaller d15 d43)(smaller d15 d44)(smaller d15 d45)(smaller d15 d46)(smaller d15 d47)(smaller d15 d48)(smaller d15 d49)(smaller d15 d50)(smaller d15 d51)(smaller d15 d52)(smaller d15 d53)(smaller d15 d54)(smaller d15 d55)(smaller d15 d56)(smaller d15 d57)(smaller d15 d58)(smaller d15 d59)(smaller d15 d60)
    (smaller d16 d17)(smaller d16 d18)(smaller d16 d19)(smaller d16 d20)(smaller d16 d21)(smaller d16 d22)(smaller d16 d23)(smaller d16 d24)(smaller d16 d25)(smaller d16 d26)(smaller d16 d27)(smaller d16 d28)(smaller d16 d29)(smaller d16 d30)(smaller d16 d31)(smaller d16 d32)(smaller d16 d33)(smaller d16 d34)(smaller d16 d35)(smaller d16 d36)(smaller d16 d37)(smaller d16 d38)(smaller d16 d39)(smaller d16 d40)(smaller d16 d41)(smaller d16 d42)(smaller d16 d43)(smaller d16 d44)(smaller d16 d45)(smaller d16 d46)(smaller d16 d47)(smaller d16 d48)(smaller d16 d49)(smaller d16 d50)(smaller d16 d51)(smaller d16 d52)(smaller d16 d53)(smaller d16 d54)(smaller d16 d55)(smaller d16 d56)(smaller d16 d57)(smaller d16 d58)(smaller d16 d59)(smaller d16 d60)
    (smaller d17 d18)(smaller d17 d19)(smaller d17 d20)(smaller d17 d21)(smaller d17 d22)(smaller d17 d23)(smaller d17 d24)(smaller d17 d25)(smaller d17 d26)(smaller d17 d27)(smaller d17 d28)(smaller d17 d29)(smaller d17 d30)(smaller d17 d31)(smaller d17 d32)(smaller d17 d33)(smaller d17 d34)(smaller d17 d35)(smaller d17 d36)(smaller d17 d37)(smaller d17 d38)(smaller d17 d39)(smaller d17 d40)(smaller d17 d41)(smaller d17 d42)(smaller d17 d43)(smaller d17 d44)(smaller d17 d45)(smaller d17 d46)(smaller d17 d47)(smaller d17 d48)(smaller d17 d49)(smaller d17 d50)(smaller d17 d51)(smaller d17 d52)(smaller d17 d53)(smaller d17 d54)(smaller d17 d55)(smaller d17 d56)(smaller d17 d57)(smaller d17 d58)(smaller d17 d59)(smaller d17 d60)
    (smaller d18 d19)(smaller d18 d20)(smaller d18 d21)(smaller d18 d22)(smaller d18 d23)(smaller d18 d24)(smaller d18 d25)(smaller d18 d26)(smaller d18 d27)(smaller d18 d28)(smaller d18 d29)(smaller d18 d30)(smaller d18 d31)(smaller d18 d32)(smaller d18 d33)(smaller d18 d34)(smaller d18 d35)(smaller d18 d36)(smaller d18 d37)(smaller d18 d38)(smaller d18 d39)(smaller d18 d40)(smaller d18 d41)(smaller d18 d42)(smaller d18 d43)(smaller d18 d44)(smaller d18 d45)(smaller d18 d46)(smaller d18 d47)(smaller d18 d48)(smaller d18 d49)(smaller d18 d50)(smaller d18 d51)(smaller d18 d52)(smaller d18 d53)(smaller d18 d54)(smaller d18 d55)(smaller d18 d56)(smaller d18 d57)(smaller d18 d58)(smaller d18 d59)(smaller d18 d60)
    (smaller d19 d20)(smaller d19 d21)(smaller d19 d22)(smaller d19 d23)(smaller d19 d24)(smaller d19 d25)(smaller d19 d26)(smaller d19 d27)(smaller d19 d28)(smaller d19 d29)(smaller d19 d30)(smaller d19 d31)(smaller d19 d32)(smaller d19 d33)(smaller d19 d34)(smaller d19 d35)(smaller d19 d36)(smaller d19 d37)(smaller d19 d38)(smaller d19 d39)(smaller d19 d40)(smaller d19 d41)(smaller d19 d42)(smaller d19 d43)(smaller d19 d44)(smaller d19 d45)(smaller d19 d46)(smaller d19 d47)(smaller d19 d48)(smaller d19 d49)(smaller d19 d50)(smaller d19 d51)(smaller d19 d52)(smaller d19 d53)(smaller d19 d54)(smaller d19 d55)(smaller d19 d56)(smaller d19 d57)(smaller d19 d58)(smaller d19 d59)(smaller d19 d60)
    (smaller d20 d21)(smaller d20 d22)(smaller d20 d23)(smaller d20 d24)(smaller d20 d25)(smaller d20 d26)(smaller d20 d27)(smaller d20 d28)(smaller d20 d29)(smaller d20 d30)(smaller d20 d31)(smaller d20 d32)(smaller d20 d33)(smaller d20 d34)(smaller d20 d35)(smaller d20 d36)(smaller d20 d37)(smaller d20 d38)(smaller d20 d39)(smaller d20 d40)(smaller d20 d41)(smaller d20 d42)(smaller d20 d43)(smaller d20 d44)(smaller d20 d45)(smaller d20 d46)(smaller d20 d47)(smaller d20 d48)(smaller d20 d49)(smaller d20 d50)(smaller d20 d51)(smaller d20 d52)(smaller d20 d53)(smaller d20 d54)(smaller d20 d55)(smaller d20 d56)(smaller d20 d57)(smaller d20 d58)(smaller d20 d59)(smaller d20 d60)
    (smaller d21 d22)(smaller d21 d23)(smaller d21 d24)(smaller d21 d25)(smaller d21 d26)(smaller d21 d27)(smaller d21 d28)(smaller d21 d29)(smaller d21 d30)(smaller d21 d31)(smaller d21 d32)(smaller d21 d33)(smaller d21 d34)(smaller d21 d35)(smaller d21 d36)(smaller d21 d37)(smaller d21 d38)(smaller d21 d39)(smaller d21 d40)(smaller d21 d41)(smaller d21 d42)(smaller d21 d43)(smaller d21 d44)(smaller d21 d45)(smaller d21 d46)(smaller d21 d47)(smaller d21 d48)(smaller d21 d49)(smaller d21 d50)(smaller d21 d51)(smaller d21 d52)(smaller d21 d53)(smaller d21 d54)(smaller d21 d55)(smaller d21 d56)(smaller d21 d57)(smaller d21 d58)(smaller d21 d59)(smaller d21 d60)
    (smaller d22 d23)(smaller d22 d24)(smaller d22 d25)(smaller d22 d26)(smaller d22 d27)(smaller d22 d28)(smaller d22 d29)(smaller d22 d30)(smaller d22 d31)(smaller d22 d32)(smaller d22 d33)(smaller d22 d34)(smaller d22 d35)(smaller d22 d36)(smaller d22 d37)(smaller d22 d38)(smaller d22 d39)(smaller d22 d40)(smaller d22 d41)(smaller d22 d42)(smaller d22 d43)(smaller d22 d44)(smaller d22 d45)(smaller d22 d46)(smaller d22 d47)(smaller d22 d48)(smaller d22 d49)(smaller d22 d50)(smaller d22 d51)(smaller d22 d52)(smaller d22 d53)(smaller d22 d54)(smaller d22 d55)(smaller d22 d56)(smaller d22 d57)(smaller d22 d58)(smaller d22 d59)(smaller d22 d60)
    (smaller d23 d24)(smaller d23 d25)(smaller d23 d26)(smaller d23 d27)(smaller d23 d28)(smaller d23 d29)(smaller d23 d30)(smaller d23 d31)(smaller d23 d32)(smaller d23 d33)(smaller d23 d34)(smaller d23 d35)(smaller d23 d36)(smaller d23 d37)(smaller d23 d38)(smaller d23 d39)(smaller d23 d40)(smaller d23 d41)(smaller d23 d42)(smaller d23 d43)(smaller d23 d44)(smaller d23 d45)(smaller d23 d46)(smaller d23 d47)(smaller d23 d48)(smaller d23 d49)(smaller d23 d50)(smaller d23 d51)(smaller d23 d52)(smaller d23 d53)(smaller d23 d54)(smaller d23 d55)(smaller d23 d56)(smaller d23 d57)(smaller d23 d58)(smaller d23 d59)(smaller d23 d60)
    (smaller d24 d25)(smaller d24 d26)(smaller d24 d27)(smaller d24 d28)(smaller d24 d29)(smaller d24 d30)(smaller d24 d31)(smaller d24 d32)(smaller d24 d33)(smaller d24 d34)(smaller d24 d35)(smaller d24 d36)(smaller d24 d37)(smaller d24 d38)(smaller d24 d39)(smaller d24 d40)(smaller d24 d41)(smaller d24 d42)(smaller d24 d43)(smaller d24 d44)(smaller d24 d45)(smaller d24 d46)(smaller d24 d47)(smaller d24 d48)(smaller d24 d49)(smaller d24 d50)(smaller d24 d51)(smaller d24 d52)(smaller d24 d53)(smaller d24 d54)(smaller d24 d55)(smaller d24 d56)(smaller d24 d57)(smaller d24 d58)(smaller d24 d59)(smaller d24 d60)
    (smaller d25 d26)(smaller d25 d27)(smaller d25 d28)(smaller d25 d29)(smaller d25 d30)(smaller d25 d31)(smaller d25 d32)(smaller d25 d33)(smaller d25 d34)(smaller d25 d35)(smaller d25 d36)(smaller d25 d37)(smaller d25 d38)(smaller d25 d39)(smaller d25 d40)(smaller d25 d41)(smaller d25 d42)(smaller d25 d43)(smaller d25 d44)(smaller d25 d45)(smaller d25 d46)(smaller d25 d47)(smaller d25 d48)(smaller d25 d49)(smaller d25 d50)(smaller d25 d51)(smaller d25 d52)(smaller d25 d53)(smaller d25 d54)(smaller d25 d55)(smaller d25 d56)(smaller d25 d57)(smaller d25 d58)(smaller d25 d59)(smaller d25 d60)
    (smaller d26 d27)(smaller d26 d28)(smaller d26 d29)(smaller d26 d30)(smaller d26 d31)(smaller d26 d32)(smaller d26 d33)(smaller d26 d34)(smaller d26 d35)(smaller d26 d36)(smaller d26 d37)(smaller d26 d38)(smaller d26 d39)(smaller d26 d40)(smaller d26 d41)(smaller d26 d42)(smaller d26 d43)(smaller d26 d44)(smaller d26 d45)(smaller d26 d46)(smaller d26 d47)(smaller d26 d48)(smaller d26 d49)(smaller d26 d50)(smaller d26 d51)(smaller d26 d52)(smaller d26 d53)(smaller d26 d54)(smaller d26 d55)(smaller d26 d56)(smaller d26 d57)(smaller d26 d58)(smaller d26 d59)(smaller d26 d60)
    (smaller d27 d28)(smaller d27 d29)(smaller d27 d30)(smaller d27 d31)(smaller d27 d32)(smaller d27 d33)(smaller d27 d34)(smaller d27 d35)(smaller d27 d36)(smaller d27 d37)(smaller d27 d38)(smaller d27 d39)(smaller d27 d40)(smaller d27 d41)(smaller d27 d42)(smaller d27 d43)(smaller d27 d44)(smaller d27 d45)(smaller d27 d46)(smaller d27 d47)(smaller d27 d48)(smaller d27 d49)(smaller d27 d50)(smaller d27 d51)(smaller d27 d52)(smaller d27 d53)(smaller d27 d54)(smaller d27 d55)(smaller d27 d56)(smaller d27 d57)(smaller d27 d58)(smaller d27 d59)(smaller d27 d60)
    (smaller d28 d29)(smaller d28 d30)(smaller d28 d31)(smaller d28 d32)(smaller d28 d33)(smaller d28 d34)(smaller d28 d35)(smaller d28 d36)(smaller d28 d37)(smaller d28 d38)(smaller d28 d39)(smaller d28 d40)(smaller d28 d41)(smaller d28 d42)(smaller d28 d43)(smaller d28 d44)(smaller d28 d45)(smaller d28 d46)(smaller d28 d47)(smaller d28 d48)(smaller d28 d49)(smaller d28 d50)(smaller d28 d51)(smaller d28 d52)(smaller d28 d53)(smaller d28 d54)(smaller d28 d55)(smaller d28 d56)(smaller d28 d57)(smaller d28 d58)(smaller d28 d59)(smaller d28 d60)
    (smaller d29 d30)(smaller d29 d31)(smaller d29 d32)(smaller d29 d33)(smaller d29 d34)(smaller d29 d35)(smaller d29 d36)(smaller d29 d37)(smaller d29 d38)(smaller d29 d39)(smaller d29 d40)(smaller d29 d41)(smaller d29 d42)(smaller d29 d43)(smaller d29 d44)(smaller d29 d45)(smaller d29 d46)(smaller d29 d47)(smaller d29 d48)(smaller d29 d49)(smaller d29 d50)(smaller d29 d51)(smaller d29 d52)(smaller d29 d53)(smaller d29 d54)(smaller d29 d55)(smaller d29 d56)(smaller d29 d57)(smaller d29 d58)(smaller d29 d59)(smaller d29 d60)
    (smaller d30 d31)(smaller d30 d32)(smaller d30 d33)(smaller d30 d34)(smaller d30 d35)(smaller d30 d36)(smaller d30 d37)(smaller d30 d38)(smaller d30 d39)(smaller d30 d40)(smaller d30 d41)(smaller d30 d42)(smaller d30 d43)(smaller d30 d44)(smaller d30 d45)(smaller d30 d46)(smaller d30 d47)(smaller d30 d48)(smaller d30 d49)(smaller d30 d50)(smaller d30 d51)(smaller d30 d52)(smaller d30 d53)(smaller d30 d54)(smaller d30 d55)(smaller d30 d56)(smaller d30 d57)(smaller d30 d58)(smaller d30 d59)(smaller d30 d60)
    (smaller d31 d32)(smaller d31 d33)(smaller d31 d34)(smaller d31 d35)(smaller d31 d36)(smaller d31 d37)(smaller d31 d38)(smaller d31 d39)(smaller d31 d40)(smaller d31 d41)(smaller d31 d42)(smaller d31 d43)(smaller d31 d44)(smaller d31 d45)(smaller d31 d46)(smaller d31 d47)(smaller d31 d48)(smaller d31 d49)(smaller d31 d50)(smaller d31 d51)(smaller d31 d52)(smaller d31 d53)(smaller d31 d54)(smaller d31 d55)(smaller d31 d56)(smaller d31 d57)(smaller d31 d58)(smaller d31 d59)(smaller d31 d60)
    (smaller d32 d33)(smaller d32 d34)(smaller d32 d35)(smaller d32 d36)(smaller d32 d37)(smaller d32 d38)(smaller d32 d39)(smaller d32 d40)(smaller d32 d41)(smaller d32 d42)(smaller d32 d43)(smaller d32 d44)(smaller d32 d45)(smaller d32 d46)(smaller d32 d47)(smaller d32 d48)(smaller d32 d49)(smaller d32 d50)(smaller d32 d51)(smaller d32 d52)(smaller d32 d53)(smaller d32 d54)(smaller d32 d55)(smaller d32 d56)(smaller d32 d57)(smaller d32 d58)(smaller d32 d59)(smaller d32 d60)
    (smaller d33 d34)(smaller d33 d35)(smaller d33 d36)(smaller d33 d37)(smaller d33 d38)(smaller d33 d39)(smaller d33 d40)(smaller d33 d41)(smaller d33 d42)(smaller d33 d43)(smaller d33 d44)(smaller d33 d45)(smaller d33 d46)(smaller d33 d47)(smaller d33 d48)(smaller d33 d49)(smaller d33 d50)(smaller d33 d51)(smaller d33 d52)(smaller d33 d53)(smaller d33 d54)(smaller d33 d55)(smaller d33 d56)(smaller d33 d57)(smaller d33 d58)(smaller d33 d59)(smaller d33 d60)
    (smaller d34 d35)(smaller d34 d36)(smaller d34 d37)(smaller d34 d38)(smaller d34 d39)(smaller d34 d40)(smaller d34 d41)(smaller d34 d42)(smaller d34 d43)(smaller d34 d44)(smaller d34 d45)(smaller d34 d46)(smaller d34 d47)(smaller d34 d48)(smaller d34 d49)(smaller d34 d50)(smaller d34 d51)(smaller d34 d52)(smaller d34 d53)(smaller d34 d54)(smaller d34 d55)(smaller d34 d56)(smaller d34 d57)(smaller d34 d58)(smaller d34 d59)(smaller d34 d60)
    (smaller d35 d36)(smaller d35 d37)(smaller d35 d38)(smaller d35 d39)(smaller d35 d40)(smaller d35 d41)(smaller d35 d42)(smaller d35 d43)(smaller d35 d44)(smaller d35 d45)(smaller d35 d46)(smaller d35 d47)(smaller d35 d48)(smaller d35 d49)(smaller d35 d50)(smaller d35 d51)(smaller d35 d52)(smaller d35 d53)(smaller d35 d54)(smaller d35 d55)(smaller d35 d56)(smaller d35 d57)(smaller d35 d58)(smaller d35 d59)(smaller d35 d60)
    (smaller d36 d37)(smaller d36 d38)(smaller d36 d39)(smaller d36 d40)(smaller d36 d41)(smaller d36 d42)(smaller d36 d43)(smaller d36 d44)(smaller d36 d45)(smaller d36 d46)(smaller d36 d47)(smaller d36 d48)(smaller d36 d49)(smaller d36 d50)(smaller d36 d51)(smaller d36 d52)(smaller d36 d53)(smaller d36 d54)(smaller d36 d55)(smaller d36 d56)(smaller d36 d57)(smaller d36 d58)(smaller d36 d59)(smaller d36 d60)
    (smaller d37 d38)(smaller d37 d39)(smaller d37 d40)(smaller d37 d41)(smaller d37 d42)(smaller d37 d43)(smaller d37 d44)(smaller d37 d45)(smaller d37 d46)(smaller d37 d47)(smaller d37 d48)(smaller d37 d49)(smaller d37 d50)(smaller d37 d51)(smaller d37 d52)(smaller d37 d53)(smaller d37 d54)(smaller d37 d55)(smaller d37 d56)(smaller d37 d57)(smaller d37 d58)(smaller d37 d59)(smaller d37 d60)
    (smaller d38 d39)(smaller d38 d40)(smaller d38 d41)(smaller d38 d42)(smaller d38 d43)(smaller d38 d44)(smaller d38 d45)(smaller d38 d46)(smaller d38 d47)(smaller d38 d48)(smaller d38 d49)(smaller d38 d50)(smaller d38 d51)(smaller d38 d52)(smaller d38 d53)(smaller d38 d54)(smaller d38 d55)(smaller d38 d56)(smaller d38 d57)(smaller d38 d58)(smaller d38 d59)(smaller d38 d60)
    (smaller d39 d40)(smaller d39 d41)(smaller d39 d42)(smaller d39 d43)(smaller d39 d44)(smaller d39 d45)(smaller d39 d46)(smaller d39 d47)(smaller d39 d48)(smaller d39 d49)(smaller d39 d50)(smaller d39 d51)(smaller d39 d52)(smaller d39 d53)(smaller d39 d54)(smaller d39 d55)(smaller d39 d56)(smaller d39 d57)(smaller d39 d58)(smaller d39 d59)(smaller d39 d60)
    (smaller d40 d41)(smaller d40 d42)(smaller d40 d43)(smaller d40 d44)(smaller d40 d45)(smaller d40 d46)(smaller d40 d47)(smaller d40 d48)(smaller d40 d49)(smaller d40 d50)(smaller d40 d51)(smaller d40 d52)(smaller d40 d53)(smaller d40 d54)(smaller d40 d55)(smaller d40 d56)(smaller d40 d57)(smaller d40 d58)(smaller d40 d59)(smaller d40 d60)
    (smaller d41 d42)(smaller d41 d43)(smaller d41 d44)(smaller d41 d45)(smaller d41 d46)(smaller d41 d47)(smaller d41 d48)(smaller d41 d49)(smaller d41 d50)(smaller d41 d51)(smaller d41 d52)(smaller d41 d53)(smaller d41 d54)(smaller d41 d55)(smaller d41 d56)(smaller d41 d57)(smaller d41 d58)(smaller d41 d59)(smaller d41 d60)
    (smaller d42 d43)(smaller d42 d44)(smaller d42 d45)(smaller d42 d46)(smaller d42 d47)(smaller d42 d48)(smaller d42 d49)(smaller d42 d50)(smaller d42 d51)(smaller d42 d52)(smaller d42 d53)(smaller d42 d54)(smaller d42 d55)(smaller d42 d56)(smaller d42 d57)(smaller d42 d58)(smaller d42 d59)(smaller d42 d60)
    (smaller d43 d44)(smaller d43 d45)(smaller d43 d46)(smaller d43 d47)(smaller d43 d48)(smaller d43 d49)(smaller d43 d50)(smaller d43 d51)(smaller d43 d52)(smaller d43 d53)(smaller d43 d54)(smaller d43 d55)(smaller d43 d56)(smaller d43 d57)(smaller d43 d58)(smaller d43 d59)(smaller d43 d60)
    (smaller d44 d45)(smaller d44 d46)(smaller d44 d47)(smaller d44 d48)(smaller d44 d49)(smaller d44 d50)(smaller d44 d51)(smaller d44 d52)(smaller d44 d53)(smaller d44 d54)(smaller d44 d55)(smaller d44 d56)(smaller d44 d57)(smaller d44 d58)(smaller d44 d59)(smaller d44 d60)
    (smaller d45 d46)(smaller d45 d47)(smaller d45 d48)(smaller d45 d49)(smaller d45 d50)(smaller d45 d51)(smaller d45 d52)(smaller d45 d53)(smaller d45 d54)(smaller d45 d55)(smaller d45 d56)(smaller d45 d57)(smaller d45 d58)(smaller d45 d59)(smaller d45 d60)
    (smaller d46 d47)(smaller d46 d48)(smaller d46 d49)(smaller d46 d50)(smaller d46 d51)(smaller d46 d52)(smaller d46 d53)(smaller d46 d54)(smaller d46 d55)(smaller d46 d56)(smaller d46 d57)(smaller d46 d58)(smaller d46 d59)(smaller d46 d60)
    (smaller d47 d48)(smaller d47 d49)(smaller d47 d50)(smaller d47 d51)(smaller d47 d52)(smaller d47 d53)(smaller d47 d54)(smaller d47 d55)(smaller d47 d56)(smaller d47 d57)(smaller d47 d58)(smaller d47 d59)(smaller d47 d60)
    (smaller d48 d49)(smaller d48 d50)(smaller d48 d51)(smaller d48 d52)(smaller d48 d53)(smaller d48 d54)(smaller d48 d55)(smaller d48 d56)(smaller d48 d57)(smaller d48 d58)(smaller d48 d59)(smaller d48 d60)
    (smaller d49 d50)(smaller d49 d51)(smaller d49 d52)(smaller d49 d53)(smaller d49 d54)(smaller d49 d55)(smaller d49 d56)(smaller d49 d57)(smaller d49 d58)(smaller d49 d59)(smaller d49 d60)
    (smaller d50 d51)(smaller d50 d52)(smaller d50 d53)(smaller d50 d54)(smaller d50 d55)(smaller d50 d56)(smaller d50 d57)(smaller d50 d58)(smaller d50 d59)(smaller d50 d60)
    (smaller d51 d52)(smaller d51 d53)(smaller d51 d54)(smaller d51 d55)(smaller d51 d56)(smaller d51 d57)(smaller d51 d58)(smaller d51 d59)(smaller d51 d60)
    (smaller d52 d53)(smaller d52 d54)(smaller d52 d55)(smaller d52 d56)(smaller d52 d57)(smaller d52 d58)(smaller d52 d59)(smaller d52 d60)
    (smaller d53 d54)(smaller d53 d55)(smaller d53 d56)(smaller d53 d57)(smaller d53 d58)(smaller d53 d59)(smaller d53 d60)
    (smaller d54 d55)(smaller d54 d56)(smaller d54 d57)(smaller d54 d58)(smaller d54 d59)(smaller d54 d60)
    (smaller d55 d56)(smaller d55 d57)(smaller d55 d58)(smaller d55 d59)(smaller d55 d60)
    (smaller d56 d57)(smaller d56 d58)(smaller d56 d59)(smaller d56 d60)
    (smaller d57 d58)(smaller d57 d59)(smaller d57 d60)
    (smaller d58 d59)(smaller d58 d60)
    (smaller d59 d60)

    (clear peg1)(clear peg2)(clear d1)

    (on d1 d2)(on d2 d3)(on d3 d4)(on d4 d5)(on d5 d6)(on d6 d7)(on d7 d8)(on d8 d9)(on d9 d10)(on d10 d11)(on d11 d12)(on d12 d13)(on d13 d14)(on d14 d15)(on d15 d16)(on d16 d17)(on d17 d18)(on d18 d19)(on d19 d20)(on d20 d21)(on d21 d22)(on d22 d23)(on d23 d24)(on d24 d25)(on d25 d26)(on d26 d27)(on d27 d28)(on d28 d29)(on d29 d30)(on d30 d31)(on d31 d32)(on d32 d33)(on d33 d34)(on d34 d35)(on d35 d36)(on d36 d37)(on d37 d38)(on d38 d39)(on d39 d40)(on d40 d41)(on d41 d42)(on d42 d43)(on d43 d44)(on d44 d45)(on d45 d46)(on d46 d47)(on d47 d48)(on d48 d49)(on d49 d50)(on d50 d51)(on d51 d52)(on d52 d53)(on d53 d54)(on d54 d55)(on d55 d56)(on d56 d57)(on d57 d58)(on d58 d59)(on d59 d60)(on d60 peg3)
  )
  (:goal 
    (and (on d1 d2)(on d2 d3)(on d3 d4)(on d4 d5)(on d5 d6)(on d6 d7)(on d7 d8)(on d8 d9)(on d9 d10)(on d10 d11)(on d11 d12)(on d12 d13)(on d13 d14)(on d14 d15)(on d15 d16)(on d16 d17)(on d17 d18)(on d18 d19)(on d19 d20)(on d20 d21)(on d21 d22)(on d22 d23)(on d23 d24)(on d24 d25)(on d25 d26)(on d26 d27)(on d27 d28)(on d28 d29)(on d29 d30)(on d30 d31)(on d31 d32)(on d32 d33)(on d33 d34)(on d34 d35)(on d35 d36)(on d36 d37)(on d37 d38)(on d38 d39)(on d39 d40)(on d40 d41)(on d41 d42)(on d42 d43)(on d43 d44)(on d44 d45)(on d45 d46)(on d46 d47)(on d47 d48)(on d48 d49)(on d49 d50)(on d50 d51)(on d51 d52)(on d52 d53)(on d53 d54)(on d54 d55)(on d55 d56)(on d56 d57)(on d57 d58)(on d58 d59)(on d59 d60)(on d60 peg1) )
  )
)
