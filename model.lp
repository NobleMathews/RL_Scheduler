\* Source Pyomo model name=unknown *\

max 
Obj:
-1 Cost
-1 StorageCost
+1 Value

s.t.

c_e_cons(1)_:
+1 S(HotA_10)
+1 S(ImpureE_10)
+1 S(IntAB_10)
+1 S(IntBC_10)
-10 S(Product_1_10)
-10 S(Product_2_10)
+1 Value
= 0

c_e_cons(2)_:
+1 Cost
= 0

c_e_cons(3)_:
-1 S(HotA_0)
-1 S(HotA_1)
-1 S(HotA_10)
-1 S(HotA_2)
-1 S(HotA_3)
-1 S(HotA_4)
-1 S(HotA_5)
-1 S(HotA_6)
-1 S(HotA_7)
-1 S(HotA_8)
-1 S(HotA_9)
+1 StorageCost
= 0

c_u_cons(4)_:
+1 W(Reaction_1_Reactor_2_0)
+1 W(Reaction_2_Reactor_2_0)
+1 W(Reaction_3_Reactor_2_0)
<= 1

c_l_cons(5)_:
+1 B(Reaction_3_Reactor_2_0)
>= 0

c_u_cons(6)_:
+1 B(Reaction_3_Reactor_2_0)
-50 W(Reaction_3_Reactor_2_0)
<= 0

c_l_cons(7)_:
+1 B(Reaction_2_Reactor_2_0)
>= 0

c_u_cons(8)_:
+1 B(Reaction_2_Reactor_2_0)
-50 W(Reaction_2_Reactor_2_0)
<= 0

c_l_cons(9)_:
+1 B(Reaction_1_Reactor_2_0)
>= 0

c_u_cons(10)_:
+1 B(Reaction_1_Reactor_2_0)
-50 W(Reaction_1_Reactor_2_0)
<= 0

c_e_cons(11)_:
-1 B(Reaction_1_Reactor_2_0)
-1 B(Reaction_2_Reactor_2_0)
-1 B(Reaction_3_Reactor_2_0)
+1 Q(Reactor_2_0)
= 0

c_e_cons(12)_:
+1 Q(Reactor_2_10)
= 0

c_u_cons(13)_:
+1 W(Reaction_1_Reactor_2_0)
+1 W(Reaction_1_Reactor_2_1)
+1 W(Reaction_2_Reactor_2_0)
+1 W(Reaction_2_Reactor_2_1)
+1 W(Reaction_3_Reactor_2_1)
<= 1

c_l_cons(14)_:
+1 B(Reaction_3_Reactor_2_1)
>= 0

c_u_cons(15)_:
+1 B(Reaction_3_Reactor_2_1)
-50 W(Reaction_3_Reactor_2_1)
<= 0

c_l_cons(16)_:
+1 B(Reaction_2_Reactor_2_1)
>= 0

c_u_cons(17)_:
+1 B(Reaction_2_Reactor_2_1)
-50 W(Reaction_2_Reactor_2_1)
<= 0

c_l_cons(18)_:
+1 B(Reaction_1_Reactor_2_1)
>= 0

c_u_cons(19)_:
+1 B(Reaction_1_Reactor_2_1)
-50 W(Reaction_1_Reactor_2_1)
<= 0

c_e_cons(20)_:
-1 B(Reaction_1_Reactor_2_1)
-1 B(Reaction_2_Reactor_2_1)
+1 B(Reaction_3_Reactor_2_0)
-1 B(Reaction_3_Reactor_2_1)
-1 Q(Reactor_2_0)
+1 Q(Reactor_2_1)
= 0

c_e_cons(21)_:
+1 Q(Reactor_2_10)
= 0

c_u_cons(22)_:
+1 W(Reaction_1_Reactor_2_1)
+1 W(Reaction_1_Reactor_2_2)
+1 W(Reaction_2_Reactor_2_1)
+1 W(Reaction_2_Reactor_2_2)
+1 W(Reaction_3_Reactor_2_2)
<= 1

c_l_cons(23)_:
+1 B(Reaction_3_Reactor_2_2)
>= 0

c_u_cons(24)_:
+1 B(Reaction_3_Reactor_2_2)
-50 W(Reaction_3_Reactor_2_2)
<= 0

c_l_cons(25)_:
+1 B(Reaction_2_Reactor_2_2)
>= 0

c_u_cons(26)_:
+1 B(Reaction_2_Reactor_2_2)
-50 W(Reaction_2_Reactor_2_2)
<= 0

c_l_cons(27)_:
+1 B(Reaction_1_Reactor_2_2)
>= 0

c_u_cons(28)_:
+1 B(Reaction_1_Reactor_2_2)
-50 W(Reaction_1_Reactor_2_2)
<= 0

c_e_cons(29)_:
+1 B(Reaction_1_Reactor_2_0)
-1 B(Reaction_1_Reactor_2_2)
+1 B(Reaction_2_Reactor_2_0)
-1 B(Reaction_2_Reactor_2_2)
+1 B(Reaction_3_Reactor_2_1)
-1 B(Reaction_3_Reactor_2_2)
-1 Q(Reactor_2_1)
+1 Q(Reactor_2_2)
= 0

c_e_cons(30)_:
+1 Q(Reactor_2_10)
= 0

c_u_cons(31)_:
+1 W(Reaction_1_Reactor_2_2)
+1 W(Reaction_1_Reactor_2_3)
+1 W(Reaction_2_Reactor_2_2)
+1 W(Reaction_2_Reactor_2_3)
+1 W(Reaction_3_Reactor_2_3)
<= 1

c_l_cons(32)_:
+1 B(Reaction_3_Reactor_2_3)
>= 0

c_u_cons(33)_:
+1 B(Reaction_3_Reactor_2_3)
-50 W(Reaction_3_Reactor_2_3)
<= 0

c_l_cons(34)_:
+1 B(Reaction_2_Reactor_2_3)
>= 0

c_u_cons(35)_:
+1 B(Reaction_2_Reactor_2_3)
-50 W(Reaction_2_Reactor_2_3)
<= 0

c_l_cons(36)_:
+1 B(Reaction_1_Reactor_2_3)
>= 0

c_u_cons(37)_:
+1 B(Reaction_1_Reactor_2_3)
-50 W(Reaction_1_Reactor_2_3)
<= 0

c_e_cons(38)_:
+1 B(Reaction_1_Reactor_2_1)
-1 B(Reaction_1_Reactor_2_3)
+1 B(Reaction_2_Reactor_2_1)
-1 B(Reaction_2_Reactor_2_3)
+1 B(Reaction_3_Reactor_2_2)
-1 B(Reaction_3_Reactor_2_3)
-1 Q(Reactor_2_2)
+1 Q(Reactor_2_3)
= 0

c_e_cons(39)_:
+1 Q(Reactor_2_10)
= 0

c_u_cons(40)_:
+1 W(Reaction_1_Reactor_2_3)
+1 W(Reaction_1_Reactor_2_4)
+1 W(Reaction_2_Reactor_2_3)
+1 W(Reaction_2_Reactor_2_4)
+1 W(Reaction_3_Reactor_2_4)
<= 1

c_l_cons(41)_:
+1 B(Reaction_3_Reactor_2_4)
>= 0

c_u_cons(42)_:
+1 B(Reaction_3_Reactor_2_4)
-50 W(Reaction_3_Reactor_2_4)
<= 0

c_l_cons(43)_:
+1 B(Reaction_2_Reactor_2_4)
>= 0

c_u_cons(44)_:
+1 B(Reaction_2_Reactor_2_4)
-50 W(Reaction_2_Reactor_2_4)
<= 0

c_l_cons(45)_:
+1 B(Reaction_1_Reactor_2_4)
>= 0

c_u_cons(46)_:
+1 B(Reaction_1_Reactor_2_4)
-50 W(Reaction_1_Reactor_2_4)
<= 0

c_e_cons(47)_:
+1 B(Reaction_1_Reactor_2_2)
-1 B(Reaction_1_Reactor_2_4)
+1 B(Reaction_2_Reactor_2_2)
-1 B(Reaction_2_Reactor_2_4)
+1 B(Reaction_3_Reactor_2_3)
-1 B(Reaction_3_Reactor_2_4)
-1 Q(Reactor_2_3)
+1 Q(Reactor_2_4)
= 0

c_e_cons(48)_:
+1 Q(Reactor_2_10)
= 0

c_u_cons(49)_:
+1 W(Reaction_1_Reactor_2_4)
+1 W(Reaction_1_Reactor_2_5)
+1 W(Reaction_2_Reactor_2_4)
+1 W(Reaction_2_Reactor_2_5)
+1 W(Reaction_3_Reactor_2_5)
<= 1

c_l_cons(50)_:
+1 B(Reaction_3_Reactor_2_5)
>= 0

c_u_cons(51)_:
+1 B(Reaction_3_Reactor_2_5)
-50 W(Reaction_3_Reactor_2_5)
<= 0

c_l_cons(52)_:
+1 B(Reaction_2_Reactor_2_5)
>= 0

c_u_cons(53)_:
+1 B(Reaction_2_Reactor_2_5)
-50 W(Reaction_2_Reactor_2_5)
<= 0

c_l_cons(54)_:
+1 B(Reaction_1_Reactor_2_5)
>= 0

c_u_cons(55)_:
+1 B(Reaction_1_Reactor_2_5)
-50 W(Reaction_1_Reactor_2_5)
<= 0

c_e_cons(56)_:
+1 B(Reaction_1_Reactor_2_3)
-1 B(Reaction_1_Reactor_2_5)
+1 B(Reaction_2_Reactor_2_3)
-1 B(Reaction_2_Reactor_2_5)
+1 B(Reaction_3_Reactor_2_4)
-1 B(Reaction_3_Reactor_2_5)
-1 Q(Reactor_2_4)
+1 Q(Reactor_2_5)
= 0

c_e_cons(57)_:
+1 Q(Reactor_2_10)
= 0

c_u_cons(58)_:
+1 W(Reaction_1_Reactor_2_5)
+1 W(Reaction_1_Reactor_2_6)
+1 W(Reaction_2_Reactor_2_5)
+1 W(Reaction_2_Reactor_2_6)
+1 W(Reaction_3_Reactor_2_6)
<= 1

c_l_cons(59)_:
+1 B(Reaction_3_Reactor_2_6)
>= 0

c_u_cons(60)_:
+1 B(Reaction_3_Reactor_2_6)
-50 W(Reaction_3_Reactor_2_6)
<= 0

c_l_cons(61)_:
+1 B(Reaction_2_Reactor_2_6)
>= 0

c_u_cons(62)_:
+1 B(Reaction_2_Reactor_2_6)
-50 W(Reaction_2_Reactor_2_6)
<= 0

c_l_cons(63)_:
+1 B(Reaction_1_Reactor_2_6)
>= 0

c_u_cons(64)_:
+1 B(Reaction_1_Reactor_2_6)
-50 W(Reaction_1_Reactor_2_6)
<= 0

c_e_cons(65)_:
+1 B(Reaction_1_Reactor_2_4)
-1 B(Reaction_1_Reactor_2_6)
+1 B(Reaction_2_Reactor_2_4)
-1 B(Reaction_2_Reactor_2_6)
+1 B(Reaction_3_Reactor_2_5)
-1 B(Reaction_3_Reactor_2_6)
-1 Q(Reactor_2_5)
+1 Q(Reactor_2_6)
= 0

c_e_cons(66)_:
+1 Q(Reactor_2_10)
= 0

c_u_cons(67)_:
+1 W(Reaction_1_Reactor_2_6)
+1 W(Reaction_1_Reactor_2_7)
+1 W(Reaction_2_Reactor_2_6)
+1 W(Reaction_2_Reactor_2_7)
+1 W(Reaction_3_Reactor_2_7)
<= 1

c_l_cons(68)_:
+1 B(Reaction_3_Reactor_2_7)
>= 0

c_u_cons(69)_:
+1 B(Reaction_3_Reactor_2_7)
-50 W(Reaction_3_Reactor_2_7)
<= 0

c_l_cons(70)_:
+1 B(Reaction_2_Reactor_2_7)
>= 0

c_u_cons(71)_:
+1 B(Reaction_2_Reactor_2_7)
-50 W(Reaction_2_Reactor_2_7)
<= 0

c_l_cons(72)_:
+1 B(Reaction_1_Reactor_2_7)
>= 0

c_u_cons(73)_:
+1 B(Reaction_1_Reactor_2_7)
-50 W(Reaction_1_Reactor_2_7)
<= 0

c_e_cons(74)_:
+1 B(Reaction_1_Reactor_2_5)
-1 B(Reaction_1_Reactor_2_7)
+1 B(Reaction_2_Reactor_2_5)
-1 B(Reaction_2_Reactor_2_7)
+1 B(Reaction_3_Reactor_2_6)
-1 B(Reaction_3_Reactor_2_7)
-1 Q(Reactor_2_6)
+1 Q(Reactor_2_7)
= 0

c_e_cons(75)_:
+1 Q(Reactor_2_10)
= 0

c_u_cons(76)_:
+1 W(Reaction_1_Reactor_2_7)
+1 W(Reaction_1_Reactor_2_8)
+1 W(Reaction_2_Reactor_2_7)
+1 W(Reaction_2_Reactor_2_8)
+1 W(Reaction_3_Reactor_2_8)
<= 1

c_l_cons(77)_:
+1 B(Reaction_3_Reactor_2_8)
>= 0

c_u_cons(78)_:
+1 B(Reaction_3_Reactor_2_8)
-50 W(Reaction_3_Reactor_2_8)
<= 0

c_l_cons(79)_:
+1 B(Reaction_2_Reactor_2_8)
>= 0

c_u_cons(80)_:
+1 B(Reaction_2_Reactor_2_8)
-50 W(Reaction_2_Reactor_2_8)
<= 0

c_l_cons(81)_:
+1 B(Reaction_1_Reactor_2_8)
>= 0

c_u_cons(82)_:
+1 B(Reaction_1_Reactor_2_8)
-50 W(Reaction_1_Reactor_2_8)
<= 0

c_e_cons(83)_:
+1 B(Reaction_1_Reactor_2_6)
-1 B(Reaction_1_Reactor_2_8)
+1 B(Reaction_2_Reactor_2_6)
-1 B(Reaction_2_Reactor_2_8)
+1 B(Reaction_3_Reactor_2_7)
-1 B(Reaction_3_Reactor_2_8)
-1 Q(Reactor_2_7)
+1 Q(Reactor_2_8)
= 0

c_e_cons(84)_:
+1 Q(Reactor_2_10)
= 0

c_u_cons(85)_:
+1 W(Reaction_1_Reactor_2_8)
+1 W(Reaction_1_Reactor_2_9)
+1 W(Reaction_2_Reactor_2_8)
+1 W(Reaction_2_Reactor_2_9)
+1 W(Reaction_3_Reactor_2_9)
<= 1

c_l_cons(86)_:
+1 B(Reaction_3_Reactor_2_9)
>= 0

c_u_cons(87)_:
+1 B(Reaction_3_Reactor_2_9)
-50 W(Reaction_3_Reactor_2_9)
<= 0

c_l_cons(88)_:
+1 B(Reaction_2_Reactor_2_9)
>= 0

c_u_cons(89)_:
+1 B(Reaction_2_Reactor_2_9)
-50 W(Reaction_2_Reactor_2_9)
<= 0

c_l_cons(90)_:
+1 B(Reaction_1_Reactor_2_9)
>= 0

c_u_cons(91)_:
+1 B(Reaction_1_Reactor_2_9)
-50 W(Reaction_1_Reactor_2_9)
<= 0

c_e_cons(92)_:
+1 B(Reaction_1_Reactor_2_7)
-1 B(Reaction_1_Reactor_2_9)
+1 B(Reaction_2_Reactor_2_7)
-1 B(Reaction_2_Reactor_2_9)
+1 B(Reaction_3_Reactor_2_8)
-1 B(Reaction_3_Reactor_2_9)
-1 Q(Reactor_2_8)
+1 Q(Reactor_2_9)
= 0

c_e_cons(93)_:
+1 Q(Reactor_2_10)
= 0

c_u_cons(94)_:
+1 W(Reaction_1_Reactor_2_10)
+1 W(Reaction_1_Reactor_2_9)
+1 W(Reaction_2_Reactor_2_10)
+1 W(Reaction_2_Reactor_2_9)
+1 W(Reaction_3_Reactor_2_10)
<= 1

c_l_cons(95)_:
+1 B(Reaction_3_Reactor_2_10)
>= 0

c_u_cons(96)_:
+1 B(Reaction_3_Reactor_2_10)
-50 W(Reaction_3_Reactor_2_10)
<= 0

c_l_cons(97)_:
+1 B(Reaction_2_Reactor_2_10)
>= 0

c_u_cons(98)_:
+1 B(Reaction_2_Reactor_2_10)
-50 W(Reaction_2_Reactor_2_10)
<= 0

c_l_cons(99)_:
+1 B(Reaction_1_Reactor_2_10)
>= 0

c_u_cons(100)_:
+1 B(Reaction_1_Reactor_2_10)
-50 W(Reaction_1_Reactor_2_10)
<= 0

c_e_cons(101)_:
-1 B(Reaction_1_Reactor_2_10)
+1 B(Reaction_1_Reactor_2_8)
-1 B(Reaction_2_Reactor_2_10)
+1 B(Reaction_2_Reactor_2_8)
-1 B(Reaction_3_Reactor_2_10)
+1 B(Reaction_3_Reactor_2_9)
+1 Q(Reactor_2_10)
-1 Q(Reactor_2_9)
= 0

c_e_cons(102)_:
+1 Q(Reactor_2_10)
= 0

c_u_cons(103)_:
+1 W(Reaction_1_Reactor_1_0)
+1 W(Reaction_2_Reactor_1_0)
+1 W(Reaction_3_Reactor_1_0)
<= 1

c_l_cons(104)_:
+1 B(Reaction_3_Reactor_1_0)
>= 0

c_u_cons(105)_:
+1 B(Reaction_3_Reactor_1_0)
-80 W(Reaction_3_Reactor_1_0)
<= 0

c_l_cons(106)_:
+1 B(Reaction_2_Reactor_1_0)
>= 0

c_u_cons(107)_:
+1 B(Reaction_2_Reactor_1_0)
-80 W(Reaction_2_Reactor_1_0)
<= 0

c_l_cons(108)_:
+1 B(Reaction_1_Reactor_1_0)
>= 0

c_u_cons(109)_:
+1 B(Reaction_1_Reactor_1_0)
-80 W(Reaction_1_Reactor_1_0)
<= 0

c_e_cons(110)_:
-1 B(Reaction_1_Reactor_1_0)
-1 B(Reaction_2_Reactor_1_0)
-1 B(Reaction_3_Reactor_1_0)
+1 Q(Reactor_1_0)
= 0

c_e_cons(111)_:
+1 Q(Reactor_1_10)
= 0

c_u_cons(112)_:
+1 W(Reaction_1_Reactor_1_0)
+1 W(Reaction_1_Reactor_1_1)
+1 W(Reaction_2_Reactor_1_0)
+1 W(Reaction_2_Reactor_1_1)
+1 W(Reaction_3_Reactor_1_1)
<= 1

c_l_cons(113)_:
+1 B(Reaction_3_Reactor_1_1)
>= 0

c_u_cons(114)_:
+1 B(Reaction_3_Reactor_1_1)
-80 W(Reaction_3_Reactor_1_1)
<= 0

c_l_cons(115)_:
+1 B(Reaction_2_Reactor_1_1)
>= 0

c_u_cons(116)_:
+1 B(Reaction_2_Reactor_1_1)
-80 W(Reaction_2_Reactor_1_1)
<= 0

c_l_cons(117)_:
+1 B(Reaction_1_Reactor_1_1)
>= 0

c_u_cons(118)_:
+1 B(Reaction_1_Reactor_1_1)
-80 W(Reaction_1_Reactor_1_1)
<= 0

c_e_cons(119)_:
-1 B(Reaction_1_Reactor_1_1)
-1 B(Reaction_2_Reactor_1_1)
+1 B(Reaction_3_Reactor_1_0)
-1 B(Reaction_3_Reactor_1_1)
-1 Q(Reactor_1_0)
+1 Q(Reactor_1_1)
= 0

c_e_cons(120)_:
+1 Q(Reactor_1_10)
= 0

c_u_cons(121)_:
+1 W(Reaction_1_Reactor_1_1)
+1 W(Reaction_1_Reactor_1_2)
+1 W(Reaction_2_Reactor_1_1)
+1 W(Reaction_2_Reactor_1_2)
+1 W(Reaction_3_Reactor_1_2)
<= 1

c_l_cons(122)_:
+1 B(Reaction_3_Reactor_1_2)
>= 0

c_u_cons(123)_:
+1 B(Reaction_3_Reactor_1_2)
-80 W(Reaction_3_Reactor_1_2)
<= 0

c_l_cons(124)_:
+1 B(Reaction_2_Reactor_1_2)
>= 0

c_u_cons(125)_:
+1 B(Reaction_2_Reactor_1_2)
-80 W(Reaction_2_Reactor_1_2)
<= 0

c_l_cons(126)_:
+1 B(Reaction_1_Reactor_1_2)
>= 0

c_u_cons(127)_:
+1 B(Reaction_1_Reactor_1_2)
-80 W(Reaction_1_Reactor_1_2)
<= 0

c_e_cons(128)_:
+1 B(Reaction_1_Reactor_1_0)
-1 B(Reaction_1_Reactor_1_2)
+1 B(Reaction_2_Reactor_1_0)
-1 B(Reaction_2_Reactor_1_2)
+1 B(Reaction_3_Reactor_1_1)
-1 B(Reaction_3_Reactor_1_2)
-1 Q(Reactor_1_1)
+1 Q(Reactor_1_2)
= 0

c_e_cons(129)_:
+1 Q(Reactor_1_10)
= 0

c_u_cons(130)_:
+1 W(Reaction_1_Reactor_1_2)
+1 W(Reaction_1_Reactor_1_3)
+1 W(Reaction_2_Reactor_1_2)
+1 W(Reaction_2_Reactor_1_3)
+1 W(Reaction_3_Reactor_1_3)
<= 1

c_l_cons(131)_:
+1 B(Reaction_3_Reactor_1_3)
>= 0

c_u_cons(132)_:
+1 B(Reaction_3_Reactor_1_3)
-80 W(Reaction_3_Reactor_1_3)
<= 0

c_l_cons(133)_:
+1 B(Reaction_2_Reactor_1_3)
>= 0

c_u_cons(134)_:
+1 B(Reaction_2_Reactor_1_3)
-80 W(Reaction_2_Reactor_1_3)
<= 0

c_l_cons(135)_:
+1 B(Reaction_1_Reactor_1_3)
>= 0

c_u_cons(136)_:
+1 B(Reaction_1_Reactor_1_3)
-80 W(Reaction_1_Reactor_1_3)
<= 0

c_e_cons(137)_:
+1 B(Reaction_1_Reactor_1_1)
-1 B(Reaction_1_Reactor_1_3)
+1 B(Reaction_2_Reactor_1_1)
-1 B(Reaction_2_Reactor_1_3)
+1 B(Reaction_3_Reactor_1_2)
-1 B(Reaction_3_Reactor_1_3)
-1 Q(Reactor_1_2)
+1 Q(Reactor_1_3)
= 0

c_e_cons(138)_:
+1 Q(Reactor_1_10)
= 0

c_u_cons(139)_:
+1 W(Reaction_1_Reactor_1_3)
+1 W(Reaction_1_Reactor_1_4)
+1 W(Reaction_2_Reactor_1_3)
+1 W(Reaction_2_Reactor_1_4)
+1 W(Reaction_3_Reactor_1_4)
<= 1

c_l_cons(140)_:
+1 B(Reaction_3_Reactor_1_4)
>= 0

c_u_cons(141)_:
+1 B(Reaction_3_Reactor_1_4)
-80 W(Reaction_3_Reactor_1_4)
<= 0

c_l_cons(142)_:
+1 B(Reaction_2_Reactor_1_4)
>= 0

c_u_cons(143)_:
+1 B(Reaction_2_Reactor_1_4)
-80 W(Reaction_2_Reactor_1_4)
<= 0

c_l_cons(144)_:
+1 B(Reaction_1_Reactor_1_4)
>= 0

c_u_cons(145)_:
+1 B(Reaction_1_Reactor_1_4)
-80 W(Reaction_1_Reactor_1_4)
<= 0

c_e_cons(146)_:
+1 B(Reaction_1_Reactor_1_2)
-1 B(Reaction_1_Reactor_1_4)
+1 B(Reaction_2_Reactor_1_2)
-1 B(Reaction_2_Reactor_1_4)
+1 B(Reaction_3_Reactor_1_3)
-1 B(Reaction_3_Reactor_1_4)
-1 Q(Reactor_1_3)
+1 Q(Reactor_1_4)
= 0

c_e_cons(147)_:
+1 Q(Reactor_1_10)
= 0

c_u_cons(148)_:
+1 W(Reaction_1_Reactor_1_4)
+1 W(Reaction_1_Reactor_1_5)
+1 W(Reaction_2_Reactor_1_4)
+1 W(Reaction_2_Reactor_1_5)
+1 W(Reaction_3_Reactor_1_5)
<= 1

c_l_cons(149)_:
+1 B(Reaction_3_Reactor_1_5)
>= 0

c_u_cons(150)_:
+1 B(Reaction_3_Reactor_1_5)
-80 W(Reaction_3_Reactor_1_5)
<= 0

c_l_cons(151)_:
+1 B(Reaction_2_Reactor_1_5)
>= 0

c_u_cons(152)_:
+1 B(Reaction_2_Reactor_1_5)
-80 W(Reaction_2_Reactor_1_5)
<= 0

c_l_cons(153)_:
+1 B(Reaction_1_Reactor_1_5)
>= 0

c_u_cons(154)_:
+1 B(Reaction_1_Reactor_1_5)
-80 W(Reaction_1_Reactor_1_5)
<= 0

c_e_cons(155)_:
+1 B(Reaction_1_Reactor_1_3)
-1 B(Reaction_1_Reactor_1_5)
+1 B(Reaction_2_Reactor_1_3)
-1 B(Reaction_2_Reactor_1_5)
+1 B(Reaction_3_Reactor_1_4)
-1 B(Reaction_3_Reactor_1_5)
-1 Q(Reactor_1_4)
+1 Q(Reactor_1_5)
= 0

c_e_cons(156)_:
+1 Q(Reactor_1_10)
= 0

c_u_cons(157)_:
+1 W(Reaction_1_Reactor_1_5)
+1 W(Reaction_1_Reactor_1_6)
+1 W(Reaction_2_Reactor_1_5)
+1 W(Reaction_2_Reactor_1_6)
+1 W(Reaction_3_Reactor_1_6)
<= 1

c_l_cons(158)_:
+1 B(Reaction_3_Reactor_1_6)
>= 0

c_u_cons(159)_:
+1 B(Reaction_3_Reactor_1_6)
-80 W(Reaction_3_Reactor_1_6)
<= 0

c_l_cons(160)_:
+1 B(Reaction_2_Reactor_1_6)
>= 0

c_u_cons(161)_:
+1 B(Reaction_2_Reactor_1_6)
-80 W(Reaction_2_Reactor_1_6)
<= 0

c_l_cons(162)_:
+1 B(Reaction_1_Reactor_1_6)
>= 0

c_u_cons(163)_:
+1 B(Reaction_1_Reactor_1_6)
-80 W(Reaction_1_Reactor_1_6)
<= 0

c_e_cons(164)_:
+1 B(Reaction_1_Reactor_1_4)
-1 B(Reaction_1_Reactor_1_6)
+1 B(Reaction_2_Reactor_1_4)
-1 B(Reaction_2_Reactor_1_6)
+1 B(Reaction_3_Reactor_1_5)
-1 B(Reaction_3_Reactor_1_6)
-1 Q(Reactor_1_5)
+1 Q(Reactor_1_6)
= 0

c_e_cons(165)_:
+1 Q(Reactor_1_10)
= 0

c_u_cons(166)_:
+1 W(Reaction_1_Reactor_1_6)
+1 W(Reaction_1_Reactor_1_7)
+1 W(Reaction_2_Reactor_1_6)
+1 W(Reaction_2_Reactor_1_7)
+1 W(Reaction_3_Reactor_1_7)
<= 1

c_l_cons(167)_:
+1 B(Reaction_3_Reactor_1_7)
>= 0

c_u_cons(168)_:
+1 B(Reaction_3_Reactor_1_7)
-80 W(Reaction_3_Reactor_1_7)
<= 0

c_l_cons(169)_:
+1 B(Reaction_2_Reactor_1_7)
>= 0

c_u_cons(170)_:
+1 B(Reaction_2_Reactor_1_7)
-80 W(Reaction_2_Reactor_1_7)
<= 0

c_l_cons(171)_:
+1 B(Reaction_1_Reactor_1_7)
>= 0

c_u_cons(172)_:
+1 B(Reaction_1_Reactor_1_7)
-80 W(Reaction_1_Reactor_1_7)
<= 0

c_e_cons(173)_:
+1 B(Reaction_1_Reactor_1_5)
-1 B(Reaction_1_Reactor_1_7)
+1 B(Reaction_2_Reactor_1_5)
-1 B(Reaction_2_Reactor_1_7)
+1 B(Reaction_3_Reactor_1_6)
-1 B(Reaction_3_Reactor_1_7)
-1 Q(Reactor_1_6)
+1 Q(Reactor_1_7)
= 0

c_e_cons(174)_:
+1 Q(Reactor_1_10)
= 0

c_u_cons(175)_:
+1 W(Reaction_1_Reactor_1_7)
+1 W(Reaction_1_Reactor_1_8)
+1 W(Reaction_2_Reactor_1_7)
+1 W(Reaction_2_Reactor_1_8)
+1 W(Reaction_3_Reactor_1_8)
<= 1

c_l_cons(176)_:
+1 B(Reaction_3_Reactor_1_8)
>= 0

c_u_cons(177)_:
+1 B(Reaction_3_Reactor_1_8)
-80 W(Reaction_3_Reactor_1_8)
<= 0

c_l_cons(178)_:
+1 B(Reaction_2_Reactor_1_8)
>= 0

c_u_cons(179)_:
+1 B(Reaction_2_Reactor_1_8)
-80 W(Reaction_2_Reactor_1_8)
<= 0

c_l_cons(180)_:
+1 B(Reaction_1_Reactor_1_8)
>= 0

c_u_cons(181)_:
+1 B(Reaction_1_Reactor_1_8)
-80 W(Reaction_1_Reactor_1_8)
<= 0

c_e_cons(182)_:
+1 B(Reaction_1_Reactor_1_6)
-1 B(Reaction_1_Reactor_1_8)
+1 B(Reaction_2_Reactor_1_6)
-1 B(Reaction_2_Reactor_1_8)
+1 B(Reaction_3_Reactor_1_7)
-1 B(Reaction_3_Reactor_1_8)
-1 Q(Reactor_1_7)
+1 Q(Reactor_1_8)
= 0

c_e_cons(183)_:
+1 Q(Reactor_1_10)
= 0

c_u_cons(184)_:
+1 W(Reaction_1_Reactor_1_8)
+1 W(Reaction_1_Reactor_1_9)
+1 W(Reaction_2_Reactor_1_8)
+1 W(Reaction_2_Reactor_1_9)
+1 W(Reaction_3_Reactor_1_9)
<= 1

c_l_cons(185)_:
+1 B(Reaction_3_Reactor_1_9)
>= 0

c_u_cons(186)_:
+1 B(Reaction_3_Reactor_1_9)
-80 W(Reaction_3_Reactor_1_9)
<= 0

c_l_cons(187)_:
+1 B(Reaction_2_Reactor_1_9)
>= 0

c_u_cons(188)_:
+1 B(Reaction_2_Reactor_1_9)
-80 W(Reaction_2_Reactor_1_9)
<= 0

c_l_cons(189)_:
+1 B(Reaction_1_Reactor_1_9)
>= 0

c_u_cons(190)_:
+1 B(Reaction_1_Reactor_1_9)
-80 W(Reaction_1_Reactor_1_9)
<= 0

c_e_cons(191)_:
+1 B(Reaction_1_Reactor_1_7)
-1 B(Reaction_1_Reactor_1_9)
+1 B(Reaction_2_Reactor_1_7)
-1 B(Reaction_2_Reactor_1_9)
+1 B(Reaction_3_Reactor_1_8)
-1 B(Reaction_3_Reactor_1_9)
-1 Q(Reactor_1_8)
+1 Q(Reactor_1_9)
= 0

c_e_cons(192)_:
+1 Q(Reactor_1_10)
= 0

c_u_cons(193)_:
+1 W(Reaction_1_Reactor_1_10)
+1 W(Reaction_1_Reactor_1_9)
+1 W(Reaction_2_Reactor_1_10)
+1 W(Reaction_2_Reactor_1_9)
+1 W(Reaction_3_Reactor_1_10)
<= 1

c_l_cons(194)_:
+1 B(Reaction_3_Reactor_1_10)
>= 0

c_u_cons(195)_:
+1 B(Reaction_3_Reactor_1_10)
-80 W(Reaction_3_Reactor_1_10)
<= 0

c_l_cons(196)_:
+1 B(Reaction_2_Reactor_1_10)
>= 0

c_u_cons(197)_:
+1 B(Reaction_2_Reactor_1_10)
-80 W(Reaction_2_Reactor_1_10)
<= 0

c_l_cons(198)_:
+1 B(Reaction_1_Reactor_1_10)
>= 0

c_u_cons(199)_:
+1 B(Reaction_1_Reactor_1_10)
-80 W(Reaction_1_Reactor_1_10)
<= 0

c_e_cons(200)_:
-1 B(Reaction_1_Reactor_1_10)
+1 B(Reaction_1_Reactor_1_8)
-1 B(Reaction_2_Reactor_1_10)
+1 B(Reaction_2_Reactor_1_8)
-1 B(Reaction_3_Reactor_1_10)
+1 B(Reaction_3_Reactor_1_9)
+1 Q(Reactor_1_10)
-1 Q(Reactor_1_9)
= 0

c_e_cons(201)_:
+1 Q(Reactor_1_10)
= 0

c_u_cons(202)_:
+1 W(Separation_Still_0)
<= 1

c_l_cons(203)_:
+1 B(Separation_Still_0)
>= 0

c_u_cons(204)_:
+1 B(Separation_Still_0)
-200 W(Separation_Still_0)
<= 0

c_e_cons(205)_:
-1 B(Separation_Still_0)
+1 Q(Still_0)
= 0

c_e_cons(206)_:
+1 Q(Still_10)
= 0

c_u_cons(207)_:
+1 W(Separation_Still_0)
+1 W(Separation_Still_1)
<= 1

c_l_cons(208)_:
+1 B(Separation_Still_1)
>= 0

c_u_cons(209)_:
+1 B(Separation_Still_1)
-200 W(Separation_Still_1)
<= 0

c_e_cons(210)_:
+0.90000000000000002 B(Separation_Still_0)
-1 B(Separation_Still_1)
-1 Q(Still_0)
+1 Q(Still_1)
= 0

c_e_cons(211)_:
+1 Q(Still_10)
= 0

c_u_cons(212)_:
+1 W(Separation_Still_1)
+1 W(Separation_Still_2)
<= 1

c_l_cons(213)_:
+1 B(Separation_Still_2)
>= 0

c_u_cons(214)_:
+1 B(Separation_Still_2)
-200 W(Separation_Still_2)
<= 0

c_e_cons(215)_:
+0.10000000000000001 B(Separation_Still_0)
+0.90000000000000002 B(Separation_Still_1)
-1 B(Separation_Still_2)
-1 Q(Still_1)
+1 Q(Still_2)
= 0

c_e_cons(216)_:
+1 Q(Still_10)
= 0

c_u_cons(217)_:
+1 W(Separation_Still_2)
+1 W(Separation_Still_3)
<= 1

c_l_cons(218)_:
+1 B(Separation_Still_3)
>= 0

c_u_cons(219)_:
+1 B(Separation_Still_3)
-200 W(Separation_Still_3)
<= 0

c_e_cons(220)_:
+0.10000000000000001 B(Separation_Still_1)
+0.90000000000000002 B(Separation_Still_2)
-1 B(Separation_Still_3)
-1 Q(Still_2)
+1 Q(Still_3)
= 0

c_e_cons(221)_:
+1 Q(Still_10)
= 0

c_u_cons(222)_:
+1 W(Separation_Still_3)
+1 W(Separation_Still_4)
<= 1

c_l_cons(223)_:
+1 B(Separation_Still_4)
>= 0

c_u_cons(224)_:
+1 B(Separation_Still_4)
-200 W(Separation_Still_4)
<= 0

c_e_cons(225)_:
+0.10000000000000001 B(Separation_Still_2)
+0.90000000000000002 B(Separation_Still_3)
-1 B(Separation_Still_4)
-1 Q(Still_3)
+1 Q(Still_4)
= 0

c_e_cons(226)_:
+1 Q(Still_10)
= 0

c_u_cons(227)_:
+1 W(Separation_Still_4)
+1 W(Separation_Still_5)
<= 1

c_l_cons(228)_:
+1 B(Separation_Still_5)
>= 0

c_u_cons(229)_:
+1 B(Separation_Still_5)
-200 W(Separation_Still_5)
<= 0

c_e_cons(230)_:
+0.10000000000000001 B(Separation_Still_3)
+0.90000000000000002 B(Separation_Still_4)
-1 B(Separation_Still_5)
-1 Q(Still_4)
+1 Q(Still_5)
= 0

c_e_cons(231)_:
+1 Q(Still_10)
= 0

c_u_cons(232)_:
+1 W(Separation_Still_5)
+1 W(Separation_Still_6)
<= 1

c_l_cons(233)_:
+1 B(Separation_Still_6)
>= 0

c_u_cons(234)_:
+1 B(Separation_Still_6)
-200 W(Separation_Still_6)
<= 0

c_e_cons(235)_:
+0.10000000000000001 B(Separation_Still_4)
+0.90000000000000002 B(Separation_Still_5)
-1 B(Separation_Still_6)
-1 Q(Still_5)
+1 Q(Still_6)
= 0

c_e_cons(236)_:
+1 Q(Still_10)
= 0

c_u_cons(237)_:
+1 W(Separation_Still_6)
+1 W(Separation_Still_7)
<= 1

c_l_cons(238)_:
+1 B(Separation_Still_7)
>= 0

c_u_cons(239)_:
+1 B(Separation_Still_7)
-200 W(Separation_Still_7)
<= 0

c_e_cons(240)_:
+0.10000000000000001 B(Separation_Still_5)
+0.90000000000000002 B(Separation_Still_6)
-1 B(Separation_Still_7)
-1 Q(Still_6)
+1 Q(Still_7)
= 0

c_e_cons(241)_:
+1 Q(Still_10)
= 0

c_u_cons(242)_:
+1 W(Separation_Still_7)
+1 W(Separation_Still_8)
<= 1

c_l_cons(243)_:
+1 B(Separation_Still_8)
>= 0

c_u_cons(244)_:
+1 B(Separation_Still_8)
-200 W(Separation_Still_8)
<= 0

c_e_cons(245)_:
+0.10000000000000001 B(Separation_Still_6)
+0.90000000000000002 B(Separation_Still_7)
-1 B(Separation_Still_8)
-1 Q(Still_7)
+1 Q(Still_8)
= 0

c_e_cons(246)_:
+1 Q(Still_10)
= 0

c_u_cons(247)_:
+1 W(Separation_Still_8)
+1 W(Separation_Still_9)
<= 1

c_l_cons(248)_:
+1 B(Separation_Still_9)
>= 0

c_u_cons(249)_:
+1 B(Separation_Still_9)
-200 W(Separation_Still_9)
<= 0

c_e_cons(250)_:
+0.10000000000000001 B(Separation_Still_7)
+0.90000000000000002 B(Separation_Still_8)
-1 B(Separation_Still_9)
-1 Q(Still_8)
+1 Q(Still_9)
= 0

c_e_cons(251)_:
+1 Q(Still_10)
= 0

c_u_cons(252)_:
+1 W(Separation_Still_10)
+1 W(Separation_Still_9)
<= 1

c_l_cons(253)_:
+1 B(Separation_Still_10)
>= 0

c_u_cons(254)_:
+1 B(Separation_Still_10)
-200 W(Separation_Still_10)
<= 0

c_e_cons(255)_:
-1 B(Separation_Still_10)
+0.10000000000000001 B(Separation_Still_8)
+0.90000000000000002 B(Separation_Still_9)
+1 Q(Still_10)
-1 Q(Still_9)
= 0

c_e_cons(256)_:
+1 Q(Still_10)
= 0

c_u_cons(257)_:
+1 W(Heating_Heater_0)
<= 1

c_l_cons(258)_:
+1 B(Heating_Heater_0)
>= 0

c_u_cons(259)_:
+1 B(Heating_Heater_0)
-100 W(Heating_Heater_0)
<= 0

c_e_cons(260)_:
-1 B(Heating_Heater_0)
+1 Q(Heater_0)
= 0

c_e_cons(261)_:
+1 Q(Heater_10)
= 0

c_u_cons(262)_:
+1 W(Heating_Heater_1)
<= 1

c_l_cons(263)_:
+1 B(Heating_Heater_1)
>= 0

c_u_cons(264)_:
+1 B(Heating_Heater_1)
-100 W(Heating_Heater_1)
<= 0

c_e_cons(265)_:
+1 B(Heating_Heater_0)
-1 B(Heating_Heater_1)
-1 Q(Heater_0)
+1 Q(Heater_1)
= 0

c_e_cons(266)_:
+1 Q(Heater_10)
= 0

c_u_cons(267)_:
+1 W(Heating_Heater_2)
<= 1

c_l_cons(268)_:
+1 B(Heating_Heater_2)
>= 0

c_u_cons(269)_:
+1 B(Heating_Heater_2)
-100 W(Heating_Heater_2)
<= 0

c_e_cons(270)_:
+1 B(Heating_Heater_1)
-1 B(Heating_Heater_2)
-1 Q(Heater_1)
+1 Q(Heater_2)
= 0

c_e_cons(271)_:
+1 Q(Heater_10)
= 0

c_u_cons(272)_:
+1 W(Heating_Heater_3)
<= 1

c_l_cons(273)_:
+1 B(Heating_Heater_3)
>= 0

c_u_cons(274)_:
+1 B(Heating_Heater_3)
-100 W(Heating_Heater_3)
<= 0

c_e_cons(275)_:
+1 B(Heating_Heater_2)
-1 B(Heating_Heater_3)
-1 Q(Heater_2)
+1 Q(Heater_3)
= 0

c_e_cons(276)_:
+1 Q(Heater_10)
= 0

c_u_cons(277)_:
+1 W(Heating_Heater_4)
<= 1

c_l_cons(278)_:
+1 B(Heating_Heater_4)
>= 0

c_u_cons(279)_:
+1 B(Heating_Heater_4)
-100 W(Heating_Heater_4)
<= 0

c_e_cons(280)_:
+1 B(Heating_Heater_3)
-1 B(Heating_Heater_4)
-1 Q(Heater_3)
+1 Q(Heater_4)
= 0

c_e_cons(281)_:
+1 Q(Heater_10)
= 0

c_u_cons(282)_:
+1 W(Heating_Heater_5)
<= 1

c_l_cons(283)_:
+1 B(Heating_Heater_5)
>= 0

c_u_cons(284)_:
+1 B(Heating_Heater_5)
-100 W(Heating_Heater_5)
<= 0

c_e_cons(285)_:
+1 B(Heating_Heater_4)
-1 B(Heating_Heater_5)
-1 Q(Heater_4)
+1 Q(Heater_5)
= 0

c_e_cons(286)_:
+1 Q(Heater_10)
= 0

c_u_cons(287)_:
+1 W(Heating_Heater_6)
<= 1

c_l_cons(288)_:
+1 B(Heating_Heater_6)
>= 0

c_u_cons(289)_:
+1 B(Heating_Heater_6)
-100 W(Heating_Heater_6)
<= 0

c_e_cons(290)_:
+1 B(Heating_Heater_5)
-1 B(Heating_Heater_6)
-1 Q(Heater_5)
+1 Q(Heater_6)
= 0

c_e_cons(291)_:
+1 Q(Heater_10)
= 0

c_u_cons(292)_:
+1 W(Heating_Heater_7)
<= 1

c_l_cons(293)_:
+1 B(Heating_Heater_7)
>= 0

c_u_cons(294)_:
+1 B(Heating_Heater_7)
-100 W(Heating_Heater_7)
<= 0

c_e_cons(295)_:
+1 B(Heating_Heater_6)
-1 B(Heating_Heater_7)
-1 Q(Heater_6)
+1 Q(Heater_7)
= 0

c_e_cons(296)_:
+1 Q(Heater_10)
= 0

c_u_cons(297)_:
+1 W(Heating_Heater_8)
<= 1

c_l_cons(298)_:
+1 B(Heating_Heater_8)
>= 0

c_u_cons(299)_:
+1 B(Heating_Heater_8)
-100 W(Heating_Heater_8)
<= 0

c_e_cons(300)_:
+1 B(Heating_Heater_7)
-1 B(Heating_Heater_8)
-1 Q(Heater_7)
+1 Q(Heater_8)
= 0

c_e_cons(301)_:
+1 Q(Heater_10)
= 0

c_u_cons(302)_:
+1 W(Heating_Heater_9)
<= 1

c_l_cons(303)_:
+1 B(Heating_Heater_9)
>= 0

c_u_cons(304)_:
+1 B(Heating_Heater_9)
-100 W(Heating_Heater_9)
<= 0

c_e_cons(305)_:
+1 B(Heating_Heater_8)
-1 B(Heating_Heater_9)
-1 Q(Heater_8)
+1 Q(Heater_9)
= 0

c_e_cons(306)_:
+1 Q(Heater_10)
= 0

c_u_cons(307)_:
+1 W(Heating_Heater_10)
<= 1

c_l_cons(308)_:
+1 B(Heating_Heater_10)
>= 0

c_u_cons(309)_:
+1 B(Heating_Heater_10)
-100 W(Heating_Heater_10)
<= 0

c_e_cons(310)_:
-1 B(Heating_Heater_10)
+1 B(Heating_Heater_9)
+1 Q(Heater_10)
-1 Q(Heater_9)
= 0

c_e_cons(311)_:
+1 Q(Heater_10)
= 0

c_e_cons(313)_:
+1 S(Product_2_0)
= 0

c_e_cons(315)_:
-0.90000000000000002 B(Separation_Still_0)
-1 S(Product_2_0)
+1 S(Product_2_1)
= 0

c_e_cons(317)_:
-0.90000000000000002 B(Separation_Still_1)
-1 S(Product_2_1)
+1 S(Product_2_2)
= 0

c_e_cons(319)_:
-0.90000000000000002 B(Separation_Still_2)
-1 S(Product_2_2)
+1 S(Product_2_3)
= 0

c_e_cons(321)_:
-0.90000000000000002 B(Separation_Still_3)
-1 S(Product_2_3)
+1 S(Product_2_4)
= 0

c_e_cons(323)_:
-0.90000000000000002 B(Separation_Still_4)
-1 S(Product_2_4)
+1 S(Product_2_5)
= 0

c_e_cons(325)_:
-0.90000000000000002 B(Separation_Still_5)
-1 S(Product_2_5)
+1 S(Product_2_6)
= 0

c_e_cons(327)_:
-0.90000000000000002 B(Separation_Still_6)
-1 S(Product_2_6)
+1 S(Product_2_7)
= 0

c_e_cons(329)_:
-0.90000000000000002 B(Separation_Still_7)
-1 S(Product_2_7)
+1 S(Product_2_8)
= 0

c_e_cons(331)_:
-0.90000000000000002 B(Separation_Still_8)
-1 S(Product_2_8)
+1 S(Product_2_9)
= 0

c_e_cons(333)_:
-0.90000000000000002 B(Separation_Still_9)
+1 S(Product_2_10)
-1 S(Product_2_9)
= 0

c_e_cons(335)_:
+0.59999999999999998 B(Reaction_2_Reactor_1_0)
+0.59999999999999998 B(Reaction_2_Reactor_2_0)
+1 S(IntBC_0)
= 0

c_e_cons(337)_:
+0.59999999999999998 B(Reaction_2_Reactor_1_1)
+0.59999999999999998 B(Reaction_2_Reactor_2_1)
-1 S(IntBC_0)
+1 S(IntBC_1)
= 0

c_e_cons(339)_:
-1 B(Reaction_1_Reactor_1_0)
-1 B(Reaction_1_Reactor_2_0)
+0.59999999999999998 B(Reaction_2_Reactor_1_2)
+0.59999999999999998 B(Reaction_2_Reactor_2_2)
-1 S(IntBC_1)
+1 S(IntBC_2)
= 0

c_e_cons(341)_:
-1 B(Reaction_1_Reactor_1_1)
-1 B(Reaction_1_Reactor_2_1)
+0.59999999999999998 B(Reaction_2_Reactor_1_3)
+0.59999999999999998 B(Reaction_2_Reactor_2_3)
-1 S(IntBC_2)
+1 S(IntBC_3)
= 0

c_e_cons(343)_:
-1 B(Reaction_1_Reactor_1_2)
-1 B(Reaction_1_Reactor_2_2)
+0.59999999999999998 B(Reaction_2_Reactor_1_4)
+0.59999999999999998 B(Reaction_2_Reactor_2_4)
-1 S(IntBC_3)
+1 S(IntBC_4)
= 0

c_e_cons(345)_:
-1 B(Reaction_1_Reactor_1_3)
-1 B(Reaction_1_Reactor_2_3)
+0.59999999999999998 B(Reaction_2_Reactor_1_5)
+0.59999999999999998 B(Reaction_2_Reactor_2_5)
-1 S(IntBC_4)
+1 S(IntBC_5)
= 0

c_e_cons(347)_:
-1 B(Reaction_1_Reactor_1_4)
-1 B(Reaction_1_Reactor_2_4)
+0.59999999999999998 B(Reaction_2_Reactor_1_6)
+0.59999999999999998 B(Reaction_2_Reactor_2_6)
-1 S(IntBC_5)
+1 S(IntBC_6)
= 0

c_e_cons(349)_:
-1 B(Reaction_1_Reactor_1_5)
-1 B(Reaction_1_Reactor_2_5)
+0.59999999999999998 B(Reaction_2_Reactor_1_7)
+0.59999999999999998 B(Reaction_2_Reactor_2_7)
-1 S(IntBC_6)
+1 S(IntBC_7)
= 0

c_e_cons(351)_:
-1 B(Reaction_1_Reactor_1_6)
-1 B(Reaction_1_Reactor_2_6)
+0.59999999999999998 B(Reaction_2_Reactor_1_8)
+0.59999999999999998 B(Reaction_2_Reactor_2_8)
-1 S(IntBC_7)
+1 S(IntBC_8)
= 0

c_e_cons(353)_:
-1 B(Reaction_1_Reactor_1_7)
-1 B(Reaction_1_Reactor_2_7)
+0.59999999999999998 B(Reaction_2_Reactor_1_9)
+0.59999999999999998 B(Reaction_2_Reactor_2_9)
-1 S(IntBC_8)
+1 S(IntBC_9)
= 0

c_e_cons(355)_:
-1 B(Reaction_1_Reactor_1_8)
-1 B(Reaction_1_Reactor_2_8)
+0.59999999999999998 B(Reaction_2_Reactor_1_10)
+0.59999999999999998 B(Reaction_2_Reactor_2_10)
+1 S(IntBC_10)
-1 S(IntBC_9)
= 0

c_e_cons(357)_:
+0.5 B(Reaction_1_Reactor_1_0)
+0.5 B(Reaction_1_Reactor_2_0)
+1 S(FeedB_0)
= 200

c_e_cons(359)_:
+0.5 B(Reaction_1_Reactor_1_1)
+0.5 B(Reaction_1_Reactor_2_1)
-1 S(FeedB_0)
+1 S(FeedB_1)
= 0

c_e_cons(361)_:
+0.5 B(Reaction_1_Reactor_1_2)
+0.5 B(Reaction_1_Reactor_2_2)
-1 S(FeedB_1)
+1 S(FeedB_2)
= 0

c_e_cons(363)_:
+0.5 B(Reaction_1_Reactor_1_3)
+0.5 B(Reaction_1_Reactor_2_3)
-1 S(FeedB_2)
+1 S(FeedB_3)
= 0

c_e_cons(365)_:
+0.5 B(Reaction_1_Reactor_1_4)
+0.5 B(Reaction_1_Reactor_2_4)
-1 S(FeedB_3)
+1 S(FeedB_4)
= 0

c_e_cons(367)_:
+0.5 B(Reaction_1_Reactor_1_5)
+0.5 B(Reaction_1_Reactor_2_5)
-1 S(FeedB_4)
+1 S(FeedB_5)
= 0

c_e_cons(369)_:
+0.5 B(Reaction_1_Reactor_1_6)
+0.5 B(Reaction_1_Reactor_2_6)
-1 S(FeedB_5)
+1 S(FeedB_6)
= 0

c_e_cons(371)_:
+0.5 B(Reaction_1_Reactor_1_7)
+0.5 B(Reaction_1_Reactor_2_7)
-1 S(FeedB_6)
+1 S(FeedB_7)
= 0

c_e_cons(373)_:
+0.5 B(Reaction_1_Reactor_1_8)
+0.5 B(Reaction_1_Reactor_2_8)
-1 S(FeedB_7)
+1 S(FeedB_8)
= 0

c_e_cons(375)_:
+0.5 B(Reaction_1_Reactor_1_9)
+0.5 B(Reaction_1_Reactor_2_9)
-1 S(FeedB_8)
+1 S(FeedB_9)
= 0

c_e_cons(377)_:
+0.5 B(Reaction_1_Reactor_1_10)
+0.5 B(Reaction_1_Reactor_2_10)
+1 S(FeedB_10)
-1 S(FeedB_9)
= 0

c_e_cons(379)_:
+1 B(Heating_Heater_0)
+1 S(FeedA_0)
= 200

c_e_cons(381)_:
+1 B(Heating_Heater_1)
-1 S(FeedA_0)
+1 S(FeedA_1)
= 0

c_e_cons(383)_:
+1 B(Heating_Heater_2)
-1 S(FeedA_1)
+1 S(FeedA_2)
= 0

c_e_cons(385)_:
+1 B(Heating_Heater_3)
-1 S(FeedA_2)
+1 S(FeedA_3)
= 0

c_e_cons(387)_:
+1 B(Heating_Heater_4)
-1 S(FeedA_3)
+1 S(FeedA_4)
= 0

c_e_cons(389)_:
+1 B(Heating_Heater_5)
-1 S(FeedA_4)
+1 S(FeedA_5)
= 0

c_e_cons(391)_:
+1 B(Heating_Heater_6)
-1 S(FeedA_5)
+1 S(FeedA_6)
= 0

c_e_cons(393)_:
+1 B(Heating_Heater_7)
-1 S(FeedA_6)
+1 S(FeedA_7)
= 0

c_e_cons(395)_:
+1 B(Heating_Heater_8)
-1 S(FeedA_7)
+1 S(FeedA_8)
= 0

c_e_cons(397)_:
+1 B(Heating_Heater_9)
-1 S(FeedA_8)
+1 S(FeedA_9)
= 0

c_e_cons(399)_:
+1 B(Heating_Heater_10)
+1 S(FeedA_10)
-1 S(FeedA_9)
= 0

c_e_cons(401)_:
+0.80000000000000004 B(Reaction_3_Reactor_1_0)
+0.80000000000000004 B(Reaction_3_Reactor_2_0)
+1 S(IntAB_0)
= 0

c_e_cons(403)_:
+0.80000000000000004 B(Reaction_3_Reactor_1_1)
+0.80000000000000004 B(Reaction_3_Reactor_2_1)
-1 S(IntAB_0)
+1 S(IntAB_1)
= 0

c_e_cons(405)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_0)
-0.59999999999999998 B(Reaction_2_Reactor_2_0)
+0.80000000000000004 B(Reaction_3_Reactor_1_2)
+0.80000000000000004 B(Reaction_3_Reactor_2_2)
-0.10000000000000001 B(Separation_Still_0)
-1 S(IntAB_1)
+1 S(IntAB_2)
= 0

c_e_cons(407)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_1)
-0.59999999999999998 B(Reaction_2_Reactor_2_1)
+0.80000000000000004 B(Reaction_3_Reactor_1_3)
+0.80000000000000004 B(Reaction_3_Reactor_2_3)
-0.10000000000000001 B(Separation_Still_1)
-1 S(IntAB_2)
+1 S(IntAB_3)
= 0

c_e_cons(409)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_2)
-0.59999999999999998 B(Reaction_2_Reactor_2_2)
+0.80000000000000004 B(Reaction_3_Reactor_1_4)
+0.80000000000000004 B(Reaction_3_Reactor_2_4)
-0.10000000000000001 B(Separation_Still_2)
-1 S(IntAB_3)
+1 S(IntAB_4)
= 0

c_e_cons(411)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_3)
-0.59999999999999998 B(Reaction_2_Reactor_2_3)
+0.80000000000000004 B(Reaction_3_Reactor_1_5)
+0.80000000000000004 B(Reaction_3_Reactor_2_5)
-0.10000000000000001 B(Separation_Still_3)
-1 S(IntAB_4)
+1 S(IntAB_5)
= 0

c_e_cons(413)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_4)
-0.59999999999999998 B(Reaction_2_Reactor_2_4)
+0.80000000000000004 B(Reaction_3_Reactor_1_6)
+0.80000000000000004 B(Reaction_3_Reactor_2_6)
-0.10000000000000001 B(Separation_Still_4)
-1 S(IntAB_5)
+1 S(IntAB_6)
= 0

c_e_cons(415)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_5)
-0.59999999999999998 B(Reaction_2_Reactor_2_5)
+0.80000000000000004 B(Reaction_3_Reactor_1_7)
+0.80000000000000004 B(Reaction_3_Reactor_2_7)
-0.10000000000000001 B(Separation_Still_5)
-1 S(IntAB_6)
+1 S(IntAB_7)
= 0

c_e_cons(417)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_6)
-0.59999999999999998 B(Reaction_2_Reactor_2_6)
+0.80000000000000004 B(Reaction_3_Reactor_1_8)
+0.80000000000000004 B(Reaction_3_Reactor_2_8)
-0.10000000000000001 B(Separation_Still_6)
-1 S(IntAB_7)
+1 S(IntAB_8)
= 0

c_e_cons(419)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_7)
-0.59999999999999998 B(Reaction_2_Reactor_2_7)
+0.80000000000000004 B(Reaction_3_Reactor_1_9)
+0.80000000000000004 B(Reaction_3_Reactor_2_9)
-0.10000000000000001 B(Separation_Still_7)
-1 S(IntAB_8)
+1 S(IntAB_9)
= 0

c_e_cons(421)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_8)
-0.59999999999999998 B(Reaction_2_Reactor_2_8)
+0.80000000000000004 B(Reaction_3_Reactor_1_10)
+0.80000000000000004 B(Reaction_3_Reactor_2_10)
-0.10000000000000001 B(Separation_Still_8)
+1 S(IntAB_10)
-1 S(IntAB_9)
= 0

c_e_cons(423)_:
+0.5 B(Reaction_1_Reactor_1_0)
+0.5 B(Reaction_1_Reactor_2_0)
+0.20000000000000001 B(Reaction_3_Reactor_1_0)
+0.20000000000000001 B(Reaction_3_Reactor_2_0)
+1 S(FeedC_0)
= 200

c_e_cons(425)_:
+0.5 B(Reaction_1_Reactor_1_1)
+0.5 B(Reaction_1_Reactor_2_1)
+0.20000000000000001 B(Reaction_3_Reactor_1_1)
+0.20000000000000001 B(Reaction_3_Reactor_2_1)
-1 S(FeedC_0)
+1 S(FeedC_1)
= 0

c_e_cons(427)_:
+0.5 B(Reaction_1_Reactor_1_2)
+0.5 B(Reaction_1_Reactor_2_2)
+0.20000000000000001 B(Reaction_3_Reactor_1_2)
+0.20000000000000001 B(Reaction_3_Reactor_2_2)
-1 S(FeedC_1)
+1 S(FeedC_2)
= 0

c_e_cons(429)_:
+0.5 B(Reaction_1_Reactor_1_3)
+0.5 B(Reaction_1_Reactor_2_3)
+0.20000000000000001 B(Reaction_3_Reactor_1_3)
+0.20000000000000001 B(Reaction_3_Reactor_2_3)
-1 S(FeedC_2)
+1 S(FeedC_3)
= 0

c_e_cons(431)_:
+0.5 B(Reaction_1_Reactor_1_4)
+0.5 B(Reaction_1_Reactor_2_4)
+0.20000000000000001 B(Reaction_3_Reactor_1_4)
+0.20000000000000001 B(Reaction_3_Reactor_2_4)
-1 S(FeedC_3)
+1 S(FeedC_4)
= 0

c_e_cons(433)_:
+0.5 B(Reaction_1_Reactor_1_5)
+0.5 B(Reaction_1_Reactor_2_5)
+0.20000000000000001 B(Reaction_3_Reactor_1_5)
+0.20000000000000001 B(Reaction_3_Reactor_2_5)
-1 S(FeedC_4)
+1 S(FeedC_5)
= 0

c_e_cons(435)_:
+0.5 B(Reaction_1_Reactor_1_6)
+0.5 B(Reaction_1_Reactor_2_6)
+0.20000000000000001 B(Reaction_3_Reactor_1_6)
+0.20000000000000001 B(Reaction_3_Reactor_2_6)
-1 S(FeedC_5)
+1 S(FeedC_6)
= 0

c_e_cons(437)_:
+0.5 B(Reaction_1_Reactor_1_7)
+0.5 B(Reaction_1_Reactor_2_7)
+0.20000000000000001 B(Reaction_3_Reactor_1_7)
+0.20000000000000001 B(Reaction_3_Reactor_2_7)
-1 S(FeedC_6)
+1 S(FeedC_7)
= 0

c_e_cons(439)_:
+0.5 B(Reaction_1_Reactor_1_8)
+0.5 B(Reaction_1_Reactor_2_8)
+0.20000000000000001 B(Reaction_3_Reactor_1_8)
+0.20000000000000001 B(Reaction_3_Reactor_2_8)
-1 S(FeedC_7)
+1 S(FeedC_8)
= 0

c_e_cons(441)_:
+0.5 B(Reaction_1_Reactor_1_9)
+0.5 B(Reaction_1_Reactor_2_9)
+0.20000000000000001 B(Reaction_3_Reactor_1_9)
+0.20000000000000001 B(Reaction_3_Reactor_2_9)
-1 S(FeedC_8)
+1 S(FeedC_9)
= 0

c_e_cons(443)_:
+0.5 B(Reaction_1_Reactor_1_10)
+0.5 B(Reaction_1_Reactor_2_10)
+0.20000000000000001 B(Reaction_3_Reactor_1_10)
+0.20000000000000001 B(Reaction_3_Reactor_2_10)
+1 S(FeedC_10)
-1 S(FeedC_9)
= 0

c_e_cons(445)_:
+1 S(Product_1_0)
= 0

c_e_cons(447)_:
-1 S(Product_1_0)
+1 S(Product_1_1)
= 0

c_e_cons(449)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_0)
-0.40000000000000002 B(Reaction_2_Reactor_2_0)
-1 S(Product_1_1)
+1 S(Product_1_2)
= 0

c_e_cons(451)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_1)
-0.40000000000000002 B(Reaction_2_Reactor_2_1)
-1 S(Product_1_2)
+1 S(Product_1_3)
= 0

c_e_cons(453)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_2)
-0.40000000000000002 B(Reaction_2_Reactor_2_2)
-1 S(Product_1_3)
+1 S(Product_1_4)
= 0

c_e_cons(455)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_3)
-0.40000000000000002 B(Reaction_2_Reactor_2_3)
-1 S(Product_1_4)
+1 S(Product_1_5)
= 0

c_e_cons(457)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_4)
-0.40000000000000002 B(Reaction_2_Reactor_2_4)
-1 S(Product_1_5)
+1 S(Product_1_6)
= 0

c_e_cons(459)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_5)
-0.40000000000000002 B(Reaction_2_Reactor_2_5)
-1 S(Product_1_6)
+1 S(Product_1_7)
= 0

c_e_cons(461)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_6)
-0.40000000000000002 B(Reaction_2_Reactor_2_6)
-1 S(Product_1_7)
+1 S(Product_1_8)
= 0

c_e_cons(463)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_7)
-0.40000000000000002 B(Reaction_2_Reactor_2_7)
-1 S(Product_1_8)
+1 S(Product_1_9)
= 0

c_e_cons(465)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_8)
-0.40000000000000002 B(Reaction_2_Reactor_2_8)
+1 S(Product_1_10)
-1 S(Product_1_9)
= 0

c_e_cons(467)_:
+1 B(Separation_Still_0)
+1 S(ImpureE_0)
= 0

c_e_cons(469)_:
-1 B(Reaction_3_Reactor_1_0)
-1 B(Reaction_3_Reactor_2_0)
+1 B(Separation_Still_1)
-1 S(ImpureE_0)
+1 S(ImpureE_1)
= 0

c_e_cons(471)_:
-1 B(Reaction_3_Reactor_1_1)
-1 B(Reaction_3_Reactor_2_1)
+1 B(Separation_Still_2)
-1 S(ImpureE_1)
+1 S(ImpureE_2)
= 0

c_e_cons(473)_:
-1 B(Reaction_3_Reactor_1_2)
-1 B(Reaction_3_Reactor_2_2)
+1 B(Separation_Still_3)
-1 S(ImpureE_2)
+1 S(ImpureE_3)
= 0

c_e_cons(475)_:
-1 B(Reaction_3_Reactor_1_3)
-1 B(Reaction_3_Reactor_2_3)
+1 B(Separation_Still_4)
-1 S(ImpureE_3)
+1 S(ImpureE_4)
= 0

c_e_cons(477)_:
-1 B(Reaction_3_Reactor_1_4)
-1 B(Reaction_3_Reactor_2_4)
+1 B(Separation_Still_5)
-1 S(ImpureE_4)
+1 S(ImpureE_5)
= 0

c_e_cons(479)_:
-1 B(Reaction_3_Reactor_1_5)
-1 B(Reaction_3_Reactor_2_5)
+1 B(Separation_Still_6)
-1 S(ImpureE_5)
+1 S(ImpureE_6)
= 0

c_e_cons(481)_:
-1 B(Reaction_3_Reactor_1_6)
-1 B(Reaction_3_Reactor_2_6)
+1 B(Separation_Still_7)
-1 S(ImpureE_6)
+1 S(ImpureE_7)
= 0

c_e_cons(483)_:
-1 B(Reaction_3_Reactor_1_7)
-1 B(Reaction_3_Reactor_2_7)
+1 B(Separation_Still_8)
-1 S(ImpureE_7)
+1 S(ImpureE_8)
= 0

c_e_cons(485)_:
-1 B(Reaction_3_Reactor_1_8)
-1 B(Reaction_3_Reactor_2_8)
+1 B(Separation_Still_9)
-1 S(ImpureE_8)
+1 S(ImpureE_9)
= 0

c_e_cons(487)_:
-1 B(Reaction_3_Reactor_1_9)
-1 B(Reaction_3_Reactor_2_9)
+1 B(Separation_Still_10)
+1 S(ImpureE_10)
-1 S(ImpureE_9)
= 0

c_e_cons(489)_:
+0.40000000000000002 B(Reaction_2_Reactor_1_0)
+0.40000000000000002 B(Reaction_2_Reactor_2_0)
+1 S(HotA_0)
= 0

c_e_cons(491)_:
-1 B(Heating_Heater_0)
+0.40000000000000002 B(Reaction_2_Reactor_1_1)
+0.40000000000000002 B(Reaction_2_Reactor_2_1)
-1 S(HotA_0)
+1 S(HotA_1)
= 0

c_e_cons(493)_:
-1 B(Heating_Heater_1)
+0.40000000000000002 B(Reaction_2_Reactor_1_2)
+0.40000000000000002 B(Reaction_2_Reactor_2_2)
-1 S(HotA_1)
+1 S(HotA_2)
= 0

c_e_cons(495)_:
-1 B(Heating_Heater_2)
+0.40000000000000002 B(Reaction_2_Reactor_1_3)
+0.40000000000000002 B(Reaction_2_Reactor_2_3)
-1 S(HotA_2)
+1 S(HotA_3)
= 0

c_e_cons(497)_:
-1 B(Heating_Heater_3)
+0.40000000000000002 B(Reaction_2_Reactor_1_4)
+0.40000000000000002 B(Reaction_2_Reactor_2_4)
-1 S(HotA_3)
+1 S(HotA_4)
= 0

c_e_cons(499)_:
-1 B(Heating_Heater_4)
+0.40000000000000002 B(Reaction_2_Reactor_1_5)
+0.40000000000000002 B(Reaction_2_Reactor_2_5)
-1 S(HotA_4)
+1 S(HotA_5)
= 0

c_e_cons(501)_:
-1 B(Heating_Heater_5)
+0.40000000000000002 B(Reaction_2_Reactor_1_6)
+0.40000000000000002 B(Reaction_2_Reactor_2_6)
-1 S(HotA_5)
+1 S(HotA_6)
= 0

c_e_cons(503)_:
-1 B(Heating_Heater_6)
+0.40000000000000002 B(Reaction_2_Reactor_1_7)
+0.40000000000000002 B(Reaction_2_Reactor_2_7)
-1 S(HotA_6)
+1 S(HotA_7)
= 0

c_e_cons(505)_:
-1 B(Heating_Heater_7)
+0.40000000000000002 B(Reaction_2_Reactor_1_8)
+0.40000000000000002 B(Reaction_2_Reactor_2_8)
-1 S(HotA_7)
+1 S(HotA_8)
= 0

c_e_cons(507)_:
-1 B(Heating_Heater_8)
+0.40000000000000002 B(Reaction_2_Reactor_1_9)
+0.40000000000000002 B(Reaction_2_Reactor_2_9)
-1 S(HotA_8)
+1 S(HotA_9)
= 0

c_e_cons(509)_:
-1 B(Heating_Heater_9)
+0.40000000000000002 B(Reaction_2_Reactor_1_10)
+0.40000000000000002 B(Reaction_2_Reactor_2_10)
+1 S(HotA_10)
-1 S(HotA_9)
= 0

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= W(Heating_Heater_0) <= +inf
   0 <= W(Heating_Heater_1) <= +inf
   0 <= W(Heating_Heater_2) <= +inf
   0 <= W(Heating_Heater_3) <= +inf
   0 <= W(Heating_Heater_4) <= +inf
   0 <= W(Heating_Heater_5) <= +inf
   0 <= W(Heating_Heater_6) <= +inf
   0 <= W(Heating_Heater_7) <= +inf
   0 <= W(Heating_Heater_8) <= +inf
   0 <= W(Heating_Heater_9) <= +inf
   0 <= W(Heating_Heater_10) <= +inf
   0 <= W(Reaction_1_Reactor_1_0) <= +inf
   0 <= W(Reaction_1_Reactor_1_1) <= +inf
   0 <= W(Reaction_1_Reactor_1_2) <= +inf
   0 <= W(Reaction_1_Reactor_1_3) <= +inf
   0 <= W(Reaction_1_Reactor_1_4) <= +inf
   0 <= W(Reaction_1_Reactor_1_5) <= +inf
   0 <= W(Reaction_1_Reactor_1_6) <= +inf
   0 <= W(Reaction_1_Reactor_1_7) <= +inf
   0 <= W(Reaction_1_Reactor_1_8) <= +inf
   0 <= W(Reaction_1_Reactor_1_9) <= +inf
   0 <= W(Reaction_1_Reactor_1_10) <= +inf
   0 <= W(Reaction_1_Reactor_2_0) <= +inf
   0 <= W(Reaction_1_Reactor_2_1) <= +inf
   0 <= W(Reaction_1_Reactor_2_2) <= +inf
   0 <= W(Reaction_1_Reactor_2_3) <= +inf
   0 <= W(Reaction_1_Reactor_2_4) <= +inf
   0 <= W(Reaction_1_Reactor_2_5) <= +inf
   0 <= W(Reaction_1_Reactor_2_6) <= +inf
   0 <= W(Reaction_1_Reactor_2_7) <= +inf
   0 <= W(Reaction_1_Reactor_2_8) <= +inf
   0 <= W(Reaction_1_Reactor_2_9) <= +inf
   0 <= W(Reaction_1_Reactor_2_10) <= +inf
   0 <= W(Reaction_2_Reactor_1_0) <= +inf
   0 <= W(Reaction_2_Reactor_1_1) <= +inf
   0 <= W(Reaction_2_Reactor_1_2) <= +inf
   0 <= W(Reaction_2_Reactor_1_3) <= +inf
   0 <= W(Reaction_2_Reactor_1_4) <= +inf
   0 <= W(Reaction_2_Reactor_1_5) <= +inf
   0 <= W(Reaction_2_Reactor_1_6) <= +inf
   0 <= W(Reaction_2_Reactor_1_7) <= +inf
   0 <= W(Reaction_2_Reactor_1_8) <= +inf
   0 <= W(Reaction_2_Reactor_1_9) <= +inf
   0 <= W(Reaction_2_Reactor_1_10) <= +inf
   0 <= W(Reaction_2_Reactor_2_0) <= +inf
   0 <= W(Reaction_2_Reactor_2_1) <= +inf
   0 <= W(Reaction_2_Reactor_2_2) <= +inf
   0 <= W(Reaction_2_Reactor_2_3) <= +inf
   0 <= W(Reaction_2_Reactor_2_4) <= +inf
   0 <= W(Reaction_2_Reactor_2_5) <= +inf
   0 <= W(Reaction_2_Reactor_2_6) <= +inf
   0 <= W(Reaction_2_Reactor_2_7) <= +inf
   0 <= W(Reaction_2_Reactor_2_8) <= +inf
   0 <= W(Reaction_2_Reactor_2_9) <= +inf
   0 <= W(Reaction_2_Reactor_2_10) <= +inf
   0 <= W(Reaction_3_Reactor_1_0) <= +inf
   0 <= W(Reaction_3_Reactor_1_1) <= +inf
   0 <= W(Reaction_3_Reactor_1_2) <= +inf
   0 <= W(Reaction_3_Reactor_1_3) <= +inf
   0 <= W(Reaction_3_Reactor_1_4) <= +inf
   0 <= W(Reaction_3_Reactor_1_5) <= +inf
   0 <= W(Reaction_3_Reactor_1_6) <= +inf
   0 <= W(Reaction_3_Reactor_1_7) <= +inf
   0 <= W(Reaction_3_Reactor_1_8) <= +inf
   0 <= W(Reaction_3_Reactor_1_9) <= +inf
   0 <= W(Reaction_3_Reactor_1_10) <= +inf
   0 <= W(Reaction_3_Reactor_2_0) <= +inf
   0 <= W(Reaction_3_Reactor_2_1) <= +inf
   0 <= W(Reaction_3_Reactor_2_2) <= +inf
   0 <= W(Reaction_3_Reactor_2_3) <= +inf
   0 <= W(Reaction_3_Reactor_2_4) <= +inf
   0 <= W(Reaction_3_Reactor_2_5) <= +inf
   0 <= W(Reaction_3_Reactor_2_6) <= +inf
   0 <= W(Reaction_3_Reactor_2_7) <= +inf
   0 <= W(Reaction_3_Reactor_2_8) <= +inf
   0 <= W(Reaction_3_Reactor_2_9) <= +inf
   0 <= W(Reaction_3_Reactor_2_10) <= +inf
   0 <= W(Separation_Still_0) <= +inf
   0 <= W(Separation_Still_1) <= +inf
   0 <= W(Separation_Still_2) <= +inf
   0 <= W(Separation_Still_3) <= +inf
   0 <= W(Separation_Still_4) <= +inf
   0 <= W(Separation_Still_5) <= +inf
   0 <= W(Separation_Still_6) <= +inf
   0 <= W(Separation_Still_7) <= +inf
   0 <= W(Separation_Still_8) <= +inf
   0 <= W(Separation_Still_9) <= +inf
   0 <= W(Separation_Still_10) <= +inf
   0 <= B(Heating_Heater_0) <= +inf
   0 <= B(Heating_Heater_1) <= +inf
   0 <= B(Heating_Heater_2) <= +inf
   0 <= B(Heating_Heater_3) <= +inf
   0 <= B(Heating_Heater_4) <= +inf
   0 <= B(Heating_Heater_5) <= +inf
   0 <= B(Heating_Heater_6) <= +inf
   0 <= B(Heating_Heater_7) <= +inf
   0 <= B(Heating_Heater_8) <= +inf
   0 <= B(Heating_Heater_9) <= +inf
   0 <= B(Heating_Heater_10) <= +inf
   0 <= B(Reaction_1_Reactor_1_0) <= +inf
   0 <= B(Reaction_1_Reactor_1_1) <= +inf
   0 <= B(Reaction_1_Reactor_1_2) <= +inf
   0 <= B(Reaction_1_Reactor_1_3) <= +inf
   0 <= B(Reaction_1_Reactor_1_4) <= +inf
   0 <= B(Reaction_1_Reactor_1_5) <= +inf
   0 <= B(Reaction_1_Reactor_1_6) <= +inf
   0 <= B(Reaction_1_Reactor_1_7) <= +inf
   0 <= B(Reaction_1_Reactor_1_8) <= +inf
   0 <= B(Reaction_1_Reactor_1_9) <= +inf
   0 <= B(Reaction_1_Reactor_1_10) <= +inf
   0 <= B(Reaction_1_Reactor_2_0) <= +inf
   0 <= B(Reaction_1_Reactor_2_1) <= +inf
   0 <= B(Reaction_1_Reactor_2_2) <= +inf
   0 <= B(Reaction_1_Reactor_2_3) <= +inf
   0 <= B(Reaction_1_Reactor_2_4) <= +inf
   0 <= B(Reaction_1_Reactor_2_5) <= +inf
   0 <= B(Reaction_1_Reactor_2_6) <= +inf
   0 <= B(Reaction_1_Reactor_2_7) <= +inf
   0 <= B(Reaction_1_Reactor_2_8) <= +inf
   0 <= B(Reaction_1_Reactor_2_9) <= +inf
   0 <= B(Reaction_1_Reactor_2_10) <= +inf
   0 <= B(Reaction_2_Reactor_1_0) <= +inf
   0 <= B(Reaction_2_Reactor_1_1) <= +inf
   0 <= B(Reaction_2_Reactor_1_2) <= +inf
   0 <= B(Reaction_2_Reactor_1_3) <= +inf
   0 <= B(Reaction_2_Reactor_1_4) <= +inf
   0 <= B(Reaction_2_Reactor_1_5) <= +inf
   0 <= B(Reaction_2_Reactor_1_6) <= +inf
   0 <= B(Reaction_2_Reactor_1_7) <= +inf
   0 <= B(Reaction_2_Reactor_1_8) <= +inf
   0 <= B(Reaction_2_Reactor_1_9) <= +inf
   0 <= B(Reaction_2_Reactor_1_10) <= +inf
   0 <= B(Reaction_2_Reactor_2_0) <= +inf
   0 <= B(Reaction_2_Reactor_2_1) <= +inf
   0 <= B(Reaction_2_Reactor_2_2) <= +inf
   0 <= B(Reaction_2_Reactor_2_3) <= +inf
   0 <= B(Reaction_2_Reactor_2_4) <= +inf
   0 <= B(Reaction_2_Reactor_2_5) <= +inf
   0 <= B(Reaction_2_Reactor_2_6) <= +inf
   0 <= B(Reaction_2_Reactor_2_7) <= +inf
   0 <= B(Reaction_2_Reactor_2_8) <= +inf
   0 <= B(Reaction_2_Reactor_2_9) <= +inf
   0 <= B(Reaction_2_Reactor_2_10) <= +inf
   0 <= B(Reaction_3_Reactor_1_0) <= +inf
   0 <= B(Reaction_3_Reactor_1_1) <= +inf
   0 <= B(Reaction_3_Reactor_1_2) <= +inf
   0 <= B(Reaction_3_Reactor_1_3) <= +inf
   0 <= B(Reaction_3_Reactor_1_4) <= +inf
   0 <= B(Reaction_3_Reactor_1_5) <= +inf
   0 <= B(Reaction_3_Reactor_1_6) <= +inf
   0 <= B(Reaction_3_Reactor_1_7) <= +inf
   0 <= B(Reaction_3_Reactor_1_8) <= +inf
   0 <= B(Reaction_3_Reactor_1_9) <= +inf
   0 <= B(Reaction_3_Reactor_1_10) <= +inf
   0 <= B(Reaction_3_Reactor_2_0) <= +inf
   0 <= B(Reaction_3_Reactor_2_1) <= +inf
   0 <= B(Reaction_3_Reactor_2_2) <= +inf
   0 <= B(Reaction_3_Reactor_2_3) <= +inf
   0 <= B(Reaction_3_Reactor_2_4) <= +inf
   0 <= B(Reaction_3_Reactor_2_5) <= +inf
   0 <= B(Reaction_3_Reactor_2_6) <= +inf
   0 <= B(Reaction_3_Reactor_2_7) <= +inf
   0 <= B(Reaction_3_Reactor_2_8) <= +inf
   0 <= B(Reaction_3_Reactor_2_9) <= +inf
   0 <= B(Reaction_3_Reactor_2_10) <= +inf
   0 <= B(Separation_Still_0) <= +inf
   0 <= B(Separation_Still_1) <= +inf
   0 <= B(Separation_Still_2) <= +inf
   0 <= B(Separation_Still_3) <= +inf
   0 <= B(Separation_Still_4) <= +inf
   0 <= B(Separation_Still_5) <= +inf
   0 <= B(Separation_Still_6) <= +inf
   0 <= B(Separation_Still_7) <= +inf
   0 <= B(Separation_Still_8) <= +inf
   0 <= B(Separation_Still_9) <= +inf
   0 <= B(Separation_Still_10) <= +inf
   0 <= S(FeedA_0) <= +inf
   0 <= S(FeedA_1) <= +inf
   0 <= S(FeedA_2) <= +inf
   0 <= S(FeedA_3) <= +inf
   0 <= S(FeedA_4) <= +inf
   0 <= S(FeedA_5) <= +inf
   0 <= S(FeedA_6) <= +inf
   0 <= S(FeedA_7) <= +inf
   0 <= S(FeedA_8) <= +inf
   0 <= S(FeedA_9) <= +inf
   0 <= S(FeedA_10) <= +inf
   0 <= S(FeedB_0) <= +inf
   0 <= S(FeedB_1) <= +inf
   0 <= S(FeedB_2) <= +inf
   0 <= S(FeedB_3) <= +inf
   0 <= S(FeedB_4) <= +inf
   0 <= S(FeedB_5) <= +inf
   0 <= S(FeedB_6) <= +inf
   0 <= S(FeedB_7) <= +inf
   0 <= S(FeedB_8) <= +inf
   0 <= S(FeedB_9) <= +inf
   0 <= S(FeedB_10) <= +inf
   0 <= S(FeedC_0) <= +inf
   0 <= S(FeedC_1) <= +inf
   0 <= S(FeedC_2) <= +inf
   0 <= S(FeedC_3) <= +inf
   0 <= S(FeedC_4) <= +inf
   0 <= S(FeedC_5) <= +inf
   0 <= S(FeedC_6) <= +inf
   0 <= S(FeedC_7) <= +inf
   0 <= S(FeedC_8) <= +inf
   0 <= S(FeedC_9) <= +inf
   0 <= S(FeedC_10) <= +inf
   0 <= S(HotA_0) <= +inf
   0 <= S(HotA_1) <= +inf
   0 <= S(HotA_2) <= +inf
   0 <= S(HotA_3) <= +inf
   0 <= S(HotA_4) <= +inf
   0 <= S(HotA_5) <= +inf
   0 <= S(HotA_6) <= +inf
   0 <= S(HotA_7) <= +inf
   0 <= S(HotA_8) <= +inf
   0 <= S(HotA_9) <= +inf
   0 <= S(HotA_10) <= +inf
   0 <= S(ImpureE_0) <= +inf
   0 <= S(ImpureE_1) <= +inf
   0 <= S(ImpureE_2) <= +inf
   0 <= S(ImpureE_3) <= +inf
   0 <= S(ImpureE_4) <= +inf
   0 <= S(ImpureE_5) <= +inf
   0 <= S(ImpureE_6) <= +inf
   0 <= S(ImpureE_7) <= +inf
   0 <= S(ImpureE_8) <= +inf
   0 <= S(ImpureE_9) <= +inf
   0 <= S(ImpureE_10) <= +inf
   0 <= S(IntAB_0) <= +inf
   0 <= S(IntAB_1) <= +inf
   0 <= S(IntAB_2) <= +inf
   0 <= S(IntAB_3) <= +inf
   0 <= S(IntAB_4) <= +inf
   0 <= S(IntAB_5) <= +inf
   0 <= S(IntAB_6) <= +inf
   0 <= S(IntAB_7) <= +inf
   0 <= S(IntAB_8) <= +inf
   0 <= S(IntAB_9) <= +inf
   0 <= S(IntAB_10) <= +inf
   0 <= S(IntBC_0) <= +inf
   0 <= S(IntBC_1) <= +inf
   0 <= S(IntBC_2) <= +inf
   0 <= S(IntBC_3) <= +inf
   0 <= S(IntBC_4) <= +inf
   0 <= S(IntBC_5) <= +inf
   0 <= S(IntBC_6) <= +inf
   0 <= S(IntBC_7) <= +inf
   0 <= S(IntBC_8) <= +inf
   0 <= S(IntBC_9) <= +inf
   0 <= S(IntBC_10) <= +inf
   0 <= S(Product_1_0) <= +inf
   0 <= S(Product_1_1) <= +inf
   0 <= S(Product_1_2) <= +inf
   0 <= S(Product_1_3) <= +inf
   0 <= S(Product_1_4) <= +inf
   0 <= S(Product_1_5) <= +inf
   0 <= S(Product_1_6) <= +inf
   0 <= S(Product_1_7) <= +inf
   0 <= S(Product_1_8) <= +inf
   0 <= S(Product_1_9) <= +inf
   0 <= S(Product_1_10) <= +inf
   0 <= S(Product_2_0) <= +inf
   0 <= S(Product_2_1) <= +inf
   0 <= S(Product_2_2) <= +inf
   0 <= S(Product_2_3) <= +inf
   0 <= S(Product_2_4) <= +inf
   0 <= S(Product_2_5) <= +inf
   0 <= S(Product_2_6) <= +inf
   0 <= S(Product_2_7) <= +inf
   0 <= S(Product_2_8) <= +inf
   0 <= S(Product_2_9) <= +inf
   0 <= S(Product_2_10) <= +inf
   0 <= Q(Heater_0) <= +inf
   0 <= Q(Heater_1) <= +inf
   0 <= Q(Heater_2) <= +inf
   0 <= Q(Heater_3) <= +inf
   0 <= Q(Heater_4) <= +inf
   0 <= Q(Heater_5) <= +inf
   0 <= Q(Heater_6) <= +inf
   0 <= Q(Heater_7) <= +inf
   0 <= Q(Heater_8) <= +inf
   0 <= Q(Heater_9) <= +inf
   0 <= Q(Heater_10) <= +inf
   0 <= Q(Reactor_1_0) <= +inf
   0 <= Q(Reactor_1_1) <= +inf
   0 <= Q(Reactor_1_2) <= +inf
   0 <= Q(Reactor_1_3) <= +inf
   0 <= Q(Reactor_1_4) <= +inf
   0 <= Q(Reactor_1_5) <= +inf
   0 <= Q(Reactor_1_6) <= +inf
   0 <= Q(Reactor_1_7) <= +inf
   0 <= Q(Reactor_1_8) <= +inf
   0 <= Q(Reactor_1_9) <= +inf
   0 <= Q(Reactor_1_10) <= +inf
   0 <= Q(Reactor_2_0) <= +inf
   0 <= Q(Reactor_2_1) <= +inf
   0 <= Q(Reactor_2_2) <= +inf
   0 <= Q(Reactor_2_3) <= +inf
   0 <= Q(Reactor_2_4) <= +inf
   0 <= Q(Reactor_2_5) <= +inf
   0 <= Q(Reactor_2_6) <= +inf
   0 <= Q(Reactor_2_7) <= +inf
   0 <= Q(Reactor_2_8) <= +inf
   0 <= Q(Reactor_2_9) <= +inf
   0 <= Q(Reactor_2_10) <= +inf
   0 <= Q(Still_0) <= +inf
   0 <= Q(Still_1) <= +inf
   0 <= Q(Still_2) <= +inf
   0 <= Q(Still_3) <= +inf
   0 <= Q(Still_4) <= +inf
   0 <= Q(Still_5) <= +inf
   0 <= Q(Still_6) <= +inf
   0 <= Q(Still_7) <= +inf
   0 <= Q(Still_8) <= +inf
   0 <= Q(Still_9) <= +inf
   0 <= Q(Still_10) <= +inf
   0 <= Cost <= +inf
   0 <= StorageCost <= +inf
   0 <= Value <= +inf
end
