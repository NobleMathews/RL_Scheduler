\* Source Pyomo model name=unknown *\

max 
Obj:
-1 Cost
-1 StorageCost
+1 Value

s.t.

c_e_cons(Cost)_:
+1 Cost
= 0

c_e_cons(StorageCost)_:
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

c_e_cons(Value)_:
+1 S(HotA_10)
+1 S(ImpureE_10)
+1 S(IntAB_10)
+1 S(IntBC_10)
-10 S(Product_1_10)
-10 S(Product_2_10)
+1 Value
= 0

c_l_cons(capacity_constraint1_Heater_time_0_Heating)_:
+1 B(Heating_Heater_0)
>= 0

c_l_cons(capacity_constraint1_Heater_time_10_Heating)_:
+1 B(Heating_Heater_10)
>= 0

c_l_cons(capacity_constraint1_Heater_time_1_Heating)_:
+1 B(Heating_Heater_1)
>= 0

c_l_cons(capacity_constraint1_Heater_time_2_Heating)_:
+1 B(Heating_Heater_2)
>= 0

c_l_cons(capacity_constraint1_Heater_time_3_Heating)_:
+1 B(Heating_Heater_3)
>= 0

c_l_cons(capacity_constraint1_Heater_time_4_Heating)_:
+1 B(Heating_Heater_4)
>= 0

c_l_cons(capacity_constraint1_Heater_time_5_Heating)_:
+1 B(Heating_Heater_5)
>= 0

c_l_cons(capacity_constraint1_Heater_time_6_Heating)_:
+1 B(Heating_Heater_6)
>= 0

c_l_cons(capacity_constraint1_Heater_time_7_Heating)_:
+1 B(Heating_Heater_7)
>= 0

c_l_cons(capacity_constraint1_Heater_time_8_Heating)_:
+1 B(Heating_Heater_8)
>= 0

c_l_cons(capacity_constraint1_Heater_time_9_Heating)_:
+1 B(Heating_Heater_9)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_0_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_0)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_0_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_0)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_0_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_0)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_10_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_10)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_10_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_10)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_10_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_10)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_1_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_1)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_1_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_1)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_1_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_1)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_2_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_2)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_2_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_2)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_2_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_2)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_3_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_3)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_3_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_3)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_3_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_3)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_4_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_4)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_4_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_4)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_4_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_4)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_5_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_5)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_5_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_5)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_5_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_5)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_6_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_6)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_6_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_6)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_6_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_6)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_7_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_7)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_7_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_7)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_7_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_7)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_8_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_8)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_8_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_8)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_8_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_8)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_9_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_9)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_9_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_9)
>= 0

c_l_cons(capacity_constraint1_Reactor_1_time_9_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_9)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_0_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_0)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_0_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_0)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_0_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_0)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_10_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_10)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_10_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_10)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_10_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_10)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_1_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_1)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_1_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_1)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_1_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_1)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_2_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_2)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_2_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_2)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_2_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_2)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_3_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_3)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_3_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_3)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_3_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_3)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_4_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_4)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_4_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_4)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_4_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_4)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_5_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_5)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_5_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_5)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_5_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_5)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_6_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_6)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_6_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_6)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_6_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_6)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_7_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_7)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_7_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_7)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_7_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_7)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_8_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_8)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_8_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_8)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_8_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_8)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_9_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_9)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_9_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_9)
>= 0

c_l_cons(capacity_constraint1_Reactor_2_time_9_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_9)
>= 0

c_l_cons(capacity_constraint1_Still_time_0_Separation)_:
+1 B(Separation_Still_0)
>= 0

c_l_cons(capacity_constraint1_Still_time_10_Separation)_:
+1 B(Separation_Still_10)
>= 0

c_l_cons(capacity_constraint1_Still_time_1_Separation)_:
+1 B(Separation_Still_1)
>= 0

c_l_cons(capacity_constraint1_Still_time_2_Separation)_:
+1 B(Separation_Still_2)
>= 0

c_l_cons(capacity_constraint1_Still_time_3_Separation)_:
+1 B(Separation_Still_3)
>= 0

c_l_cons(capacity_constraint1_Still_time_4_Separation)_:
+1 B(Separation_Still_4)
>= 0

c_l_cons(capacity_constraint1_Still_time_5_Separation)_:
+1 B(Separation_Still_5)
>= 0

c_l_cons(capacity_constraint1_Still_time_6_Separation)_:
+1 B(Separation_Still_6)
>= 0

c_l_cons(capacity_constraint1_Still_time_7_Separation)_:
+1 B(Separation_Still_7)
>= 0

c_l_cons(capacity_constraint1_Still_time_8_Separation)_:
+1 B(Separation_Still_8)
>= 0

c_l_cons(capacity_constraint1_Still_time_9_Separation)_:
+1 B(Separation_Still_9)
>= 0

c_u_cons(capacity_constraint2_Heater_time_0_Heating)_:
+1 B(Heating_Heater_0)
-100 W(Heating_Heater_0)
<= 0

c_u_cons(capacity_constraint2_Heater_time_10_Heating)_:
+1 B(Heating_Heater_10)
-100 W(Heating_Heater_10)
<= 0

c_u_cons(capacity_constraint2_Heater_time_1_Heating)_:
+1 B(Heating_Heater_1)
-100 W(Heating_Heater_1)
<= 0

c_u_cons(capacity_constraint2_Heater_time_2_Heating)_:
+1 B(Heating_Heater_2)
-100 W(Heating_Heater_2)
<= 0

c_u_cons(capacity_constraint2_Heater_time_3_Heating)_:
+1 B(Heating_Heater_3)
-100 W(Heating_Heater_3)
<= 0

c_u_cons(capacity_constraint2_Heater_time_4_Heating)_:
+1 B(Heating_Heater_4)
-100 W(Heating_Heater_4)
<= 0

c_u_cons(capacity_constraint2_Heater_time_5_Heating)_:
+1 B(Heating_Heater_5)
-100 W(Heating_Heater_5)
<= 0

c_u_cons(capacity_constraint2_Heater_time_6_Heating)_:
+1 B(Heating_Heater_6)
-100 W(Heating_Heater_6)
<= 0

c_u_cons(capacity_constraint2_Heater_time_7_Heating)_:
+1 B(Heating_Heater_7)
-100 W(Heating_Heater_7)
<= 0

c_u_cons(capacity_constraint2_Heater_time_8_Heating)_:
+1 B(Heating_Heater_8)
-100 W(Heating_Heater_8)
<= 0

c_u_cons(capacity_constraint2_Heater_time_9_Heating)_:
+1 B(Heating_Heater_9)
-100 W(Heating_Heater_9)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_0_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_0)
-80 W(Reaction_1_Reactor_1_0)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_0_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_0)
-80 W(Reaction_2_Reactor_1_0)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_0_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_0)
-80 W(Reaction_3_Reactor_1_0)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_10_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_10)
-80 W(Reaction_1_Reactor_1_10)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_10_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_10)
-80 W(Reaction_2_Reactor_1_10)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_10_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_10)
-80 W(Reaction_3_Reactor_1_10)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_1_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_1)
-80 W(Reaction_1_Reactor_1_1)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_1_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_1)
-80 W(Reaction_2_Reactor_1_1)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_1_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_1)
-80 W(Reaction_3_Reactor_1_1)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_2_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_2)
-80 W(Reaction_1_Reactor_1_2)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_2_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_2)
-80 W(Reaction_2_Reactor_1_2)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_2_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_2)
-80 W(Reaction_3_Reactor_1_2)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_3_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_3)
-80 W(Reaction_1_Reactor_1_3)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_3_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_3)
-80 W(Reaction_2_Reactor_1_3)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_3_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_3)
-80 W(Reaction_3_Reactor_1_3)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_4_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_4)
-80 W(Reaction_1_Reactor_1_4)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_4_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_4)
-80 W(Reaction_2_Reactor_1_4)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_4_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_4)
-80 W(Reaction_3_Reactor_1_4)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_5_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_5)
-80 W(Reaction_1_Reactor_1_5)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_5_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_5)
-80 W(Reaction_2_Reactor_1_5)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_5_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_5)
-80 W(Reaction_3_Reactor_1_5)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_6_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_6)
-80 W(Reaction_1_Reactor_1_6)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_6_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_6)
-80 W(Reaction_2_Reactor_1_6)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_6_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_6)
-80 W(Reaction_3_Reactor_1_6)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_7_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_7)
-80 W(Reaction_1_Reactor_1_7)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_7_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_7)
-80 W(Reaction_2_Reactor_1_7)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_7_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_7)
-80 W(Reaction_3_Reactor_1_7)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_8_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_8)
-80 W(Reaction_1_Reactor_1_8)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_8_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_8)
-80 W(Reaction_2_Reactor_1_8)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_8_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_8)
-80 W(Reaction_3_Reactor_1_8)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_9_Reaction_1)_:
+1 B(Reaction_1_Reactor_1_9)
-80 W(Reaction_1_Reactor_1_9)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_9_Reaction_2)_:
+1 B(Reaction_2_Reactor_1_9)
-80 W(Reaction_2_Reactor_1_9)
<= 0

c_u_cons(capacity_constraint2_Reactor_1_time_9_Reaction_3)_:
+1 B(Reaction_3_Reactor_1_9)
-80 W(Reaction_3_Reactor_1_9)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_0_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_0)
-50 W(Reaction_1_Reactor_2_0)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_0_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_0)
-50 W(Reaction_2_Reactor_2_0)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_0_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_0)
-50 W(Reaction_3_Reactor_2_0)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_10_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_10)
-50 W(Reaction_1_Reactor_2_10)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_10_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_10)
-50 W(Reaction_2_Reactor_2_10)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_10_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_10)
-50 W(Reaction_3_Reactor_2_10)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_1_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_1)
-50 W(Reaction_1_Reactor_2_1)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_1_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_1)
-50 W(Reaction_2_Reactor_2_1)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_1_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_1)
-50 W(Reaction_3_Reactor_2_1)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_2_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_2)
-50 W(Reaction_1_Reactor_2_2)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_2_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_2)
-50 W(Reaction_2_Reactor_2_2)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_2_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_2)
-50 W(Reaction_3_Reactor_2_2)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_3_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_3)
-50 W(Reaction_1_Reactor_2_3)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_3_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_3)
-50 W(Reaction_2_Reactor_2_3)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_3_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_3)
-50 W(Reaction_3_Reactor_2_3)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_4_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_4)
-50 W(Reaction_1_Reactor_2_4)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_4_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_4)
-50 W(Reaction_2_Reactor_2_4)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_4_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_4)
-50 W(Reaction_3_Reactor_2_4)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_5_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_5)
-50 W(Reaction_1_Reactor_2_5)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_5_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_5)
-50 W(Reaction_2_Reactor_2_5)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_5_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_5)
-50 W(Reaction_3_Reactor_2_5)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_6_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_6)
-50 W(Reaction_1_Reactor_2_6)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_6_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_6)
-50 W(Reaction_2_Reactor_2_6)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_6_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_6)
-50 W(Reaction_3_Reactor_2_6)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_7_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_7)
-50 W(Reaction_1_Reactor_2_7)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_7_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_7)
-50 W(Reaction_2_Reactor_2_7)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_7_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_7)
-50 W(Reaction_3_Reactor_2_7)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_8_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_8)
-50 W(Reaction_1_Reactor_2_8)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_8_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_8)
-50 W(Reaction_2_Reactor_2_8)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_8_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_8)
-50 W(Reaction_3_Reactor_2_8)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_9_Reaction_1)_:
+1 B(Reaction_1_Reactor_2_9)
-50 W(Reaction_1_Reactor_2_9)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_9_Reaction_2)_:
+1 B(Reaction_2_Reactor_2_9)
-50 W(Reaction_2_Reactor_2_9)
<= 0

c_u_cons(capacity_constraint2_Reactor_2_time_9_Reaction_3)_:
+1 B(Reaction_3_Reactor_2_9)
-50 W(Reaction_3_Reactor_2_9)
<= 0

c_u_cons(capacity_constraint2_Still_time_0_Separation)_:
+1 B(Separation_Still_0)
-200 W(Separation_Still_0)
<= 0

c_u_cons(capacity_constraint2_Still_time_10_Separation)_:
+1 B(Separation_Still_10)
-200 W(Separation_Still_10)
<= 0

c_u_cons(capacity_constraint2_Still_time_1_Separation)_:
+1 B(Separation_Still_1)
-200 W(Separation_Still_1)
<= 0

c_u_cons(capacity_constraint2_Still_time_2_Separation)_:
+1 B(Separation_Still_2)
-200 W(Separation_Still_2)
<= 0

c_u_cons(capacity_constraint2_Still_time_3_Separation)_:
+1 B(Separation_Still_3)
-200 W(Separation_Still_3)
<= 0

c_u_cons(capacity_constraint2_Still_time_4_Separation)_:
+1 B(Separation_Still_4)
-200 W(Separation_Still_4)
<= 0

c_u_cons(capacity_constraint2_Still_time_5_Separation)_:
+1 B(Separation_Still_5)
-200 W(Separation_Still_5)
<= 0

c_u_cons(capacity_constraint2_Still_time_6_Separation)_:
+1 B(Separation_Still_6)
-200 W(Separation_Still_6)
<= 0

c_u_cons(capacity_constraint2_Still_time_7_Separation)_:
+1 B(Separation_Still_7)
-200 W(Separation_Still_7)
<= 0

c_u_cons(capacity_constraint2_Still_time_8_Separation)_:
+1 B(Separation_Still_8)
-200 W(Separation_Still_8)
<= 0

c_u_cons(capacity_constraint2_Still_time_9_Separation)_:
+1 B(Separation_Still_9)
-200 W(Separation_Still_9)
<= 0

c_e_cons(state_FeedA_mass_balance_time_0)_:
+1 B(Heating_Heater_0)
+1 S(FeedA_0)
= 200

c_e_cons(state_FeedA_mass_balance_time_1)_:
+1 B(Heating_Heater_1)
-1 S(FeedA_0)
+1 S(FeedA_1)
= 0

c_e_cons(state_FeedA_mass_balance_time_10)_:
+1 B(Heating_Heater_10)
+1 S(FeedA_10)
-1 S(FeedA_9)
= 0

c_e_cons(state_FeedA_mass_balance_time_2)_:
+1 B(Heating_Heater_2)
-1 S(FeedA_1)
+1 S(FeedA_2)
= 0

c_e_cons(state_FeedA_mass_balance_time_3)_:
+1 B(Heating_Heater_3)
-1 S(FeedA_2)
+1 S(FeedA_3)
= 0

c_e_cons(state_FeedA_mass_balance_time_4)_:
+1 B(Heating_Heater_4)
-1 S(FeedA_3)
+1 S(FeedA_4)
= 0

c_e_cons(state_FeedA_mass_balance_time_5)_:
+1 B(Heating_Heater_5)
-1 S(FeedA_4)
+1 S(FeedA_5)
= 0

c_e_cons(state_FeedA_mass_balance_time_6)_:
+1 B(Heating_Heater_6)
-1 S(FeedA_5)
+1 S(FeedA_6)
= 0

c_e_cons(state_FeedA_mass_balance_time_7)_:
+1 B(Heating_Heater_7)
-1 S(FeedA_6)
+1 S(FeedA_7)
= 0

c_e_cons(state_FeedA_mass_balance_time_8)_:
+1 B(Heating_Heater_8)
-1 S(FeedA_7)
+1 S(FeedA_8)
= 0

c_e_cons(state_FeedA_mass_balance_time_9)_:
+1 B(Heating_Heater_9)
-1 S(FeedA_8)
+1 S(FeedA_9)
= 0

c_e_cons(state_FeedB_mass_balance_time_0)_:
+0.5 B(Reaction_1_Reactor_1_0)
+0.5 B(Reaction_1_Reactor_2_0)
+1 S(FeedB_0)
= 200

c_e_cons(state_FeedB_mass_balance_time_1)_:
+0.5 B(Reaction_1_Reactor_1_1)
+0.5 B(Reaction_1_Reactor_2_1)
-1 S(FeedB_0)
+1 S(FeedB_1)
= 0

c_e_cons(state_FeedB_mass_balance_time_10)_:
+0.5 B(Reaction_1_Reactor_1_10)
+0.5 B(Reaction_1_Reactor_2_10)
+1 S(FeedB_10)
-1 S(FeedB_9)
= 0

c_e_cons(state_FeedB_mass_balance_time_2)_:
+0.5 B(Reaction_1_Reactor_1_2)
+0.5 B(Reaction_1_Reactor_2_2)
-1 S(FeedB_1)
+1 S(FeedB_2)
= 0

c_e_cons(state_FeedB_mass_balance_time_3)_:
+0.5 B(Reaction_1_Reactor_1_3)
+0.5 B(Reaction_1_Reactor_2_3)
-1 S(FeedB_2)
+1 S(FeedB_3)
= 0

c_e_cons(state_FeedB_mass_balance_time_4)_:
+0.5 B(Reaction_1_Reactor_1_4)
+0.5 B(Reaction_1_Reactor_2_4)
-1 S(FeedB_3)
+1 S(FeedB_4)
= 0

c_e_cons(state_FeedB_mass_balance_time_5)_:
+0.5 B(Reaction_1_Reactor_1_5)
+0.5 B(Reaction_1_Reactor_2_5)
-1 S(FeedB_4)
+1 S(FeedB_5)
= 0

c_e_cons(state_FeedB_mass_balance_time_6)_:
+0.5 B(Reaction_1_Reactor_1_6)
+0.5 B(Reaction_1_Reactor_2_6)
-1 S(FeedB_5)
+1 S(FeedB_6)
= 0

c_e_cons(state_FeedB_mass_balance_time_7)_:
+0.5 B(Reaction_1_Reactor_1_7)
+0.5 B(Reaction_1_Reactor_2_7)
-1 S(FeedB_6)
+1 S(FeedB_7)
= 0

c_e_cons(state_FeedB_mass_balance_time_8)_:
+0.5 B(Reaction_1_Reactor_1_8)
+0.5 B(Reaction_1_Reactor_2_8)
-1 S(FeedB_7)
+1 S(FeedB_8)
= 0

c_e_cons(state_FeedB_mass_balance_time_9)_:
+0.5 B(Reaction_1_Reactor_1_9)
+0.5 B(Reaction_1_Reactor_2_9)
-1 S(FeedB_8)
+1 S(FeedB_9)
= 0

c_e_cons(state_FeedC_mass_balance_time_0)_:
+0.5 B(Reaction_1_Reactor_1_0)
+0.5 B(Reaction_1_Reactor_2_0)
+0.20000000000000001 B(Reaction_3_Reactor_1_0)
+0.20000000000000001 B(Reaction_3_Reactor_2_0)
+1 S(FeedC_0)
= 200

c_e_cons(state_FeedC_mass_balance_time_1)_:
+0.5 B(Reaction_1_Reactor_1_1)
+0.5 B(Reaction_1_Reactor_2_1)
+0.20000000000000001 B(Reaction_3_Reactor_1_1)
+0.20000000000000001 B(Reaction_3_Reactor_2_1)
-1 S(FeedC_0)
+1 S(FeedC_1)
= 0

c_e_cons(state_FeedC_mass_balance_time_10)_:
+0.5 B(Reaction_1_Reactor_1_10)
+0.5 B(Reaction_1_Reactor_2_10)
+0.20000000000000001 B(Reaction_3_Reactor_1_10)
+0.20000000000000001 B(Reaction_3_Reactor_2_10)
+1 S(FeedC_10)
-1 S(FeedC_9)
= 0

c_e_cons(state_FeedC_mass_balance_time_2)_:
+0.5 B(Reaction_1_Reactor_1_2)
+0.5 B(Reaction_1_Reactor_2_2)
+0.20000000000000001 B(Reaction_3_Reactor_1_2)
+0.20000000000000001 B(Reaction_3_Reactor_2_2)
-1 S(FeedC_1)
+1 S(FeedC_2)
= 0

c_e_cons(state_FeedC_mass_balance_time_3)_:
+0.5 B(Reaction_1_Reactor_1_3)
+0.5 B(Reaction_1_Reactor_2_3)
+0.20000000000000001 B(Reaction_3_Reactor_1_3)
+0.20000000000000001 B(Reaction_3_Reactor_2_3)
-1 S(FeedC_2)
+1 S(FeedC_3)
= 0

c_e_cons(state_FeedC_mass_balance_time_4)_:
+0.5 B(Reaction_1_Reactor_1_4)
+0.5 B(Reaction_1_Reactor_2_4)
+0.20000000000000001 B(Reaction_3_Reactor_1_4)
+0.20000000000000001 B(Reaction_3_Reactor_2_4)
-1 S(FeedC_3)
+1 S(FeedC_4)
= 0

c_e_cons(state_FeedC_mass_balance_time_5)_:
+0.5 B(Reaction_1_Reactor_1_5)
+0.5 B(Reaction_1_Reactor_2_5)
+0.20000000000000001 B(Reaction_3_Reactor_1_5)
+0.20000000000000001 B(Reaction_3_Reactor_2_5)
-1 S(FeedC_4)
+1 S(FeedC_5)
= 0

c_e_cons(state_FeedC_mass_balance_time_6)_:
+0.5 B(Reaction_1_Reactor_1_6)
+0.5 B(Reaction_1_Reactor_2_6)
+0.20000000000000001 B(Reaction_3_Reactor_1_6)
+0.20000000000000001 B(Reaction_3_Reactor_2_6)
-1 S(FeedC_5)
+1 S(FeedC_6)
= 0

c_e_cons(state_FeedC_mass_balance_time_7)_:
+0.5 B(Reaction_1_Reactor_1_7)
+0.5 B(Reaction_1_Reactor_2_7)
+0.20000000000000001 B(Reaction_3_Reactor_1_7)
+0.20000000000000001 B(Reaction_3_Reactor_2_7)
-1 S(FeedC_6)
+1 S(FeedC_7)
= 0

c_e_cons(state_FeedC_mass_balance_time_8)_:
+0.5 B(Reaction_1_Reactor_1_8)
+0.5 B(Reaction_1_Reactor_2_8)
+0.20000000000000001 B(Reaction_3_Reactor_1_8)
+0.20000000000000001 B(Reaction_3_Reactor_2_8)
-1 S(FeedC_7)
+1 S(FeedC_8)
= 0

c_e_cons(state_FeedC_mass_balance_time_9)_:
+0.5 B(Reaction_1_Reactor_1_9)
+0.5 B(Reaction_1_Reactor_2_9)
+0.20000000000000001 B(Reaction_3_Reactor_1_9)
+0.20000000000000001 B(Reaction_3_Reactor_2_9)
-1 S(FeedC_8)
+1 S(FeedC_9)
= 0

c_e_cons(state_HotA_mass_balance_time_0)_:
+0.40000000000000002 B(Reaction_2_Reactor_1_0)
+0.40000000000000002 B(Reaction_2_Reactor_2_0)
+1 S(HotA_0)
= 0

c_e_cons(state_HotA_mass_balance_time_1)_:
-1 B(Heating_Heater_0)
+0.40000000000000002 B(Reaction_2_Reactor_1_1)
+0.40000000000000002 B(Reaction_2_Reactor_2_1)
-1 S(HotA_0)
+1 S(HotA_1)
= 0

c_e_cons(state_HotA_mass_balance_time_10)_:
-1 B(Heating_Heater_9)
+0.40000000000000002 B(Reaction_2_Reactor_1_10)
+0.40000000000000002 B(Reaction_2_Reactor_2_10)
+1 S(HotA_10)
-1 S(HotA_9)
= 0

c_e_cons(state_HotA_mass_balance_time_2)_:
-1 B(Heating_Heater_1)
+0.40000000000000002 B(Reaction_2_Reactor_1_2)
+0.40000000000000002 B(Reaction_2_Reactor_2_2)
-1 S(HotA_1)
+1 S(HotA_2)
= 0

c_e_cons(state_HotA_mass_balance_time_3)_:
-1 B(Heating_Heater_2)
+0.40000000000000002 B(Reaction_2_Reactor_1_3)
+0.40000000000000002 B(Reaction_2_Reactor_2_3)
-1 S(HotA_2)
+1 S(HotA_3)
= 0

c_e_cons(state_HotA_mass_balance_time_4)_:
-1 B(Heating_Heater_3)
+0.40000000000000002 B(Reaction_2_Reactor_1_4)
+0.40000000000000002 B(Reaction_2_Reactor_2_4)
-1 S(HotA_3)
+1 S(HotA_4)
= 0

c_e_cons(state_HotA_mass_balance_time_5)_:
-1 B(Heating_Heater_4)
+0.40000000000000002 B(Reaction_2_Reactor_1_5)
+0.40000000000000002 B(Reaction_2_Reactor_2_5)
-1 S(HotA_4)
+1 S(HotA_5)
= 0

c_e_cons(state_HotA_mass_balance_time_6)_:
-1 B(Heating_Heater_5)
+0.40000000000000002 B(Reaction_2_Reactor_1_6)
+0.40000000000000002 B(Reaction_2_Reactor_2_6)
-1 S(HotA_5)
+1 S(HotA_6)
= 0

c_e_cons(state_HotA_mass_balance_time_7)_:
-1 B(Heating_Heater_6)
+0.40000000000000002 B(Reaction_2_Reactor_1_7)
+0.40000000000000002 B(Reaction_2_Reactor_2_7)
-1 S(HotA_6)
+1 S(HotA_7)
= 0

c_e_cons(state_HotA_mass_balance_time_8)_:
-1 B(Heating_Heater_7)
+0.40000000000000002 B(Reaction_2_Reactor_1_8)
+0.40000000000000002 B(Reaction_2_Reactor_2_8)
-1 S(HotA_7)
+1 S(HotA_8)
= 0

c_e_cons(state_HotA_mass_balance_time_9)_:
-1 B(Heating_Heater_8)
+0.40000000000000002 B(Reaction_2_Reactor_1_9)
+0.40000000000000002 B(Reaction_2_Reactor_2_9)
-1 S(HotA_8)
+1 S(HotA_9)
= 0

c_e_cons(state_ImpureE_mass_balance_time_0)_:
+1 B(Separation_Still_0)
+1 S(ImpureE_0)
= 0

c_e_cons(state_ImpureE_mass_balance_time_1)_:
-1 B(Reaction_3_Reactor_1_0)
-1 B(Reaction_3_Reactor_2_0)
+1 B(Separation_Still_1)
-1 S(ImpureE_0)
+1 S(ImpureE_1)
= 0

c_e_cons(state_ImpureE_mass_balance_time_10)_:
-1 B(Reaction_3_Reactor_1_9)
-1 B(Reaction_3_Reactor_2_9)
+1 B(Separation_Still_10)
+1 S(ImpureE_10)
-1 S(ImpureE_9)
= 0

c_e_cons(state_ImpureE_mass_balance_time_2)_:
-1 B(Reaction_3_Reactor_1_1)
-1 B(Reaction_3_Reactor_2_1)
+1 B(Separation_Still_2)
-1 S(ImpureE_1)
+1 S(ImpureE_2)
= 0

c_e_cons(state_ImpureE_mass_balance_time_3)_:
-1 B(Reaction_3_Reactor_1_2)
-1 B(Reaction_3_Reactor_2_2)
+1 B(Separation_Still_3)
-1 S(ImpureE_2)
+1 S(ImpureE_3)
= 0

c_e_cons(state_ImpureE_mass_balance_time_4)_:
-1 B(Reaction_3_Reactor_1_3)
-1 B(Reaction_3_Reactor_2_3)
+1 B(Separation_Still_4)
-1 S(ImpureE_3)
+1 S(ImpureE_4)
= 0

c_e_cons(state_ImpureE_mass_balance_time_5)_:
-1 B(Reaction_3_Reactor_1_4)
-1 B(Reaction_3_Reactor_2_4)
+1 B(Separation_Still_5)
-1 S(ImpureE_4)
+1 S(ImpureE_5)
= 0

c_e_cons(state_ImpureE_mass_balance_time_6)_:
-1 B(Reaction_3_Reactor_1_5)
-1 B(Reaction_3_Reactor_2_5)
+1 B(Separation_Still_6)
-1 S(ImpureE_5)
+1 S(ImpureE_6)
= 0

c_e_cons(state_ImpureE_mass_balance_time_7)_:
-1 B(Reaction_3_Reactor_1_6)
-1 B(Reaction_3_Reactor_2_6)
+1 B(Separation_Still_7)
-1 S(ImpureE_6)
+1 S(ImpureE_7)
= 0

c_e_cons(state_ImpureE_mass_balance_time_8)_:
-1 B(Reaction_3_Reactor_1_7)
-1 B(Reaction_3_Reactor_2_7)
+1 B(Separation_Still_8)
-1 S(ImpureE_7)
+1 S(ImpureE_8)
= 0

c_e_cons(state_ImpureE_mass_balance_time_9)_:
-1 B(Reaction_3_Reactor_1_8)
-1 B(Reaction_3_Reactor_2_8)
+1 B(Separation_Still_9)
-1 S(ImpureE_8)
+1 S(ImpureE_9)
= 0

c_e_cons(state_IntAB_mass_balance_time_0)_:
+0.80000000000000004 B(Reaction_3_Reactor_1_0)
+0.80000000000000004 B(Reaction_3_Reactor_2_0)
+1 S(IntAB_0)
= 0

c_e_cons(state_IntAB_mass_balance_time_1)_:
+0.80000000000000004 B(Reaction_3_Reactor_1_1)
+0.80000000000000004 B(Reaction_3_Reactor_2_1)
-1 S(IntAB_0)
+1 S(IntAB_1)
= 0

c_e_cons(state_IntAB_mass_balance_time_10)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_8)
-0.59999999999999998 B(Reaction_2_Reactor_2_8)
+0.80000000000000004 B(Reaction_3_Reactor_1_10)
+0.80000000000000004 B(Reaction_3_Reactor_2_10)
-0.10000000000000001 B(Separation_Still_8)
+1 S(IntAB_10)
-1 S(IntAB_9)
= 0

c_e_cons(state_IntAB_mass_balance_time_2)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_0)
-0.59999999999999998 B(Reaction_2_Reactor_2_0)
+0.80000000000000004 B(Reaction_3_Reactor_1_2)
+0.80000000000000004 B(Reaction_3_Reactor_2_2)
-0.10000000000000001 B(Separation_Still_0)
-1 S(IntAB_1)
+1 S(IntAB_2)
= 0

c_e_cons(state_IntAB_mass_balance_time_3)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_1)
-0.59999999999999998 B(Reaction_2_Reactor_2_1)
+0.80000000000000004 B(Reaction_3_Reactor_1_3)
+0.80000000000000004 B(Reaction_3_Reactor_2_3)
-0.10000000000000001 B(Separation_Still_1)
-1 S(IntAB_2)
+1 S(IntAB_3)
= 0

c_e_cons(state_IntAB_mass_balance_time_4)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_2)
-0.59999999999999998 B(Reaction_2_Reactor_2_2)
+0.80000000000000004 B(Reaction_3_Reactor_1_4)
+0.80000000000000004 B(Reaction_3_Reactor_2_4)
-0.10000000000000001 B(Separation_Still_2)
-1 S(IntAB_3)
+1 S(IntAB_4)
= 0

c_e_cons(state_IntAB_mass_balance_time_5)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_3)
-0.59999999999999998 B(Reaction_2_Reactor_2_3)
+0.80000000000000004 B(Reaction_3_Reactor_1_5)
+0.80000000000000004 B(Reaction_3_Reactor_2_5)
-0.10000000000000001 B(Separation_Still_3)
-1 S(IntAB_4)
+1 S(IntAB_5)
= 0

c_e_cons(state_IntAB_mass_balance_time_6)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_4)
-0.59999999999999998 B(Reaction_2_Reactor_2_4)
+0.80000000000000004 B(Reaction_3_Reactor_1_6)
+0.80000000000000004 B(Reaction_3_Reactor_2_6)
-0.10000000000000001 B(Separation_Still_4)
-1 S(IntAB_5)
+1 S(IntAB_6)
= 0

c_e_cons(state_IntAB_mass_balance_time_7)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_5)
-0.59999999999999998 B(Reaction_2_Reactor_2_5)
+0.80000000000000004 B(Reaction_3_Reactor_1_7)
+0.80000000000000004 B(Reaction_3_Reactor_2_7)
-0.10000000000000001 B(Separation_Still_5)
-1 S(IntAB_6)
+1 S(IntAB_7)
= 0

c_e_cons(state_IntAB_mass_balance_time_8)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_6)
-0.59999999999999998 B(Reaction_2_Reactor_2_6)
+0.80000000000000004 B(Reaction_3_Reactor_1_8)
+0.80000000000000004 B(Reaction_3_Reactor_2_8)
-0.10000000000000001 B(Separation_Still_6)
-1 S(IntAB_7)
+1 S(IntAB_8)
= 0

c_e_cons(state_IntAB_mass_balance_time_9)_:
-0.59999999999999998 B(Reaction_2_Reactor_1_7)
-0.59999999999999998 B(Reaction_2_Reactor_2_7)
+0.80000000000000004 B(Reaction_3_Reactor_1_9)
+0.80000000000000004 B(Reaction_3_Reactor_2_9)
-0.10000000000000001 B(Separation_Still_7)
-1 S(IntAB_8)
+1 S(IntAB_9)
= 0

c_e_cons(state_IntBC_mass_balance_time_0)_:
+0.59999999999999998 B(Reaction_2_Reactor_1_0)
+0.59999999999999998 B(Reaction_2_Reactor_2_0)
+1 S(IntBC_0)
= 0

c_e_cons(state_IntBC_mass_balance_time_1)_:
+0.59999999999999998 B(Reaction_2_Reactor_1_1)
+0.59999999999999998 B(Reaction_2_Reactor_2_1)
-1 S(IntBC_0)
+1 S(IntBC_1)
= 0

c_e_cons(state_IntBC_mass_balance_time_10)_:
-1 B(Reaction_1_Reactor_1_8)
-1 B(Reaction_1_Reactor_2_8)
+0.59999999999999998 B(Reaction_2_Reactor_1_10)
+0.59999999999999998 B(Reaction_2_Reactor_2_10)
+1 S(IntBC_10)
-1 S(IntBC_9)
= 0

c_e_cons(state_IntBC_mass_balance_time_2)_:
-1 B(Reaction_1_Reactor_1_0)
-1 B(Reaction_1_Reactor_2_0)
+0.59999999999999998 B(Reaction_2_Reactor_1_2)
+0.59999999999999998 B(Reaction_2_Reactor_2_2)
-1 S(IntBC_1)
+1 S(IntBC_2)
= 0

c_e_cons(state_IntBC_mass_balance_time_3)_:
-1 B(Reaction_1_Reactor_1_1)
-1 B(Reaction_1_Reactor_2_1)
+0.59999999999999998 B(Reaction_2_Reactor_1_3)
+0.59999999999999998 B(Reaction_2_Reactor_2_3)
-1 S(IntBC_2)
+1 S(IntBC_3)
= 0

c_e_cons(state_IntBC_mass_balance_time_4)_:
-1 B(Reaction_1_Reactor_1_2)
-1 B(Reaction_1_Reactor_2_2)
+0.59999999999999998 B(Reaction_2_Reactor_1_4)
+0.59999999999999998 B(Reaction_2_Reactor_2_4)
-1 S(IntBC_3)
+1 S(IntBC_4)
= 0

c_e_cons(state_IntBC_mass_balance_time_5)_:
-1 B(Reaction_1_Reactor_1_3)
-1 B(Reaction_1_Reactor_2_3)
+0.59999999999999998 B(Reaction_2_Reactor_1_5)
+0.59999999999999998 B(Reaction_2_Reactor_2_5)
-1 S(IntBC_4)
+1 S(IntBC_5)
= 0

c_e_cons(state_IntBC_mass_balance_time_6)_:
-1 B(Reaction_1_Reactor_1_4)
-1 B(Reaction_1_Reactor_2_4)
+0.59999999999999998 B(Reaction_2_Reactor_1_6)
+0.59999999999999998 B(Reaction_2_Reactor_2_6)
-1 S(IntBC_5)
+1 S(IntBC_6)
= 0

c_e_cons(state_IntBC_mass_balance_time_7)_:
-1 B(Reaction_1_Reactor_1_5)
-1 B(Reaction_1_Reactor_2_5)
+0.59999999999999998 B(Reaction_2_Reactor_1_7)
+0.59999999999999998 B(Reaction_2_Reactor_2_7)
-1 S(IntBC_6)
+1 S(IntBC_7)
= 0

c_e_cons(state_IntBC_mass_balance_time_8)_:
-1 B(Reaction_1_Reactor_1_6)
-1 B(Reaction_1_Reactor_2_6)
+0.59999999999999998 B(Reaction_2_Reactor_1_8)
+0.59999999999999998 B(Reaction_2_Reactor_2_8)
-1 S(IntBC_7)
+1 S(IntBC_8)
= 0

c_e_cons(state_IntBC_mass_balance_time_9)_:
-1 B(Reaction_1_Reactor_1_7)
-1 B(Reaction_1_Reactor_2_7)
+0.59999999999999998 B(Reaction_2_Reactor_1_9)
+0.59999999999999998 B(Reaction_2_Reactor_2_9)
-1 S(IntBC_8)
+1 S(IntBC_9)
= 0

c_e_cons(state_Product_1_mass_balance_time_0)_:
+1 S(Product_1_0)
= 0

c_e_cons(state_Product_1_mass_balance_time_1)_:
-1 S(Product_1_0)
+1 S(Product_1_1)
= 0

c_e_cons(state_Product_1_mass_balance_time_10)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_8)
-0.40000000000000002 B(Reaction_2_Reactor_2_8)
+1 S(Product_1_10)
-1 S(Product_1_9)
= 0

c_e_cons(state_Product_1_mass_balance_time_2)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_0)
-0.40000000000000002 B(Reaction_2_Reactor_2_0)
-1 S(Product_1_1)
+1 S(Product_1_2)
= 0

c_e_cons(state_Product_1_mass_balance_time_3)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_1)
-0.40000000000000002 B(Reaction_2_Reactor_2_1)
-1 S(Product_1_2)
+1 S(Product_1_3)
= 0

c_e_cons(state_Product_1_mass_balance_time_4)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_2)
-0.40000000000000002 B(Reaction_2_Reactor_2_2)
-1 S(Product_1_3)
+1 S(Product_1_4)
= 0

c_e_cons(state_Product_1_mass_balance_time_5)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_3)
-0.40000000000000002 B(Reaction_2_Reactor_2_3)
-1 S(Product_1_4)
+1 S(Product_1_5)
= 0

c_e_cons(state_Product_1_mass_balance_time_6)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_4)
-0.40000000000000002 B(Reaction_2_Reactor_2_4)
-1 S(Product_1_5)
+1 S(Product_1_6)
= 0

c_e_cons(state_Product_1_mass_balance_time_7)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_5)
-0.40000000000000002 B(Reaction_2_Reactor_2_5)
-1 S(Product_1_6)
+1 S(Product_1_7)
= 0

c_e_cons(state_Product_1_mass_balance_time_8)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_6)
-0.40000000000000002 B(Reaction_2_Reactor_2_6)
-1 S(Product_1_7)
+1 S(Product_1_8)
= 0

c_e_cons(state_Product_1_mass_balance_time_9)_:
-0.40000000000000002 B(Reaction_2_Reactor_1_7)
-0.40000000000000002 B(Reaction_2_Reactor_2_7)
-1 S(Product_1_8)
+1 S(Product_1_9)
= 0

c_e_cons(state_Product_2_mass_balance_time_0)_:
+1 S(Product_2_0)
= 0

c_e_cons(state_Product_2_mass_balance_time_1)_:
-0.90000000000000002 B(Separation_Still_0)
-1 S(Product_2_0)
+1 S(Product_2_1)
= 0

c_e_cons(state_Product_2_mass_balance_time_10)_:
-0.90000000000000002 B(Separation_Still_9)
+1 S(Product_2_10)
-1 S(Product_2_9)
= 0

c_e_cons(state_Product_2_mass_balance_time_2)_:
-0.90000000000000002 B(Separation_Still_1)
-1 S(Product_2_1)
+1 S(Product_2_2)
= 0

c_e_cons(state_Product_2_mass_balance_time_3)_:
-0.90000000000000002 B(Separation_Still_2)
-1 S(Product_2_2)
+1 S(Product_2_3)
= 0

c_e_cons(state_Product_2_mass_balance_time_4)_:
-0.90000000000000002 B(Separation_Still_3)
-1 S(Product_2_3)
+1 S(Product_2_4)
= 0

c_e_cons(state_Product_2_mass_balance_time_5)_:
-0.90000000000000002 B(Separation_Still_4)
-1 S(Product_2_4)
+1 S(Product_2_5)
= 0

c_e_cons(state_Product_2_mass_balance_time_6)_:
-0.90000000000000002 B(Separation_Still_5)
-1 S(Product_2_5)
+1 S(Product_2_6)
= 0

c_e_cons(state_Product_2_mass_balance_time_7)_:
-0.90000000000000002 B(Separation_Still_6)
-1 S(Product_2_6)
+1 S(Product_2_7)
= 0

c_e_cons(state_Product_2_mass_balance_time_8)_:
-0.90000000000000002 B(Separation_Still_7)
-1 S(Product_2_7)
+1 S(Product_2_8)
= 0

c_e_cons(state_Product_2_mass_balance_time_9)_:
-0.90000000000000002 B(Separation_Still_8)
-1 S(Product_2_8)
+1 S(Product_2_9)
= 0

c_e_cons(terminal_condition_Heater_time_0)_:
+1 Q(Heater_10)
= 0

c_e_cons(terminal_condition_Heater_time_1)_:
+1 Q(Heater_10)
= 0

c_e_cons(terminal_condition_Heater_time_10)_:
+1 Q(Heater_10)
= 0

c_e_cons(terminal_condition_Heater_time_2)_:
+1 Q(Heater_10)
= 0

c_e_cons(terminal_condition_Heater_time_3)_:
+1 Q(Heater_10)
= 0

c_e_cons(terminal_condition_Heater_time_4)_:
+1 Q(Heater_10)
= 0

c_e_cons(terminal_condition_Heater_time_5)_:
+1 Q(Heater_10)
= 0

c_e_cons(terminal_condition_Heater_time_6)_:
+1 Q(Heater_10)
= 0

c_e_cons(terminal_condition_Heater_time_7)_:
+1 Q(Heater_10)
= 0

c_e_cons(terminal_condition_Heater_time_8)_:
+1 Q(Heater_10)
= 0

c_e_cons(terminal_condition_Heater_time_9)_:
+1 Q(Heater_10)
= 0

c_e_cons(terminal_condition_Reactor_1_time_0)_:
+1 Q(Reactor_1_10)
= 0

c_e_cons(terminal_condition_Reactor_1_time_1)_:
+1 Q(Reactor_1_10)
= 0

c_e_cons(terminal_condition_Reactor_1_time_10)_:
+1 Q(Reactor_1_10)
= 0

c_e_cons(terminal_condition_Reactor_1_time_2)_:
+1 Q(Reactor_1_10)
= 0

c_e_cons(terminal_condition_Reactor_1_time_3)_:
+1 Q(Reactor_1_10)
= 0

c_e_cons(terminal_condition_Reactor_1_time_4)_:
+1 Q(Reactor_1_10)
= 0

c_e_cons(terminal_condition_Reactor_1_time_5)_:
+1 Q(Reactor_1_10)
= 0

c_e_cons(terminal_condition_Reactor_1_time_6)_:
+1 Q(Reactor_1_10)
= 0

c_e_cons(terminal_condition_Reactor_1_time_7)_:
+1 Q(Reactor_1_10)
= 0

c_e_cons(terminal_condition_Reactor_1_time_8)_:
+1 Q(Reactor_1_10)
= 0

c_e_cons(terminal_condition_Reactor_1_time_9)_:
+1 Q(Reactor_1_10)
= 0

c_e_cons(terminal_condition_Reactor_2_time_0)_:
+1 Q(Reactor_2_10)
= 0

c_e_cons(terminal_condition_Reactor_2_time_1)_:
+1 Q(Reactor_2_10)
= 0

c_e_cons(terminal_condition_Reactor_2_time_10)_:
+1 Q(Reactor_2_10)
= 0

c_e_cons(terminal_condition_Reactor_2_time_2)_:
+1 Q(Reactor_2_10)
= 0

c_e_cons(terminal_condition_Reactor_2_time_3)_:
+1 Q(Reactor_2_10)
= 0

c_e_cons(terminal_condition_Reactor_2_time_4)_:
+1 Q(Reactor_2_10)
= 0

c_e_cons(terminal_condition_Reactor_2_time_5)_:
+1 Q(Reactor_2_10)
= 0

c_e_cons(terminal_condition_Reactor_2_time_6)_:
+1 Q(Reactor_2_10)
= 0

c_e_cons(terminal_condition_Reactor_2_time_7)_:
+1 Q(Reactor_2_10)
= 0

c_e_cons(terminal_condition_Reactor_2_time_8)_:
+1 Q(Reactor_2_10)
= 0

c_e_cons(terminal_condition_Reactor_2_time_9)_:
+1 Q(Reactor_2_10)
= 0

c_e_cons(terminal_condition_Still_time_0)_:
+1 Q(Still_10)
= 0

c_e_cons(terminal_condition_Still_time_1)_:
+1 Q(Still_10)
= 0

c_e_cons(terminal_condition_Still_time_10)_:
+1 Q(Still_10)
= 0

c_e_cons(terminal_condition_Still_time_2)_:
+1 Q(Still_10)
= 0

c_e_cons(terminal_condition_Still_time_3)_:
+1 Q(Still_10)
= 0

c_e_cons(terminal_condition_Still_time_4)_:
+1 Q(Still_10)
= 0

c_e_cons(terminal_condition_Still_time_5)_:
+1 Q(Still_10)
= 0

c_e_cons(terminal_condition_Still_time_6)_:
+1 Q(Still_10)
= 0

c_e_cons(terminal_condition_Still_time_7)_:
+1 Q(Still_10)
= 0

c_e_cons(terminal_condition_Still_time_8)_:
+1 Q(Still_10)
= 0

c_e_cons(terminal_condition_Still_time_9)_:
+1 Q(Still_10)
= 0

c_u_cons(unit_constraint_Heater_time_0)_:
+1 W(Heating_Heater_0)
<= 1

c_u_cons(unit_constraint_Heater_time_1)_:
+1 W(Heating_Heater_1)
<= 1

c_u_cons(unit_constraint_Heater_time_10)_:
+1 W(Heating_Heater_10)
<= 1

c_u_cons(unit_constraint_Heater_time_2)_:
+1 W(Heating_Heater_2)
<= 1

c_u_cons(unit_constraint_Heater_time_3)_:
+1 W(Heating_Heater_3)
<= 1

c_u_cons(unit_constraint_Heater_time_4)_:
+1 W(Heating_Heater_4)
<= 1

c_u_cons(unit_constraint_Heater_time_5)_:
+1 W(Heating_Heater_5)
<= 1

c_u_cons(unit_constraint_Heater_time_6)_:
+1 W(Heating_Heater_6)
<= 1

c_u_cons(unit_constraint_Heater_time_7)_:
+1 W(Heating_Heater_7)
<= 1

c_u_cons(unit_constraint_Heater_time_8)_:
+1 W(Heating_Heater_8)
<= 1

c_u_cons(unit_constraint_Heater_time_9)_:
+1 W(Heating_Heater_9)
<= 1

c_u_cons(unit_constraint_Reactor_1_time_0)_:
+1 W(Reaction_1_Reactor_1_0)
+1 W(Reaction_2_Reactor_1_0)
+1 W(Reaction_3_Reactor_1_0)
<= 1

c_u_cons(unit_constraint_Reactor_1_time_1)_:
+1 W(Reaction_1_Reactor_1_0)
+1 W(Reaction_1_Reactor_1_1)
+1 W(Reaction_2_Reactor_1_0)
+1 W(Reaction_2_Reactor_1_1)
+1 W(Reaction_3_Reactor_1_1)
<= 1

c_u_cons(unit_constraint_Reactor_1_time_10)_:
+1 W(Reaction_1_Reactor_1_10)
+1 W(Reaction_1_Reactor_1_9)
+1 W(Reaction_2_Reactor_1_10)
+1 W(Reaction_2_Reactor_1_9)
+1 W(Reaction_3_Reactor_1_10)
<= 1

c_u_cons(unit_constraint_Reactor_1_time_2)_:
+1 W(Reaction_1_Reactor_1_1)
+1 W(Reaction_1_Reactor_1_2)
+1 W(Reaction_2_Reactor_1_1)
+1 W(Reaction_2_Reactor_1_2)
+1 W(Reaction_3_Reactor_1_2)
<= 1

c_u_cons(unit_constraint_Reactor_1_time_3)_:
+1 W(Reaction_1_Reactor_1_2)
+1 W(Reaction_1_Reactor_1_3)
+1 W(Reaction_2_Reactor_1_2)
+1 W(Reaction_2_Reactor_1_3)
+1 W(Reaction_3_Reactor_1_3)
<= 1

c_u_cons(unit_constraint_Reactor_1_time_4)_:
+1 W(Reaction_1_Reactor_1_3)
+1 W(Reaction_1_Reactor_1_4)
+1 W(Reaction_2_Reactor_1_3)
+1 W(Reaction_2_Reactor_1_4)
+1 W(Reaction_3_Reactor_1_4)
<= 1

c_u_cons(unit_constraint_Reactor_1_time_5)_:
+1 W(Reaction_1_Reactor_1_4)
+1 W(Reaction_1_Reactor_1_5)
+1 W(Reaction_2_Reactor_1_4)
+1 W(Reaction_2_Reactor_1_5)
+1 W(Reaction_3_Reactor_1_5)
<= 1

c_u_cons(unit_constraint_Reactor_1_time_6)_:
+1 W(Reaction_1_Reactor_1_5)
+1 W(Reaction_1_Reactor_1_6)
+1 W(Reaction_2_Reactor_1_5)
+1 W(Reaction_2_Reactor_1_6)
+1 W(Reaction_3_Reactor_1_6)
<= 1

c_u_cons(unit_constraint_Reactor_1_time_7)_:
+1 W(Reaction_1_Reactor_1_6)
+1 W(Reaction_1_Reactor_1_7)
+1 W(Reaction_2_Reactor_1_6)
+1 W(Reaction_2_Reactor_1_7)
+1 W(Reaction_3_Reactor_1_7)
<= 1

c_u_cons(unit_constraint_Reactor_1_time_8)_:
+1 W(Reaction_1_Reactor_1_7)
+1 W(Reaction_1_Reactor_1_8)
+1 W(Reaction_2_Reactor_1_7)
+1 W(Reaction_2_Reactor_1_8)
+1 W(Reaction_3_Reactor_1_8)
<= 1

c_u_cons(unit_constraint_Reactor_1_time_9)_:
+1 W(Reaction_1_Reactor_1_8)
+1 W(Reaction_1_Reactor_1_9)
+1 W(Reaction_2_Reactor_1_8)
+1 W(Reaction_2_Reactor_1_9)
+1 W(Reaction_3_Reactor_1_9)
<= 1

c_u_cons(unit_constraint_Reactor_2_time_0)_:
+1 W(Reaction_1_Reactor_2_0)
+1 W(Reaction_2_Reactor_2_0)
+1 W(Reaction_3_Reactor_2_0)
<= 1

c_u_cons(unit_constraint_Reactor_2_time_1)_:
+1 W(Reaction_1_Reactor_2_0)
+1 W(Reaction_1_Reactor_2_1)
+1 W(Reaction_2_Reactor_2_0)
+1 W(Reaction_2_Reactor_2_1)
+1 W(Reaction_3_Reactor_2_1)
<= 1

c_u_cons(unit_constraint_Reactor_2_time_10)_:
+1 W(Reaction_1_Reactor_2_10)
+1 W(Reaction_1_Reactor_2_9)
+1 W(Reaction_2_Reactor_2_10)
+1 W(Reaction_2_Reactor_2_9)
+1 W(Reaction_3_Reactor_2_10)
<= 1

c_u_cons(unit_constraint_Reactor_2_time_2)_:
+1 W(Reaction_1_Reactor_2_1)
+1 W(Reaction_1_Reactor_2_2)
+1 W(Reaction_2_Reactor_2_1)
+1 W(Reaction_2_Reactor_2_2)
+1 W(Reaction_3_Reactor_2_2)
<= 1

c_u_cons(unit_constraint_Reactor_2_time_3)_:
+1 W(Reaction_1_Reactor_2_2)
+1 W(Reaction_1_Reactor_2_3)
+1 W(Reaction_2_Reactor_2_2)
+1 W(Reaction_2_Reactor_2_3)
+1 W(Reaction_3_Reactor_2_3)
<= 1

c_u_cons(unit_constraint_Reactor_2_time_4)_:
+1 W(Reaction_1_Reactor_2_3)
+1 W(Reaction_1_Reactor_2_4)
+1 W(Reaction_2_Reactor_2_3)
+1 W(Reaction_2_Reactor_2_4)
+1 W(Reaction_3_Reactor_2_4)
<= 1

c_u_cons(unit_constraint_Reactor_2_time_5)_:
+1 W(Reaction_1_Reactor_2_4)
+1 W(Reaction_1_Reactor_2_5)
+1 W(Reaction_2_Reactor_2_4)
+1 W(Reaction_2_Reactor_2_5)
+1 W(Reaction_3_Reactor_2_5)
<= 1

c_u_cons(unit_constraint_Reactor_2_time_6)_:
+1 W(Reaction_1_Reactor_2_5)
+1 W(Reaction_1_Reactor_2_6)
+1 W(Reaction_2_Reactor_2_5)
+1 W(Reaction_2_Reactor_2_6)
+1 W(Reaction_3_Reactor_2_6)
<= 1

c_u_cons(unit_constraint_Reactor_2_time_7)_:
+1 W(Reaction_1_Reactor_2_6)
+1 W(Reaction_1_Reactor_2_7)
+1 W(Reaction_2_Reactor_2_6)
+1 W(Reaction_2_Reactor_2_7)
+1 W(Reaction_3_Reactor_2_7)
<= 1

c_u_cons(unit_constraint_Reactor_2_time_8)_:
+1 W(Reaction_1_Reactor_2_7)
+1 W(Reaction_1_Reactor_2_8)
+1 W(Reaction_2_Reactor_2_7)
+1 W(Reaction_2_Reactor_2_8)
+1 W(Reaction_3_Reactor_2_8)
<= 1

c_u_cons(unit_constraint_Reactor_2_time_9)_:
+1 W(Reaction_1_Reactor_2_8)
+1 W(Reaction_1_Reactor_2_9)
+1 W(Reaction_2_Reactor_2_8)
+1 W(Reaction_2_Reactor_2_9)
+1 W(Reaction_3_Reactor_2_9)
<= 1

c_u_cons(unit_constraint_Still_time_0)_:
+1 W(Separation_Still_0)
<= 1

c_u_cons(unit_constraint_Still_time_1)_:
+1 W(Separation_Still_0)
+1 W(Separation_Still_1)
<= 1

c_u_cons(unit_constraint_Still_time_10)_:
+1 W(Separation_Still_10)
+1 W(Separation_Still_9)
<= 1

c_u_cons(unit_constraint_Still_time_2)_:
+1 W(Separation_Still_1)
+1 W(Separation_Still_2)
<= 1

c_u_cons(unit_constraint_Still_time_3)_:
+1 W(Separation_Still_2)
+1 W(Separation_Still_3)
<= 1

c_u_cons(unit_constraint_Still_time_4)_:
+1 W(Separation_Still_3)
+1 W(Separation_Still_4)
<= 1

c_u_cons(unit_constraint_Still_time_5)_:
+1 W(Separation_Still_4)
+1 W(Separation_Still_5)
<= 1

c_u_cons(unit_constraint_Still_time_6)_:
+1 W(Separation_Still_5)
+1 W(Separation_Still_6)
<= 1

c_u_cons(unit_constraint_Still_time_7)_:
+1 W(Separation_Still_6)
+1 W(Separation_Still_7)
<= 1

c_u_cons(unit_constraint_Still_time_8)_:
+1 W(Separation_Still_7)
+1 W(Separation_Still_8)
<= 1

c_u_cons(unit_constraint_Still_time_9)_:
+1 W(Separation_Still_8)
+1 W(Separation_Still_9)
<= 1

c_e_cons(unit_mass_balance_Heater_time_0)_:
-1 B(Heating_Heater_0)
+1 Q(Heater_0)
= 0

c_e_cons(unit_mass_balance_Heater_time_1)_:
+1 B(Heating_Heater_0)
-1 B(Heating_Heater_1)
-1 Q(Heater_0)
+1 Q(Heater_1)
= 0

c_e_cons(unit_mass_balance_Heater_time_10)_:
-1 B(Heating_Heater_10)
+1 B(Heating_Heater_9)
+1 Q(Heater_10)
-1 Q(Heater_9)
= 0

c_e_cons(unit_mass_balance_Heater_time_2)_:
+1 B(Heating_Heater_1)
-1 B(Heating_Heater_2)
-1 Q(Heater_1)
+1 Q(Heater_2)
= 0

c_e_cons(unit_mass_balance_Heater_time_3)_:
+1 B(Heating_Heater_2)
-1 B(Heating_Heater_3)
-1 Q(Heater_2)
+1 Q(Heater_3)
= 0

c_e_cons(unit_mass_balance_Heater_time_4)_:
+1 B(Heating_Heater_3)
-1 B(Heating_Heater_4)
-1 Q(Heater_3)
+1 Q(Heater_4)
= 0

c_e_cons(unit_mass_balance_Heater_time_5)_:
+1 B(Heating_Heater_4)
-1 B(Heating_Heater_5)
-1 Q(Heater_4)
+1 Q(Heater_5)
= 0

c_e_cons(unit_mass_balance_Heater_time_6)_:
+1 B(Heating_Heater_5)
-1 B(Heating_Heater_6)
-1 Q(Heater_5)
+1 Q(Heater_6)
= 0

c_e_cons(unit_mass_balance_Heater_time_7)_:
+1 B(Heating_Heater_6)
-1 B(Heating_Heater_7)
-1 Q(Heater_6)
+1 Q(Heater_7)
= 0

c_e_cons(unit_mass_balance_Heater_time_8)_:
+1 B(Heating_Heater_7)
-1 B(Heating_Heater_8)
-1 Q(Heater_7)
+1 Q(Heater_8)
= 0

c_e_cons(unit_mass_balance_Heater_time_9)_:
+1 B(Heating_Heater_8)
-1 B(Heating_Heater_9)
-1 Q(Heater_8)
+1 Q(Heater_9)
= 0

c_e_cons(unit_mass_balance_Reactor_1_time_0)_:
-1 B(Reaction_1_Reactor_1_0)
-1 B(Reaction_2_Reactor_1_0)
-1 B(Reaction_3_Reactor_1_0)
+1 Q(Reactor_1_0)
= 0

c_e_cons(unit_mass_balance_Reactor_1_time_1)_:
-1 B(Reaction_1_Reactor_1_1)
-1 B(Reaction_2_Reactor_1_1)
+1 B(Reaction_3_Reactor_1_0)
-1 B(Reaction_3_Reactor_1_1)
-1 Q(Reactor_1_0)
+1 Q(Reactor_1_1)
= 0

c_e_cons(unit_mass_balance_Reactor_1_time_10)_:
-1 B(Reaction_1_Reactor_1_10)
+1 B(Reaction_1_Reactor_1_8)
-1 B(Reaction_2_Reactor_1_10)
+1 B(Reaction_2_Reactor_1_8)
-1 B(Reaction_3_Reactor_1_10)
+1 B(Reaction_3_Reactor_1_9)
+1 Q(Reactor_1_10)
-1 Q(Reactor_1_9)
= 0

c_e_cons(unit_mass_balance_Reactor_1_time_2)_:
+1 B(Reaction_1_Reactor_1_0)
-1 B(Reaction_1_Reactor_1_2)
+1 B(Reaction_2_Reactor_1_0)
-1 B(Reaction_2_Reactor_1_2)
+1 B(Reaction_3_Reactor_1_1)
-1 B(Reaction_3_Reactor_1_2)
-1 Q(Reactor_1_1)
+1 Q(Reactor_1_2)
= 0

c_e_cons(unit_mass_balance_Reactor_1_time_3)_:
+1 B(Reaction_1_Reactor_1_1)
-1 B(Reaction_1_Reactor_1_3)
+1 B(Reaction_2_Reactor_1_1)
-1 B(Reaction_2_Reactor_1_3)
+1 B(Reaction_3_Reactor_1_2)
-1 B(Reaction_3_Reactor_1_3)
-1 Q(Reactor_1_2)
+1 Q(Reactor_1_3)
= 0

c_e_cons(unit_mass_balance_Reactor_1_time_4)_:
+1 B(Reaction_1_Reactor_1_2)
-1 B(Reaction_1_Reactor_1_4)
+1 B(Reaction_2_Reactor_1_2)
-1 B(Reaction_2_Reactor_1_4)
+1 B(Reaction_3_Reactor_1_3)
-1 B(Reaction_3_Reactor_1_4)
-1 Q(Reactor_1_3)
+1 Q(Reactor_1_4)
= 0

c_e_cons(unit_mass_balance_Reactor_1_time_5)_:
+1 B(Reaction_1_Reactor_1_3)
-1 B(Reaction_1_Reactor_1_5)
+1 B(Reaction_2_Reactor_1_3)
-1 B(Reaction_2_Reactor_1_5)
+1 B(Reaction_3_Reactor_1_4)
-1 B(Reaction_3_Reactor_1_5)
-1 Q(Reactor_1_4)
+1 Q(Reactor_1_5)
= 0

c_e_cons(unit_mass_balance_Reactor_1_time_6)_:
+1 B(Reaction_1_Reactor_1_4)
-1 B(Reaction_1_Reactor_1_6)
+1 B(Reaction_2_Reactor_1_4)
-1 B(Reaction_2_Reactor_1_6)
+1 B(Reaction_3_Reactor_1_5)
-1 B(Reaction_3_Reactor_1_6)
-1 Q(Reactor_1_5)
+1 Q(Reactor_1_6)
= 0

c_e_cons(unit_mass_balance_Reactor_1_time_7)_:
+1 B(Reaction_1_Reactor_1_5)
-1 B(Reaction_1_Reactor_1_7)
+1 B(Reaction_2_Reactor_1_5)
-1 B(Reaction_2_Reactor_1_7)
+1 B(Reaction_3_Reactor_1_6)
-1 B(Reaction_3_Reactor_1_7)
-1 Q(Reactor_1_6)
+1 Q(Reactor_1_7)
= 0

c_e_cons(unit_mass_balance_Reactor_1_time_8)_:
+1 B(Reaction_1_Reactor_1_6)
-1 B(Reaction_1_Reactor_1_8)
+1 B(Reaction_2_Reactor_1_6)
-1 B(Reaction_2_Reactor_1_8)
+1 B(Reaction_3_Reactor_1_7)
-1 B(Reaction_3_Reactor_1_8)
-1 Q(Reactor_1_7)
+1 Q(Reactor_1_8)
= 0

c_e_cons(unit_mass_balance_Reactor_1_time_9)_:
+1 B(Reaction_1_Reactor_1_7)
-1 B(Reaction_1_Reactor_1_9)
+1 B(Reaction_2_Reactor_1_7)
-1 B(Reaction_2_Reactor_1_9)
+1 B(Reaction_3_Reactor_1_8)
-1 B(Reaction_3_Reactor_1_9)
-1 Q(Reactor_1_8)
+1 Q(Reactor_1_9)
= 0

c_e_cons(unit_mass_balance_Reactor_2_time_0)_:
-1 B(Reaction_1_Reactor_2_0)
-1 B(Reaction_2_Reactor_2_0)
-1 B(Reaction_3_Reactor_2_0)
+1 Q(Reactor_2_0)
= 0

c_e_cons(unit_mass_balance_Reactor_2_time_1)_:
-1 B(Reaction_1_Reactor_2_1)
-1 B(Reaction_2_Reactor_2_1)
+1 B(Reaction_3_Reactor_2_0)
-1 B(Reaction_3_Reactor_2_1)
-1 Q(Reactor_2_0)
+1 Q(Reactor_2_1)
= 0

c_e_cons(unit_mass_balance_Reactor_2_time_10)_:
-1 B(Reaction_1_Reactor_2_10)
+1 B(Reaction_1_Reactor_2_8)
-1 B(Reaction_2_Reactor_2_10)
+1 B(Reaction_2_Reactor_2_8)
-1 B(Reaction_3_Reactor_2_10)
+1 B(Reaction_3_Reactor_2_9)
+1 Q(Reactor_2_10)
-1 Q(Reactor_2_9)
= 0

c_e_cons(unit_mass_balance_Reactor_2_time_2)_:
+1 B(Reaction_1_Reactor_2_0)
-1 B(Reaction_1_Reactor_2_2)
+1 B(Reaction_2_Reactor_2_0)
-1 B(Reaction_2_Reactor_2_2)
+1 B(Reaction_3_Reactor_2_1)
-1 B(Reaction_3_Reactor_2_2)
-1 Q(Reactor_2_1)
+1 Q(Reactor_2_2)
= 0

c_e_cons(unit_mass_balance_Reactor_2_time_3)_:
+1 B(Reaction_1_Reactor_2_1)
-1 B(Reaction_1_Reactor_2_3)
+1 B(Reaction_2_Reactor_2_1)
-1 B(Reaction_2_Reactor_2_3)
+1 B(Reaction_3_Reactor_2_2)
-1 B(Reaction_3_Reactor_2_3)
-1 Q(Reactor_2_2)
+1 Q(Reactor_2_3)
= 0

c_e_cons(unit_mass_balance_Reactor_2_time_4)_:
+1 B(Reaction_1_Reactor_2_2)
-1 B(Reaction_1_Reactor_2_4)
+1 B(Reaction_2_Reactor_2_2)
-1 B(Reaction_2_Reactor_2_4)
+1 B(Reaction_3_Reactor_2_3)
-1 B(Reaction_3_Reactor_2_4)
-1 Q(Reactor_2_3)
+1 Q(Reactor_2_4)
= 0

c_e_cons(unit_mass_balance_Reactor_2_time_5)_:
+1 B(Reaction_1_Reactor_2_3)
-1 B(Reaction_1_Reactor_2_5)
+1 B(Reaction_2_Reactor_2_3)
-1 B(Reaction_2_Reactor_2_5)
+1 B(Reaction_3_Reactor_2_4)
-1 B(Reaction_3_Reactor_2_5)
-1 Q(Reactor_2_4)
+1 Q(Reactor_2_5)
= 0

c_e_cons(unit_mass_balance_Reactor_2_time_6)_:
+1 B(Reaction_1_Reactor_2_4)
-1 B(Reaction_1_Reactor_2_6)
+1 B(Reaction_2_Reactor_2_4)
-1 B(Reaction_2_Reactor_2_6)
+1 B(Reaction_3_Reactor_2_5)
-1 B(Reaction_3_Reactor_2_6)
-1 Q(Reactor_2_5)
+1 Q(Reactor_2_6)
= 0

c_e_cons(unit_mass_balance_Reactor_2_time_7)_:
+1 B(Reaction_1_Reactor_2_5)
-1 B(Reaction_1_Reactor_2_7)
+1 B(Reaction_2_Reactor_2_5)
-1 B(Reaction_2_Reactor_2_7)
+1 B(Reaction_3_Reactor_2_6)
-1 B(Reaction_3_Reactor_2_7)
-1 Q(Reactor_2_6)
+1 Q(Reactor_2_7)
= 0

c_e_cons(unit_mass_balance_Reactor_2_time_8)_:
+1 B(Reaction_1_Reactor_2_6)
-1 B(Reaction_1_Reactor_2_8)
+1 B(Reaction_2_Reactor_2_6)
-1 B(Reaction_2_Reactor_2_8)
+1 B(Reaction_3_Reactor_2_7)
-1 B(Reaction_3_Reactor_2_8)
-1 Q(Reactor_2_7)
+1 Q(Reactor_2_8)
= 0

c_e_cons(unit_mass_balance_Reactor_2_time_9)_:
+1 B(Reaction_1_Reactor_2_7)
-1 B(Reaction_1_Reactor_2_9)
+1 B(Reaction_2_Reactor_2_7)
-1 B(Reaction_2_Reactor_2_9)
+1 B(Reaction_3_Reactor_2_8)
-1 B(Reaction_3_Reactor_2_9)
-1 Q(Reactor_2_8)
+1 Q(Reactor_2_9)
= 0

c_e_cons(unit_mass_balance_Still_time_0)_:
-1 B(Separation_Still_0)
+1 Q(Still_0)
= 0

c_e_cons(unit_mass_balance_Still_time_1)_:
+0.90000000000000002 B(Separation_Still_0)
-1 B(Separation_Still_1)
-1 Q(Still_0)
+1 Q(Still_1)
= 0

c_e_cons(unit_mass_balance_Still_time_10)_:
-1 B(Separation_Still_10)
+0.10000000000000001 B(Separation_Still_8)
+0.90000000000000002 B(Separation_Still_9)
+1 Q(Still_10)
-1 Q(Still_9)
= 0

c_e_cons(unit_mass_balance_Still_time_2)_:
+0.10000000000000001 B(Separation_Still_0)
+0.90000000000000002 B(Separation_Still_1)
-1 B(Separation_Still_2)
-1 Q(Still_1)
+1 Q(Still_2)
= 0

c_e_cons(unit_mass_balance_Still_time_3)_:
+0.10000000000000001 B(Separation_Still_1)
+0.90000000000000002 B(Separation_Still_2)
-1 B(Separation_Still_3)
-1 Q(Still_2)
+1 Q(Still_3)
= 0

c_e_cons(unit_mass_balance_Still_time_4)_:
+0.10000000000000001 B(Separation_Still_2)
+0.90000000000000002 B(Separation_Still_3)
-1 B(Separation_Still_4)
-1 Q(Still_3)
+1 Q(Still_4)
= 0

c_e_cons(unit_mass_balance_Still_time_5)_:
+0.10000000000000001 B(Separation_Still_3)
+0.90000000000000002 B(Separation_Still_4)
-1 B(Separation_Still_5)
-1 Q(Still_4)
+1 Q(Still_5)
= 0

c_e_cons(unit_mass_balance_Still_time_6)_:
+0.10000000000000001 B(Separation_Still_4)
+0.90000000000000002 B(Separation_Still_5)
-1 B(Separation_Still_6)
-1 Q(Still_5)
+1 Q(Still_6)
= 0

c_e_cons(unit_mass_balance_Still_time_7)_:
+0.10000000000000001 B(Separation_Still_5)
+0.90000000000000002 B(Separation_Still_6)
-1 B(Separation_Still_7)
-1 Q(Still_6)
+1 Q(Still_7)
= 0

c_e_cons(unit_mass_balance_Still_time_8)_:
+0.10000000000000001 B(Separation_Still_6)
+0.90000000000000002 B(Separation_Still_7)
-1 B(Separation_Still_8)
-1 Q(Still_7)
+1 Q(Still_8)
= 0

c_e_cons(unit_mass_balance_Still_time_9)_:
+0.10000000000000001 B(Separation_Still_7)
+0.90000000000000002 B(Separation_Still_8)
-1 B(Separation_Still_9)
-1 Q(Still_8)
+1 Q(Still_9)
= 0

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= W(Heating_Heater_0) <= 1
   0 <= W(Heating_Heater_1) <= 1
   0 <= W(Heating_Heater_2) <= 1
   0 <= W(Heating_Heater_3) <= 1
   0 <= W(Heating_Heater_4) <= 1
   0 <= W(Heating_Heater_5) <= 1
   0 <= W(Heating_Heater_6) <= 1
   0 <= W(Heating_Heater_7) <= 1
   0 <= W(Heating_Heater_8) <= 1
   0 <= W(Heating_Heater_9) <= 1
   0 <= W(Heating_Heater_10) <= 1
   0 <= W(Reaction_1_Reactor_1_0) <= 1
   0 <= W(Reaction_1_Reactor_1_1) <= 1
   0 <= W(Reaction_1_Reactor_1_2) <= 1
   0 <= W(Reaction_1_Reactor_1_3) <= 1
   0 <= W(Reaction_1_Reactor_1_4) <= 1
   0 <= W(Reaction_1_Reactor_1_5) <= 1
   0 <= W(Reaction_1_Reactor_1_6) <= 1
   0 <= W(Reaction_1_Reactor_1_7) <= 1
   0 <= W(Reaction_1_Reactor_1_8) <= 1
   0 <= W(Reaction_1_Reactor_1_9) <= 1
   0 <= W(Reaction_1_Reactor_1_10) <= 1
   0 <= W(Reaction_1_Reactor_2_0) <= 1
   0 <= W(Reaction_1_Reactor_2_1) <= 1
   0 <= W(Reaction_1_Reactor_2_2) <= 1
   0 <= W(Reaction_1_Reactor_2_3) <= 1
   0 <= W(Reaction_1_Reactor_2_4) <= 1
   0 <= W(Reaction_1_Reactor_2_5) <= 1
   0 <= W(Reaction_1_Reactor_2_6) <= 1
   0 <= W(Reaction_1_Reactor_2_7) <= 1
   0 <= W(Reaction_1_Reactor_2_8) <= 1
   0 <= W(Reaction_1_Reactor_2_9) <= 1
   0 <= W(Reaction_1_Reactor_2_10) <= 1
   0 <= W(Reaction_2_Reactor_1_0) <= 1
   0 <= W(Reaction_2_Reactor_1_1) <= 1
   0 <= W(Reaction_2_Reactor_1_2) <= 1
   0 <= W(Reaction_2_Reactor_1_3) <= 1
   0 <= W(Reaction_2_Reactor_1_4) <= 1
   0 <= W(Reaction_2_Reactor_1_5) <= 1
   0 <= W(Reaction_2_Reactor_1_6) <= 1
   0 <= W(Reaction_2_Reactor_1_7) <= 1
   0 <= W(Reaction_2_Reactor_1_8) <= 1
   0 <= W(Reaction_2_Reactor_1_9) <= 1
   0 <= W(Reaction_2_Reactor_1_10) <= 1
   0 <= W(Reaction_2_Reactor_2_0) <= 1
   0 <= W(Reaction_2_Reactor_2_1) <= 1
   0 <= W(Reaction_2_Reactor_2_2) <= 1
   0 <= W(Reaction_2_Reactor_2_3) <= 1
   0 <= W(Reaction_2_Reactor_2_4) <= 1
   0 <= W(Reaction_2_Reactor_2_5) <= 1
   0 <= W(Reaction_2_Reactor_2_6) <= 1
   0 <= W(Reaction_2_Reactor_2_7) <= 1
   0 <= W(Reaction_2_Reactor_2_8) <= 1
   0 <= W(Reaction_2_Reactor_2_9) <= 1
   0 <= W(Reaction_2_Reactor_2_10) <= 1
   0 <= W(Reaction_3_Reactor_1_0) <= 1
   0 <= W(Reaction_3_Reactor_1_1) <= 1
   0 <= W(Reaction_3_Reactor_1_2) <= 1
   0 <= W(Reaction_3_Reactor_1_3) <= 1
   0 <= W(Reaction_3_Reactor_1_4) <= 1
   0 <= W(Reaction_3_Reactor_1_5) <= 1
   0 <= W(Reaction_3_Reactor_1_6) <= 1
   0 <= W(Reaction_3_Reactor_1_7) <= 1
   0 <= W(Reaction_3_Reactor_1_8) <= 1
   0 <= W(Reaction_3_Reactor_1_9) <= 1
   0 <= W(Reaction_3_Reactor_1_10) <= 1
   0 <= W(Reaction_3_Reactor_2_0) <= 1
   0 <= W(Reaction_3_Reactor_2_1) <= 1
   0 <= W(Reaction_3_Reactor_2_2) <= 1
   0 <= W(Reaction_3_Reactor_2_3) <= 1
   0 <= W(Reaction_3_Reactor_2_4) <= 1
   0 <= W(Reaction_3_Reactor_2_5) <= 1
   0 <= W(Reaction_3_Reactor_2_6) <= 1
   0 <= W(Reaction_3_Reactor_2_7) <= 1
   0 <= W(Reaction_3_Reactor_2_8) <= 1
   0 <= W(Reaction_3_Reactor_2_9) <= 1
   0 <= W(Reaction_3_Reactor_2_10) <= 1
   0 <= W(Separation_Still_0) <= 1
   0 <= W(Separation_Still_1) <= 1
   0 <= W(Separation_Still_2) <= 1
   0 <= W(Separation_Still_3) <= 1
   0 <= W(Separation_Still_4) <= 1
   0 <= W(Separation_Still_5) <= 1
   0 <= W(Separation_Still_6) <= 1
   0 <= W(Separation_Still_7) <= 1
   0 <= W(Separation_Still_8) <= 1
   0 <= W(Separation_Still_9) <= 1
   0 <= W(Separation_Still_10) <= 1
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
binary
  W(Heating_Heater_0)
  W(Heating_Heater_1)
  W(Heating_Heater_2)
  W(Heating_Heater_3)
  W(Heating_Heater_4)
  W(Heating_Heater_5)
  W(Heating_Heater_6)
  W(Heating_Heater_7)
  W(Heating_Heater_8)
  W(Heating_Heater_9)
  W(Heating_Heater_10)
  W(Reaction_1_Reactor_1_0)
  W(Reaction_1_Reactor_1_1)
  W(Reaction_1_Reactor_1_2)
  W(Reaction_1_Reactor_1_3)
  W(Reaction_1_Reactor_1_4)
  W(Reaction_1_Reactor_1_5)
  W(Reaction_1_Reactor_1_6)
  W(Reaction_1_Reactor_1_7)
  W(Reaction_1_Reactor_1_8)
  W(Reaction_1_Reactor_1_9)
  W(Reaction_1_Reactor_1_10)
  W(Reaction_1_Reactor_2_0)
  W(Reaction_1_Reactor_2_1)
  W(Reaction_1_Reactor_2_2)
  W(Reaction_1_Reactor_2_3)
  W(Reaction_1_Reactor_2_4)
  W(Reaction_1_Reactor_2_5)
  W(Reaction_1_Reactor_2_6)
  W(Reaction_1_Reactor_2_7)
  W(Reaction_1_Reactor_2_8)
  W(Reaction_1_Reactor_2_9)
  W(Reaction_1_Reactor_2_10)
  W(Reaction_2_Reactor_1_0)
  W(Reaction_2_Reactor_1_1)
  W(Reaction_2_Reactor_1_2)
  W(Reaction_2_Reactor_1_3)
  W(Reaction_2_Reactor_1_4)
  W(Reaction_2_Reactor_1_5)
  W(Reaction_2_Reactor_1_6)
  W(Reaction_2_Reactor_1_7)
  W(Reaction_2_Reactor_1_8)
  W(Reaction_2_Reactor_1_9)
  W(Reaction_2_Reactor_1_10)
  W(Reaction_2_Reactor_2_0)
  W(Reaction_2_Reactor_2_1)
  W(Reaction_2_Reactor_2_2)
  W(Reaction_2_Reactor_2_3)
  W(Reaction_2_Reactor_2_4)
  W(Reaction_2_Reactor_2_5)
  W(Reaction_2_Reactor_2_6)
  W(Reaction_2_Reactor_2_7)
  W(Reaction_2_Reactor_2_8)
  W(Reaction_2_Reactor_2_9)
  W(Reaction_2_Reactor_2_10)
  W(Reaction_3_Reactor_1_0)
  W(Reaction_3_Reactor_1_1)
  W(Reaction_3_Reactor_1_2)
  W(Reaction_3_Reactor_1_3)
  W(Reaction_3_Reactor_1_4)
  W(Reaction_3_Reactor_1_5)
  W(Reaction_3_Reactor_1_6)
  W(Reaction_3_Reactor_1_7)
  W(Reaction_3_Reactor_1_8)
  W(Reaction_3_Reactor_1_9)
  W(Reaction_3_Reactor_1_10)
  W(Reaction_3_Reactor_2_0)
  W(Reaction_3_Reactor_2_1)
  W(Reaction_3_Reactor_2_2)
  W(Reaction_3_Reactor_2_3)
  W(Reaction_3_Reactor_2_4)
  W(Reaction_3_Reactor_2_5)
  W(Reaction_3_Reactor_2_6)
  W(Reaction_3_Reactor_2_7)
  W(Reaction_3_Reactor_2_8)
  W(Reaction_3_Reactor_2_9)
  W(Reaction_3_Reactor_2_10)
  W(Separation_Still_0)
  W(Separation_Still_1)
  W(Separation_Still_2)
  W(Separation_Still_3)
  W(Separation_Still_4)
  W(Separation_Still_5)
  W(Separation_Still_6)
  W(Separation_Still_7)
  W(Separation_Still_8)
  W(Separation_Still_9)
  W(Separation_Still_10)
end
