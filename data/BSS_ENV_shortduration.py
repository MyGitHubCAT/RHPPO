import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple,deque

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Short-duration interval; the charging process is extended to four stages.
class Battery(object):
    def __init__(self,id: int, cpt, btransit_step=1,bstageduration = 1):
        self.step_duration = 1 #

        self.SOC = 1
        self.SOH = 1 #0.8 + round(random.random(),2) *0.1
        self.MIN_SOH = 0.8 #
        self.capacity = cpt #
        self.ID = id
        self.CCOUNT = 0 #

        #
        self.MAXCCOUNT = 2000 #
        self.SOH_decrease = (1-self.MIN_SOH)/2000

        self.if_charge_end = True #
        self.charge_type = 0 # 0 for full charged, 1 for 80 percentage charged
        #
        self.charge_stage = 0 #
        self.charge_target = 0 #
        
        self.SOC_stage1 = 100/120 #
        self.SOC_stage2 = 1
        
        

        self.chargingpower1 = 100 #
        self.chargingpower2 = 20 #
        
        self.request_total_energy = 0
        self.request_stage_energy = 0

        self.chargingSOC = 0 # Required charging percentage.
        self.charging_request_time = 0 # Remaining charging time for the current charging session.
        self.charging_total_time = 0 # Planned total charging time for the current charging session.

        self.transit_k = btransit_step # Total transit time; number of time steps required between the BSS and charging station.
        self.ctransit = 0 # Record elapsed transit time.
    
    def FreshSOH(self):
        self.SOH = self.SOH-self.SOH_decrease

    # Start the charging plan, set the target parameters, and enter the waiting-for-charging state.
    def ChargePlanStart(self, full_charge : int):#ctype 0 for full charge, 1 for 80 percentage charge
        
        self.charge_stage = 0 #
        self.request_total_energy = 0
        self.request_stage_energy = 0# Reset information.

        self.charge_type = full_charge
        self.charge_target = self.SOC_stage2*self.SOH if full_charge == 0 else self.SOC_stage1*self.SOH
        self.if_charge_end = False
        self.CCOUNT += 1
        
        self.request_total_energy = self.capacity * self.charge_target
        self.request_stage_energy  = self.capacity * self.SOC_stage1*self.SOH
        #print('Charging plan: current SOC {0}, target charging SOC {1}'.format(self.SOC,self.charge_target))
        # self.CalChargingTime()
        # self.charging_request_time = self.charging_total_time # Record the remaining charging time.

    

    # Battery degradation is temporarily ignored, but a battery degradation cost can be considered for each charging session.
    #def CalChargingTime(self,chargingpower):# Charging time, divided into two stages.
    #     changesoc_stage1 = self.SOC_stage1-self.SOC
    #     changesoc_stage2 = self.charge_target - self.SOC_stage1
        
    #     rated_capacity = self.SOH * self.capacity
    #     stage1_duration = (changesoc_stage1 * rated_capacity)/chargingpower

    #     # The charging process in the constant-voltage stage requires further reference data.
    #     # beta = ((1-self.SOC_stage1)*rated_capacity)/chargingpower
    #     # stage2_duration = beta * math.log((1-self.SOC_stage1)/())
    #     stage2_duration = 2 if changesoc_stage2 == 0.2 else 0 # Use a fixed value temporarily.
    #     self.charging_total_time = stage1_duration + stage2_duration

    def Charge(self):# Charge for a fixed duration under the given rated power.
        if self.if_charge_end:
            return
        
        if (self.charge_stage == 0) or (self.charge_stage==1):
            oldsoc = self.SOC
            self.SOC = min(self.SOC_stage1*self.SOH ,self.SOC + self.SOC_stage1*self.SOH /2)
            self.request_stage_energy =self.capacity*(self.SOC - oldsoc)

        if (self.charge_stage == 2)or(self.charge_stage == 3):
            oldsoc = self.SOC
            self.SOC = min(1*self.SOH, self.SOC +(1-self.SOC_stage1)*self.SOH /2)
            self.request_stage_energy =self.capacity * (self.SOC - oldsoc)

        message1 = 'Battery {0}, target charging SOC {1}, total energy required to complete the charging plan {2}, energy required in this time stage {3}\n'.format(self.ID,self.charge_target,self.request_total_energy,self.request_stage_energy)    
        
        self.charge_stage += 1
        stagechargestage = self.charge_stage
        if abs(self.SOC-self.charge_target)<0.001 :
                self.FinishCharging()
        message2 = 'Battery {0} is charged once, charging stage {1}, current SOC {2}, target reached {3}'.format(self.ID,stagechargestage,self.SOC,self.if_charge_end)
        return message1+message2

    def FinishCharging(self):
        self.charge_stage = 0
        self.if_charge_end = True
            
            #self.FreshSOH()

    # Battery energy consumption.
    def SOCChange(self,changesoc):
        oldsoc = self.SOC
        self.SOC = round(max(0,self.SOC-changesoc),4)
        changedsoc = oldsoc - self.SOC
        message = 'Battery {0}, consumed SOC {1:.3f}, current SOC {2:.3f}\n'.format(self.ID,changedsoc,self.SOC)
        return changedsoc,message          


    def Message(self,if_print= False):
        # message = 'Battery ID {0}, rated capacity {1}.\n'.format(self.ID,self.capacity) +\
        #       'Remaining SOC percentage {0}, battery health {1}, maximum battery capacity {2}.\n'.format(self.SOC,self.SOH,self.SOH*self.capacity)
        message = 'Battery ID {0}, remaining SOC percentage {1}'.format(self.ID,self.SOC)
        if if_print:
            print('Battery ID {0}, rated capacity {1}.\n'.format(self.ID,self.capacity) +
                'Remaining SOC percentage {0}, battery health {1}, maximum battery capacity {2}.\n'.format(self.SOC,self.SOH,self.SOH*self.capacity))
        return message
    #def Charge(self):


class ElectricVehicle(object):
    def __init__(self, id,evaverageworksoc, cpt = 120,evtransit_step = 1,evstageduration = 1):
        self.cpt = cpt
        self.evtransitstep = evtransit_step 
        self.stageduration = evstageduration
        self.battery = Battery(id,cpt,btransit_step = evtransit_step,bstageduration= evstageduration)
        self.id = id
        self.workstate=0 # 0 normal working state, 1 stopped.
        self.energy_consumption = 30 # Energy consumption + 0.
        self.swap_soc =0# 0.2 # Battery-swapping threshold; swapping is required below this value.

        # Stage energy demand minus consumed energy equals the remaining energy demand in the current stage. This is used for EVs that swap batteries mid-stage. The value is a relative SOC consumption ratio rather than an absolute consumption value; for example, stageworksoc=0.2 means that 20% of the theoretical energy demand is consumed in the current stage.
        self.AVERGE_WORKSOC = evaverageworksoc# Fixed mean energy consumption.
        self.stage_worksoc = evaverageworksoc# Actual energy demand in the stage.
        self.used_soc = 0# Temporary consumed SOC in the stage; if a mid-stage battery swap occurs, this intermediate value is used to determine the remaining stage demand.

    
    def Reinitial(self):
        self.battery = Battery(self.id,self.cpt,self.evtransitstep,self.stageduration)
        self.stage_worksoc = self.AVERGE_WORKSOC
        self.worked_soc = 0
        self.workstate=0 # 0 normal working state, 1 stopped.

    def NeedSwap(self):
        needswap = True if abs(self.battery.SOC - self.swap_soc)<0.01 else False
        return needswap
    
    def Update(self,workingtime,emergencefactor):# Fixed mean, working time, and emergency factor. This update is called under multiple cases for working EVs: (1) the EV works for the first time and its energy consumption meets the expected demand; (2) the EV depletes its battery after initial operation but successfully swaps within the same stage and continues consuming the remaining demand; (3) the EV depletes its battery after initial operation but fails to swap and queues at the station, where the actual consumption is limited by the remaining battery SOC.
        timereturn = 48
        if self.used_soc == 0:# When the working state is calculated for the first time, determine the expected energy consumption for the current stage.
            worktimefactor =   1 * math.cos(workingtime*math.pi/timereturn) + 1# Fixed at 1. Periodic variation over 0-48 with a trigonometric cycle from 2 to 0 to 2.
            stage_need_soc = self.AVERGE_WORKSOC * worktimefactor * emergencefactor  + random.random()*0.1# Fixed energy consumption is mean consumption multiplied by the working-time factor and emergency factor, plus random energy consumption.
            self.stage_worksoc = stage_need_soc
        updatesoc = self.stage_worksoc - self.used_soc
        needswap,workingmessage = self.Working(updatesoc)
        return needswap,workingmessage
        
    def Working(self,updatesoc):
        message1 = 'EV {0} is about to enter the working state. Theoretical SOC consumption in this stage is {1:.3f}, consumed SOC is {2:.3f}. Battery status changes are as follows:\n'.format(self.id,self.stage_worksoc,self.used_soc)
        consumsoc,bmessage = self.battery.SOCChange(updatesoc)
        self.used_soc += consumsoc# Record the consumed SOC.
        needswap = self.NeedSwap()
        message2 = 'EV {0} actual SOC consumption in this working state is {1:.3f}, cumulative consumed SOC in the current stage is {2:.3f}, whether the EV needs battery swapping: {3}'.format(self.id,consumsoc,self.used_soc,needswap)
  
        message = message1+bmessage+message2
        return needswap,message
    
    def Message(self):
        print('EV ID {0}. Vehicle working state {1}, stage SOC consumption {2:.3f}.'.format(self.id,self.workstate,self.stage_worksoc))
        print('Battery information:') 
        self.battery.Message()   


class BSS(object):
    def __init__(self,id_start, SCALE: int = 20, cpt = 120, bsstransit_step = 1 ,charger_num = 5,stageduration = 1):

        self.bsstransitstep = bsstransit_step 
        self.stage_duration = stageduration
        self.T_BNUM = SCALE

        self.charger_NUM = charger_num#
        self.charging_power = 100 # Rated power is 100 kW.

        #full charged batteries
        self.BatteryT1_wait = deque()
        self.BatteryT1_available = deque()
        self.BatteryT1_quit = deque()

        #80 percentage charged batteries
        self.BatteryT2_wait = deque()
        self.BatteryT2_available = deque()
        self.BatteryT2_quit = deque()

        #swapped batteries
        self.BatteryT3_wait = deque()# Batteries waiting at the current time step.
        self.BatteryT3_available = deque()# Batteries requiring charging-plan assignment at the current time step.
        self.BatteryT3_plan = deque()# Batteries currently under a charging plan.
        self.BatteryT3_charging = deque()# Batteries currently being charged; batteries that are not completed after one charging operation temporarily enter this queue.


        self.EV_wait = deque()
    
        self.EV_quit = deque()

        self.Initial(id_start, cpt)

        
        # Charging-count control.
        self.plan_charge_count = 0
        self.stage_charge_count = 0
        self.out_count = 0 # Number of chargers reserved beyond actual demand.

        # Electric-energy charging control.
        self.MAX_energy_request = self.charging_power * self.charger_NUM
        self.energy_request = 0# Required energy.

        self.ps_energy = 0
        self.stage_energy_get = 0# Actual obtained energy; when charging by charger count, it equals the requested energy; when charging by energy amount, it is no greater than the requested energy.

        self.stage_energy_out=0 # Excess energy at each time step; when charging by charger count, it is zero; when charging by energy amount, it equals the difference between requested and obtained energy.
        # Electricity-purchase cost.
        self.stage_carbon = 0
        self.stage_expense = 0
        self.carbon_cost = 0
        self.energy_cost = 0

        self.stage_swapped_ev_num = 0
        self.stage_unswapped_ev_num = 0
        self.mask = np.zeros((self.T_BNUM))

        # Count of batteries that will enter the available queue at each time step.
        self.BT1_readdy_num = 0# Number of fully charged batteries.
        self.BT2_readdy_num = 0# Number of batteries charged to 80%.
        self.BT3_readdy_num = 0# Number of swapped-out batteries.



    def Initial(self,ids,b_cpt):
        for i in range(self.T_BNUM):
            battery = Battery(i+ids, b_cpt,self.bsstransitstep,self.stage_duration)
            self.BatteryT1_available.append(battery)
        self.stage_swapped_ev_num = 0

    # def Update(self):
    #     self.BSSSwapAction()

# Battery-swapping process of the BSS.
    def BSSAddEV(self, ev):
        self.EV_wait.append(ev)

    def BSSSwapAction(self):# Preparation: identify EVs requiring swapping. T1 batteries, i.e., fully charged batteries, are provided first, followed by 80%-charged batteries.
        available_battery_num = len(self.BatteryT1_available)+len(self.BatteryT2_available)
        waiting_num = len(self.EV_wait)
        swap_num = min(waiting_num,available_battery_num)# Total number of swaps.
        T1_num = min(swap_num,len(self.BatteryT1_available))# Number of full-charge battery swaps.
        T2_num = min(len(self.BatteryT2_available),swap_num-T1_num)# Number of 80%-charge battery swaps.
        messages = []
        message1 = 'Total EVs requesting battery swapping in the BSS: {0}, total available batteries in the BSS: {1}, EVs that can be served at the current time step: {2}, full-charge battery demand: {3}, 80%-charge battery demand: {4}'.format(
            waiting_num,available_battery_num, swap_num,T1_num,T2_num)
        messages.append(message1)

        for i in range(T1_num):
            swap_batteryT1:Battery = self.BatteryT1_available.popleft()
            swapped_ev:ElectricVehicle = self.EV_wait.popleft()
            old_b:Battery = self.SwapBatteryforOneEV(swapped_ev,swap_batteryT1)

            self.BatteryT3_wait.append(old_b)
            self.EV_quit.append(swapped_ev)
            messageev = 'Battery-swapping process: EV {0}, onboard battery ID {1} is removed, full-charge battery ID {2} is installed'.format(swapped_ev.id,old_b.ID,swap_batteryT1.ID)
            messages.append(messageev)

        for i in range(T2_num):
            swap_batteryT2 = self.BatteryT2_available.popleft()
            swapped_ev = self.EV_wait.popleft()
            old_b = self.SwapBatteryforOneEV(swapped_ev,swap_batteryT2)
            
            self.BatteryT3_wait.append(old_b)
            self.EV_quit.append(swapped_ev)
            messageev = 'Battery-swapping process: EV {0}, onboard battery ID {1} is removed, full-charge battery ID {2} is installed'.format(swapped_ev.id,old_b.ID,swap_batteryT2.ID)
            messages.append(messageev)

        self.stage_swapped_ev_num = len(self.EV_quit)
        self.stage_unswapped_ev_num = waiting_num - self.stage_swapped_ev_num

        message2 = 'Battery swapping is completed. {0} EVs have completed swapping and are ready to leave the BSS; the remaining {1} EVs failed to swap and enter the waiting queue'.format(self.stage_swapped_ev_num,self.stage_unswapped_ev_num)
        messages.append(message2)  
        return messages
        
    def SwapBatteryforOneEV(self,ev: ElectricVehicle, avl_battery:Battery):
        old_battery = ev.battery
        ev.battery = avl_battery
        return old_battery

    # Charging process.
    # Assign charging plans.
    def BatteriesChargingPlan(self,batteryplan = 0):# Determine the charging amount according to the charging plan. Note that the shape of the charging plan is (self.T_BNUM,).
        self.BatteryT3_Mask()# Update mask.
        battery_plan =batteryplan * self.mask # np.ones((self.T_BNUM,)) * self.mask# If no plan is provided, all existing batteries are fully charged by default.
        #print(self.mask)
        
        # if batteryplan != None:
        #     battery_plan = batteryplan * self.mask
        for i in range(len(self.BatteryT3_available)):
            chargingplan_b = self.BatteryT3_available.popleft()
            chargingplan_b.ChargePlanStart(battery_plan[i])
            self.BatteryT3_plan.append(chargingplan_b)
        new_Cplan = battery_plan[:int(sum(self.mask))] 
        current_Cplan = [ i.charge_type for i in self.BatteryT3_plan]
        message1 = 'The new charging plan is {0}; after adding the new plan, the currently executed charging plans are {1}'.format(new_Cplan,current_Cplan)
        return message1

    def BatteryT3_Mask(self):# The dimension of the battery charging plan is fixed as the initial number of swappable batteries. During operation, the batteryT3 dimension is no greater than the initial number of swappable batteries, and a mask is used for marking.
        self.mask = self.mask * 0
        for i in range(len(self.BatteryT3_available)):
            self.mask[i] = 1


    # Charge by charger count.
    def Chargercount(self,chargecount=5):# Ensure the reserved number does not exceed the available number of chargers.
        self.plan_charge_count = min(chargecount,self.charger_NUM)      

    def BatterieschargingbycountOneStage(self,pstation):
        charging_inf = []# Information record.
        finish_charge_count = 0
        finish_t1charge_count = 0
        finish_t2charge_count = 0
        plan_count = len(self.BatteryT3_plan)# Number of planned charging operations.
        self.stage_charge_count = min(plan_count,self.plan_charge_count)# The actual number of charging operations is limited by the reserved chargers and batteries awaiting charging.
        message1 = 'Use {} chargers for charging at the current time step'.format(self.stage_charge_count)
        charging_inf.append(message1)

        self.stage_energy_get=0# Reset used energy.
        self.out_count = self.plan_charge_count - self.stage_charge_count# Number of wasted chargers in the current stage.

        for i in range(self.stage_charge_count):
            b_plan:Battery = self.BatteryT3_plan.popleft()
            message2 = 'Battery ID {0} starts charging'.format(b_plan.ID)
            charging_inf.append(message2)          
            self.stage_energy_get += b_plan.request_stage_energy
            message3 = b_plan.Charge()
            charging_inf.append(message3)
            #print(b_plan.if_charge_end)
            if b_plan.if_charge_end:
                finish_charge_count += 1
                if abs(b_plan.charge_type - 0)<0.0001:
                    finish_t1charge_count += 1
                    self.BatteryT1_wait.append(b_plan)
                   
                if abs(b_plan.charge_type - 1)<0.0001:
                    finish_t1charge_count += 1
                    self.BatteryT2_wait.append(b_plan)
            else:
                self.BatteryT3_charging.append(b_plan)
            
        for i in range(len(self.BatteryT3_charging)):
            b_still_need_charging = self.BatteryT3_charging.pop()
            self.BatteryT3_plan.appendleft(b_still_need_charging)

        self.energy_request = self.stage_energy_get
        self.stage_energy_out = 0

        message5='Charging summary: requested chargers {0}, active chargers {1}, total batteries requiring charging {2}.\nAfter charging, consumed energy is {3}, batteries reaching the charging target {4}, including full-charge batteries {5} and 80%-charge batteries {6}'.format(
            self.plan_charge_count,self.stage_charge_count, plan_count,self.stage_energy_get ,
            finish_charge_count,finish_t1charge_count,finish_t2charge_count)
        charging_inf.append(message5)

        getenergymessage = self.GetEnergy(pstation)
        charging_inf.append(getenergymessage)

        return charging_inf

# Determine energy allocation.
    def GetEnergy(self, pstation):
        self.ps_energy = self.energy_request             
        pscarb,psexpense,psmessage = self.CostInOnestageInPS(self.ps_energy,pstation)   
        self.stage_carbon = pscarb
        self.stage_expense = psexpense
        message1= 'Total purchased energy at the current time step {0:.3f}, carbon emission {1:.3f}, electricity cost {2:.3f}'.format(
                    self.energy_request,self.stage_carbon,self.stage_expense)
        return message1 + psmessage 
    
    def CostInOnestageInPS(self,energyrequest,powerstation):
        carb,expense,message = powerstation.ChargeCost(energyrequest)
        return carb,expense,message
    

    def FreshWaitBattery(self):
        messages = ['Battery transit process is updated by one stage']
        self.BT1_readdy_num = len(self.BatteryT1_wait)
        self.BT2_readdy_num = len(self.BatteryT2_wait)
        self.BT3_readdy_num = len(self.BatteryT3_wait)

        messages.append('Full-charge battery transit update: {0} full-charge batteries are in transit'.format(self.BT1_readdy_num))
        for i in range(self.BT1_readdy_num):
            BT1_readdy:Battery = self.BatteryT1_wait.popleft()
            messageb = 'Full-charge battery ID {0}, transit process update, required transit time {1}, current transit time {2},'.format(BT1_readdy.ID,BT1_readdy.transit_k,BT1_readdy.ctransit)
            if BT1_readdy.ctransit >= BT1_readdy.transit_k:
                self.BatteryT1_available.append(BT1_readdy)
                BT1_readdy.ctransit = 0
                messageb += 'After the current stage ends, battery transit is completed, and the battery enters the BSS at the beginning of the next stage to provide swapping service'
            else:
                BT1_readdy.ctransit += 1
                self.BatteryT1_wait.append(BT1_readdy)
                messageb += 'After the current stage ends, battery transit continues, and the transit process continues at the beginning of the next stage'
            messages.append(messageb)

        for i in range(self.BT2_readdy_num):
            BT2_readdy = self.BatteryT2_wait.popleft()
            messageb = '80%-charge battery ID {0}, transit process update, required transit time {1}, current transit time {2},'.format(BT2_readdy.ID,BT2_readdy.transit_k,BT2_readdy.ctransit)
            
            if BT2_readdy.ctransit >= BT2_readdy.transit_k:
                self.BatteryT2_available.append(BT2_readdy)
                BT2_readdy.ctransit = 0
                messageb += 'After the current stage ends, battery transit is completed, and the battery enters the BSS at the beginning of the next stage to provide swapping service'
            else:
                BT2_readdy.ctransit += 1
                self.BatteryT2_wait.append(BT2_readdy)
                messageb += 'After the current stage ends, battery transit continues, and the transit process continues at the beginning of the next stage'
            messages.append(messageb)

        for i in range(self.BT3_readdy_num):
            BT3_readdy = self.BatteryT3_wait.popleft()
            messageb = 'Depleted battery ID {0}, transit process update, required transit time {1}, current transit time {2},'.format(BT3_readdy.ID,BT3_readdy.transit_k,BT3_readdy.ctransit)
            if BT3_readdy.ctransit >= BT3_readdy.transit_k:
                self.BatteryT3_available.append(BT3_readdy)
                BT3_readdy.ctransit = 0
                messageb += 'After the current stage ends, battery transit is completed, and the battery enters the BCC at the beginning of the next stage for charging preparation'
            else:
                BT3_readdy.ctransit += 1
                self.BatteryT3_wait.append(BT3_readdy)
                messageb += 'After the current stage ends, battery transit continues, and the transit process continues at the beginning of the next stage'
            messages.append(messageb)
        return messages

    def MessageChargingBattertplan(self):
        print('Total batteries currently under charging plans: {0}'.format(len(self.BatteryT3_plan)))
        for i in range(len(self.BatteryT3_plan)):
            print('Battery ID {0}, remaining SOC {1}, charging plan {2}, current charging stage {3}'.format(self.BatteryT3_plan[i].ID,
                                                                self.BatteryT3_plan[i].SOC,
                                                                self.BatteryT3_plan[i].charge_target,
                                                                self.BatteryT3_plan[i].charge_stage))
    def MessageWaitingPlanBattery(self):
        print('Total batteries waiting for charging-plan assignment: {0}'.format(len(self.BatteryT3_available)))
        for i in range(len(self.BatteryT3_available)):
            print('Battery ID {0}, remaining SOC {1}'.format(self.BatteryT3_available[i].ID,self.BatteryT3_available[i].SOC))
    def Message(self):
        
        print('BSS station information:')
        print('Total batteries under charging plans: {0}'.format(len(self.BatteryT3_plan)))
        

class PS(object): # Power station.
    def __init__(self, carbonfactor = 570.3 ,MAX_ENERGY_PERSTAGE = 99999):
        
        self.ptype = 0 #
        # 2023 electricity data: average grid CO2 emission factor is 0.5703 t/MWh, i.e., 570.3 g/kWh. Clean energy, represented by wind power data, has an average CO2 emission factor of 2-81 g/kWh; the mean value 40 is used.
        self.carbon_factor = carbonfactor 
        self.stage_energy = MAX_ENERGY_PERSTAGE # Maximum energy supply; a very large value is assigned to a normal power station, while solar power supply can vary.
        self.price = 0
        
    def ChangePrice(self,cprice):
        self.price = cprice

    def ChargeCost(self,usedenergy):
        carb = self.CarbonEmission(usedenergy)
        expense = self.EnergyExpense(usedenergy)
        typename = 'Thermal Power Station' if self.ptype == 0 else 'Wind and Solar Power Station'
        message = '\nStation type: {0}. Carbon emission factor: {1} g/kWh. Current electricity price: {2} $/kWh. Current energy supply: {3}. Generated carbon emission: {4}, generated cost: {5}'.format(
             typename,self.carbon_factor,self.price,usedenergy, carb,expense)
        return carb,expense,message
    
    def CarbonEmission(self,usedenergy):# usedenergy is measured in kWh.
        carbon = usedenergy * self.carbon_factor
        return carbon
    def EnergyExpense(self,usedenergy):#
        expense = usedenergy *self.price
        return expense
    
    def Message(self):
        print('Station type: {0}.\nCarbon emission factor: {1} g/kWh.\nCurrent energy supply: {2}.\nCurrent electricity price: {3} $/kWh.'.format(
            'thermal power' if self.ptype == 0 else 'renewable energy',self.carbon_factor,self.stage_energy,self.price))


class GameManager(object):
    def __init__(self,opt, default_energydata_length = 300, 
    adenergydatanum = 24,ps_eprice_ds = None):
        
        # Environment objects.
        # region
        # Fixed environment information.
        self.EV_NUM = opt.ev_num
        self.bssbatterynum = opt.bssbatterynum 

        # Overall fleet information.
        self.EV_fleet = []# Aggregate collection for recording EV information.
        self.EV_working_fleet = deque()# Collection of EVs working at the beginning of the time step.
        self.EV_swapped_fleet = deque()# Collection of EVs that swap batteries mid-stage. This queue allows EVs that complete swapping within a stage to continue working with the remaining demand, which better reflects reality.
        self.fleet_ev_num =  opt.ev_num
        self.fleet_workingev_num = opt.ev_num# This variable may be of limited use; it records EVs in the working state at the beginning of the time step and does not include EVs that continue working after swapping.
        self.fleet_swappedev_num = 0

        self.bsschargernum = opt.charger_num
        self.bss = BSS(id_start = self.EV_NUM,SCALE=self.bssbatterynum, bsstransit_step=opt.transit_step,charger_num=opt.charger_num ) # Battery IDs in the BSS start from the number of EVs.
        
        self.ps = PS()
        # endregion


        # Environment time information.
        self.transitstep = opt.transit_step
        self.time = 0
      

        # Dynamic environment information.
        # Pre-action attributes: at time T, the state at time t is observed.
 
        self.ava_bnum = 0 # Total number of swappable batteries available in the BSS at the current time step.
        self.ava_bt1num = 0 # Total number of type-1 swappable batteries available in the BSS at the current time step.
        self.ava_bt2num = 0 # Total number of type-1 swappable batteries available in the BSS at the current time step.
        self.bss_queuelength = 0 # EV queue length.
        # Post-action attributes: at time T, the state at time t-1 is observed.
        self.stage_request_chargers = 0 # Number of chargers requested at each time step.
        self.stage_work_chargers = 0 # Number of chargers actually operating at each time step.

        self.stage_used_energy = 0# Total consumed energy.
        self.stage_used_gdenergy = 0# Consumed grid energy.
  

        # Dataset information.
        # Energy data variations are embedded in the environment and are no longer externally provided.
        # Price dataset.
        self.ps_energy_price = []
   
        # Mean EV energy consumption.
        self.ev_average_worksoc = opt.ev_average_worksoc

        # Data length information.
        self.energydata_length = default_energydata_length # Default minimum data length of the environment.
        self.adenergydata_num = adenergydatanum # Length of supplementary energy data before time 0, used to provide historical price information for the prediction model.
        self.ed_length = self.energydata_length + adenergydatanum+1 # Energy data before time 0 are available but not involved in environment calculation; they only provide historical data support for the model.
        
        
        # Text information.
        self.Update_Stage_Inf = [] # Collect updated text information.

        # Initialize all data and provide information.
        self.Initial(ps_eprice_ds, False)
        
    def Initial(self,ps_eprice_ds = None, reinitial: bool = True):
        # Clear data before initialization.
        self.EV_fleet.clear()   
        self.EV_working_fleet.clear()
        self.EV_swapped_fleet.clear()
        for i in range(self.EV_NUM):
            init_ev = ElectricVehicle(i,evaverageworksoc=self.ev_average_worksoc,evtransit_step=self.transitstep) # EV battery IDs start from 0 and end at the number of EVs minus 1.
            self.EV_fleet.append(init_ev)
            self.EV_working_fleet.append(init_ev)
        self.fleet_ev_num = self.EV_NUM
        self.fleet_workingev_num = self.EV_NUM
        self.fleet_swappedev_num = 0

        self.bss = BSS(self.EV_NUM,self.bssbatterynum,bsstransit_step=self.transitstep,charger_num=self.bsschargernum) # Battery IDs in the BSS start from the number of EVs.
        
        self.ps = PS()
   
        self.time = 0
        
        self.is_done = 0# Whether the environment is terminated.

        # Environment information.
        self.bss_queuelength = 0 # Queue length at the beginning of the time step or at the end of the previous time step.
        
        # Battery information.
        self.UpdateBSSInf()

        self.Update_Stage_Inf.clear()
        if not reinitial:# Reinitializing GM does not require updating the price database.
            self.InitialEnergyDataset(ps_eprice_ds)
            
        self.FreshEnergyData()

    def InitialEnergyDataset(self,ps_ep_ds = None):   # Electricity price data.
        self.ps_energy_price.clear()
        if (type(ps_ep_ds) != type(None)) and (len(ps_ep_ds)>= self.ed_length):
            for i in ps_ep_ds:
                self.ps_energy_price.append(i)
        else:           
            for i in range(self.ed_length):
                self.ps_energy_price.append(4+8*math.sin((-math.pi) + (((i-24)*math.pi)/24)))
        
                
    def norm_data(self, i,mu,sigma ):# Normal distribution.
        return math.exp(-(math.pow((i-mu)/sigma,2)/2))/(sigma*math.sqrt(2*math.pi)) 

    def FreshEnergyData(self):
        self.ps.ChangePrice(self.ps_energy_price[self.time + self.adenergydata_num])

    def EnvState(self):
        ava_bt1_num = self.ava_bt1num
        ava_bt2_num = self.ava_bt2num
        ava_b_num = ava_bt1_num + ava_bt2_num
        
        energy_cost = self.bss.stage_expense
    
        carbon_cost = self.bss.stage_carbon

        stage_bssevqueue_length = self.bss_queuelength # Queue status.

        wasted_count = self.bss.out_count# Number of wasted chargers.

        ctime = int(self.time%96)# Determined according to the time span of the stage.
        old_price = self.ps_energy_price[self.time: self.time + self.adenergydata_num]
                
        is_done = self.is_done  
   
        return ava_bt1_num,ava_bt2_num,ava_b_num,energy_cost,carbon_cost,stage_bssevqueue_length,wasted_count,ctime,is_done
        #step_ask_energy = 
   
    def Update(self,reserve_act,chargeplan,emergencefactor = 1,showmessages = False):# The emergency state considers a surge in energy consumption, controlled by emergencefactor.

        self.Update_Stage_Inf.clear()

        self.UpdateEnvdata()# Update environment data first and reset necessary state information, including EV consumed SOC in the current stage.

        self.UpdateWorkingEVFleet(emergencefactor)# Update fleet information; first update.

        self.UpdateSwapEVBSS()# Update battery swapping between EVs and the BSS according to demand.

        self.UpdateSwappedEVFleet(emergencefactor)

        self.UpdateBSSPlan(chargeplan)# Update the BSS planning process.
        
        self.UpdateBSSChargingByCount(reserve_act)

        self.UpdateEnvInf()# Update environment information.
        if showmessages:
            print('Current time is {0}, current electricity price is {1:.3f}'.format(self.time,self.ps.price))
            print('Environment update information at time {0} is as follows'.format(self.time))
            for i in self.Update_Stage_Inf:
                print(i)


    def UpdateEnvdata(self):  
        self.time += 1
        
        self.FreshEnergyData()
        for ev in self.EV_fleet:
            ev.used_soc = 0
        

    def UpdateWorkingEVFleet(self,emergencefactor):
        self.fleet_workingev_num= len(self.EV_working_fleet)
        
        ccount = 0

        message1 = 'Initial update stage of the working EV fleet:'
        message2 = 'Total number of working EVs: {0}'.format(self.fleet_workingev_num)
        self.Update_Stage_Inf.append(message1)
        self.Update_Stage_Inf.append(message2)
        
        needswapevs = []
        for i in range(self.fleet_workingev_num):
            ev = self.EV_working_fleet.popleft()

            timeratio = self.time # averageworksoc,workingtime,emergencefactor
            needswap,ev_message = ev.Update(timeratio,emergencefactor)#
            self.Update_Stage_Inf.append(ev_message)
            if needswap:
                needswapevs.append(ev)
                ccount+=1
            else:
                self.EV_working_fleet.append(ev)#
        needswapevs.sort(key = lambda x:x.used_soc)#
        for ev in needswapevs:
            self.bss.BSSAddEV(ev)

        
        message3 = 'The initial fleet update is completed. After operation, {0} EVs have depleted batteries and enter the BSS waiting queue; the remaining {1} EVs wait for operation in the next time step'.format(ccount,self.fleet_workingev_num-ccount)

        self.Update_Stage_Inf.append(message3)
    
    def UpdateSwapEVBSS(self):
        message1 = 'Battery-swapping stage between EVs and the BSS'
        self.Update_Stage_Inf.append(message1)
        
        swap_message = self.bss.BSSSwapAction() # Swapped-out batteries enter the waiting transit queue. Even if the transit time is zero, they can only be assigned charging plans at the BCC in the next stage.
        self.Update_Stage_Inf +=swap_message

        for i in range(self.bss.stage_swapped_ev_num):# EVs that complete swapping leave the station and enter the secondary working queue. All swapped EVs enter this secondary update queue and operate according to their remaining stage demand.
            swaped_ev = self.bss.EV_quit.popleft()
            self.EV_swapped_fleet.append(swaped_ev)
        message2 = 'Battery-swapping stage between EVs and the BSS is completed'
        self.Update_Stage_Inf.append(message2)

    def UpdateSwappedEVFleet(self,emergencefactor):
        self.wait_evnum = len(self.bss.EV_wait)# Record the queue length of EVs that fail to swap after all swapping operations in the current stage.
        self.fleet_swappedev_num= len(self.EV_swapped_fleet)
        #print('Fleet operation stage: start updating the fleet. Number of EVs operating at the current time step: {0}'.format(self.fleet_ev_num))
        ccount = 0
        message1 = 'Update stage for the working EV fleet after battery swapping:'
        message2 = 'Number of EVs continuing operation after battery swapping: {0}'.format(self.fleet_swappedev_num)
        self.Update_Stage_Inf.append(message1)
        self.Update_Stage_Inf.append(message2)

        needswapevs = []
        for i in range(self.fleet_swappedev_num):
            ev = self.EV_swapped_fleet.popleft()

            timeratio = self.time # averageworksoc,workingtime,emergencefactor
            needswap,ev_message = ev.Update(timeratio,emergencefactor)# Fixed energy consumption equals mean consumption multiplied by the working-time factor and emergency factor; emergency variation is considered on top of periodic energy consumption.
            self.Update_Stage_Inf.append(ev_message)
            if needswap:
                needswapevs.append(ev)
                ccount+=1
            else:
                self.EV_working_fleet.append(ev)# After operation, EVs that have completed battery swapping enter the working-state queue and directly operate in the next stage.
        
        # In general, a short-duration environment does not deplete a newly swapped battery within one stage. Therefore, EVs that have just swapped batteries should not deplete them again and re-enter the BSS waiting queue; this part is commented out.
        needswapevs.sort(key = lambda x:x.used_soc)# EVs that consume their battery earlier enter the swapping queue first.
        for ev in needswapevs:
            self.bss.BSSAddEV(ev)

        
        message3 = 'The secondary fleet update is completed. After operation, among EVs that re-enter the working state after swapping, {0} EVs have depleted batteries and enter the BSS waiting queue; the remaining {1} EVs enter the working-state queue and continue operation in the next time step'.format(ccount,self.fleet_swappedev_num-ccount)

        self.Update_Stage_Inf.append(message3)

    # Multi-step processing of self.BatteryT3_available.
    def UpdateBSSPlan(self,BT3_ava_chargingplan=1):# BT3_ava_chargingplan is a fixed-length vector.
        message1 = self.bss.BatteriesChargingPlan(BT3_ava_chargingplan)
        self.Update_Stage_Inf.append(message1)
        #self.bss.MessageWaitingPlanBattery()
    
    # Charge according to the number of charging bays.
    def UpdateBSSChargingByCount(self,stagechargecount=5):
        message1 = 'BSS charging stage'
        self.bss.Chargercount(stagechargecount)
        message2_list = self.bss.BatterieschargingbycountOneStage(self.ps)
        # Information collection.
        self.Update_Stage_Inf.append(message1)
        self.Update_Stage_Inf += message2_list
        #self.bss.MessageChargingBattertplan()
        message3 = 'BSS charging stage is completed'
        self.Update_Stage_Inf.append(message3)
    


    def UpdateEnvInf(self):
        message1 = 'Battery transit process update:'
        self.Update_Stage_Inf.append(message1)
        messages = self.bss.FreshWaitBattery()# Update the battery waiting queue, treated as transit between the BSS and the charging station.
        self.Update_Stage_Inf += messages 
        self.is_done = 0 if self.time <= self.energydata_length -1 else 1

        self.UpdateBSSInf()

        if self.is_done:
            message = 'Environment terminated; operating time has exceeded '+ str(self.energydata_length)
            self.Update_Stage_Inf.append(message)
        return 

    # Update BSS information in the environment.
    def UpdateBSSInf(self):
        self.ava_bt1num = len(self.bss.BatteryT1_available)
        self.aba_bt2num = len(self.bss.BatteryT2_available)
        self.ava_bnum = self.ava_bt1num + self.aba_bt2num 
        self.bss_queuelength = len(self.bss.EV_wait)

        self.stage_request_chargers = self.bss.plan_charge_count
        self.stage_work_chargers = self.bss.stage_charge_count

        self.stage_used_energy = self.bss.energy_request
  
        self.stage_used_gdenergy = self.bss.ps_energy


    def Message(self):
        print('Environment information at the end of time {0} is as follows'.format(self.time))
        print('At the end of the current time step, the number of EVs in the working state is {0}, and the number of EVs queued at the station is {1}'.format(len(self.EV_working_fleet),self.bss_queuelength))
        print('In the station, available full-charge batteries: {0}, 80%-charge batteries: {1}, queued EVs: {2}'.format(len(self.bss.BatteryT1_available),len(self.bss.BatteryT2_available),self.bss_queuelength))   
        print('\n')  
       