import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple,deque
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = ("cuda" 
          if torch.cuda.is_available() 
          else "mps" 
          if torch.backends.mps.is_available() 
          else "cpu")
 

class PPONetwork(torch.nn.Module):
    def __init__(self,inpdim,reserve_dim=5,init_cpratio_id = 1):
        super(PPONetwork,self).__init__()
        self.inputlayer = torch.nn.Linear(inpdim,64).to(device)
        self.linear1 = torch.nn.Linear(64,128).to(device)

        #self.norm = torch.nn.BatchNorm1d(4).to(device)
        #self.relu = torch.nn.LeakyReLU().to(device)

        self.linear2 = torch.nn.Linear(128,64).to(device)

        self.out1 = torch.nn.Linear(64,reserve_dim).to(device)
        #self.out2 = torch.nn.Linear(64,chargeplan_dim).to(device)
        self.out3 = torch.nn.Linear(64,1).to(device)
        self.softmax = torch.nn.Softmax(dim=-1)
        
        
    def forward(self,inp):
        x = inp.to(device)
        inp1 = self.inputlayer(x)
        tanh1 = torch.nn.Tanh()(inp1)
        line1 = self.linear1(tanh1)
        line2 = self.linear2(line1)#(reshape_128)
        tanh2 = torch.nn.Tanh()(line2)
        #print(tanh2)
        out_1 = self.out1(tanh2)
        
        out_3 = self.out3(tanh2)
        res1 = self.softmax(out_1)

        #print(out_3)
        res3 = out_3
        return res1,res3 #bs*reservedim+1,bs*2,bs*1





BSSTransition = namedtuple('BSSTransition',('state_pool','value_pool','policy_ratio_pool','clipres_pool','log_prob_pool','reward_pool','done_pool'))#收集连续动作
class BSSMemory(object):#
    def __init__(self,capacity):
        self.memory = deque([],maxlen = capacity)
    def push(self,*args):
        self.memory.append(BSSTransition(*args))
    def sample(self):
        batch = list(self.memory)
        return zip(*batch)
    def __len__(self):
        return len(self.memory)
    def clear(self):
        self.memory.clear()

#####BSSMemory for update training strategy
# BSSTransition = namedtuple('BSSTransition',('state_pool','value_pool','act_pool','old_prob_pool','reward_pool','done_pool'))#
# class BSSMemory(object):#
#     def __init__(self,capacity):
#         self.memory = deque([],maxlen = capacity)
#     def push(self,*args):
#         self.memory.append(BSSTransition(*args))
#     def sample(self):
#         batch = list(self.memory)
#         return zip(*batch)
#     def __len__(self):
#         return len(self.memory)
#     def clear(self):
#         self.memory.clear()


class BSS_DRL(object):#er_dim=5*5+1
    def __init__(self,opt):#ifstep 
        self.options =  opt
        self.lr = opt.learning_ratio
        
        self.bss_agent = PPONetwork(opt.inp_dim,reserve_dim=opt.charger_num +1,init_cpratio_id= 0).to(device)
        
        self.bss_agent_target = PPONetwork(opt.inp_dim,reserve_dim=opt.charger_num +1,init_cpratio_id=0).to(device)
        self.bss_agent_target.load_state_dict(self.bss_agent.state_dict())  
        self.bss_agent_target.eval()

        self.bss_agent_optimizer = torch.optim.Adam(self.bss_agent.parameters(),lr = self.lr)
        #self.bss_a2c_target_optimizer = torch.optim.Adam(self.evrp_critic.parameters(),lr = self.lr)
        self.if_clamp = opt.if_clamp
        self.bssmemory_num = opt.MEMORY_NUM
        self.bssmemorys = [] 
        for i in range(4):
            self.bssmemorys.append(BSSMemory(2500))
        
    #  
    def select_action(self,bssstate,if_train=False):#state_s的格式为bs*ev_num*(state_n)
        prob1,state_value = self.bss_agent(bssstate)
        prob1_target,state_value_target = self.bss_agent_target(bssstate)
        
        reserve_dist = torch.distributions.categorical.Categorical(prob1)
   
        reserve_dist_target = torch.distributions.categorical.Categorical(prob1_target)
        
        
        reserve_act = reserve_dist.sample()

        reserve_act_target = reserve_dist_target.sample()
        
        
        if if_train:
            return reserve_act_target,reserve_dist_target,reserve_dist,state_value_target
                
        return reserve_act,prob1
    
    
    def cliprange(self,prob,prob_target,segma=0.2):
        policy_ratio = prob/prob_target
        max_clip = 1+segma
        min_clip = 1-segma
        clipres = torch.clamp(policy_ratio,min = min_clip,max = max_clip)
        return clipres
    
    def compute_returns(self,next_value, rewards, masks, gamma=0.99):
        value = next_value.detach()#return 1*1
        rewards = (rewards - rewards.mean())/rewards.std()
       
        returns = torch.zeros_like(rewards)    #
        for act_i in reversed(range(len(rewards))):
            #print(rewards[act_i],gamma , value , masks[act_i])
            value = rewards[act_i] + gamma * value * (1-masks[act_i])
            returns[act_i] = value
        return returns #pool_size*1*1

    #######update trainging strategy
    # def update_bss(self,bssmemorys,nstate_s_list):
    #     #print(0.001*entropy)
    #     mmnum = len(bssmemorys)
    #     totallossvalue = 0
    #     inpoches = 6#
    #     for _ in range(inpoches): #
    #         losses = 0
    #         for i in range(self.bssmemory_num):
    #             'state_pool','value_pool','act_pool','old_prob_pool','reward_pool','done_pool'
    #             #tate_pool, value_pool, policy_ratio_pool, clipres_pool, reward_pool, done_pool = bssmemorys[i].sample()
    #             state_pool, value_pool, act_pool,old_prob_pool, reward_pool, done_pool = bssmemorys[i].sample()
    #             #在bs维度进行合并
    #             state_poolstack =torch.concat(state_pool,dim=0).to(device).detach()#pool_length*evnum*evdim #
    #             old_value_poolstack = torch.concat(value_pool,dim=0).to(device).detach()#pool_length*1#
    #             act_poolstack = torch.concat(act_pool,dim=0).to(device).detach()#pool_length*evnum*1
    #             old_prob_poolstack = torch.concat(old_prob_pool,dim=0).to(device).detach()#pool_length*evnum*1
    #             #print(value_poolstack)
    #             reward_poolstack = torch.concat(reward_pool,dim=0).to(device)#pool_length*1 #
    #             #print(reward_poolstack)
    #             done_poolstack = torch.concat(done_pool,dim=0).to(device)#pool_length*1
    #             #print(done_poolstack)
    #             # print(state_poolstack.shape, old_value_poolstack.shape,\
    #             #         policy_ratio_poolstack.shape, clipres_poolstack.shape,\
    #             #             reward_poolstack.shape, done_poolstack.shape)
                

    #             current_prob_poolstack, current_bssqstatevalue  =self.bss_agent(state_poolstack)#bs*evnum*evactnum
    #             current_dist_poolstack = torch.distributions.categorical.Categorical(current_prob_poolstack)
    #             current_entropy = current_dist_poolstack.entropy().mean()
                              
    #             current_actprob_poolstack = torch.gather(current_prob_poolstack,dim=-1,index = act_poolstack)

    #             policy_ratio_poolstack = current_actprob_poolstack/old_prob_poolstack
    #             eps_clip = 0.2#clip范围值
    #             policy_clipres_poolstack = torch.clamp(policy_ratio_poolstack, 1-eps_clip, 1+eps_clip) #

    #             #1 1 1
    #             _,next_value = self.bss_agent_target(nstate_s_list[i])#bs*1
            
    #             #
    #             returns = self.compute_returns(next_value.squeeze(0),reward_poolstack,done_poolstack,gamma = self.options.GAMMA)
    #             advantages = returns - old_value_poolstack
    #             with torch.no_grad():#
    #                 adv_t = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)

    #             _,current_value = self.bss_agent(state_poolstack)
    #             #print(current_value.shape,returns.shape)
    #             current_advantage = returns - current_value
  
    #             clip_advantages =  adv_t.detach() * policy_clipres_poolstack
    #             policy_ratio_advantages = advantages * policy_ratio_poolstack
    #             # print(clip_advantages)
    #             # print(policy_ratio_advantages)
    #             stack_advantages = torch.cat((clip_advantages,policy_ratio_advantages),dim=-1)
    #             ppo_advantages = stack_advantages.min(dim=-1)[0].unsqueeze(-1)
                
    #             #print( ppo_advantages)
    #             #print(advantages)
    #             #print(log_prob_poolstack)
    #             #print(advantages)
                
    #             actor_loss = -ppo_advantages.mean() # (-log_prob_poolstack * advantages).mean()
    #             critic_loss = 0.5 * current_advantage.pow(2).mean()
               
    #             total_loss = 10* actor_loss + critic_loss - 0.0001* current_entropy
    #             losses+= total_loss
            
    #         mmloss = losses/mmnum
    #         lossvalue = mmloss.cpu().detach().item()
    #         totallossvalue += lossvalue
           
    #         #      
    #         self.bss_agent_optimizer.zero_grad()      
    #         mmloss.backward()
    #         for name, param in self.bss_agent.named_parameters():
    #             #print(name,param.grad)
    #             if param.grad is not None:
    #                 param.grad.clamp_(-1,1)
    #         #for param in self.evrp_critic.parameters():           
    #         # print(param.grad)
    #         self.bss_agent_optimizer.step()    
    #     return totallossvalue/inpoches

    def update_bss(self,nstate_s_list,entropys):
        #print(0.001*entropy)
        mmnum = self.bssmemory_num
        losses = []
        for i in range(self.bssmemory_num):
            state_pool,value_pool,policy_ratio_pool,clipres_pool,log_prob_pool,reward_pool,done_pool= self.bssmemorys[i].sample()

            state_poolstack =torch.stack(state_pool).to(device)
            old_value_poolstack = torch.stack(value_pool).to(device)#pool_length*bs*valuedim
            policy_ratio_poolstack = torch.stack(policy_ratio_pool).to(device) #pool_length*bs*valuedim
            clipres_poolstack = torch.stack(clipres_pool).to(device) #pool_length*bs*valuedim
            #print(value_poolstack)
            log_prob_poolstack = torch.stack(log_prob_pool).to(device) #pool_length*bs*valuedim
            #print(log_prob_poolstack)
            reward_poolstack = torch.stack(reward_pool).to(device)#pool_length*bs*valuedim
            #print(reward_poolstack)
            done_poolstack = torch.stack(done_pool).to(device)#pool_length*bs*valuedim
            #print(done_poolstack)
       
            #1 1 1
            _,next_value = self.bss_agent_target(nstate_s_list[i])#bs*1
        
            returns = self.compute_returns(next_value.squeeze(0),reward_poolstack,done_poolstack,gamma = self.options.GAMMA)
            advantages = returns - old_value_poolstack

            _,current_value = self.bss_agent(state_poolstack)
            #print(current_value.shape,returns.shape)
            current_advantage = returns - current_value
            
            #print(advantages.shape)
            # print('clip'+str(clipres_poolstack))
            # print('policy' +str( policy_ratio_poolstack))
            clip_advantages = advantages * clipres_poolstack
            policy_ratio_advantages = advantages * policy_ratio_poolstack
            # print(clip_advantages)
            # print(policy_ratio_advantages)
            stack_advantages = torch.cat((clip_advantages,policy_ratio_advantages),dim=-1)
            ppo_advantages = stack_advantages.min(dim=-1)[0].unsqueeze(-1)
            
            #print( ppo_advantages)
            #print(advantages)
            #print(log_prob_poolstack)
            #print(advantages)
            
            actor_loss = -ppo_advantages.mean() # (-log_prob_poolstack * advantages).mean()
            critic_loss = 0.5 * current_advantage.pow(2).mean()
            
            total_loss = 10* actor_loss + critic_loss - 0.0001*entropys[i]
            losses.append(total_loss)
        
        mmloss = losses[0]
        for i in range( mmnum-1):
            mmloss = mmloss + losses[i+1]
        mmloss = mmloss/mmnum
        
                     
        self.bss_agent_optimizer.zero_grad()      
        mmloss.backward()
        if self.if_clamp:
            for param in self.bss_agent.parameters():           
                #print(param.grad)
                if param.grad is not None:
                    param.grad.data.clamp(-20,20)
        #for param in self.evrp_critic.parameters():           
           # print(param.grad)
        self.bss_agent_optimizer.step()    
        
        
    def savemodel(self,dicpath,addname):
        torch.save(self.bss_agent.state_dict(),os.path.join(dicpath,addname+'.pth'))

