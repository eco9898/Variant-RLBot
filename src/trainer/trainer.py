import glob, sys, multiprocessing, pathlib
from time import sleep, time
from  datetime import datetime
from typing import Dict
import numpy as np
from rlgym.envs import Match
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym.utils.state_setters import RandomState
from rlgym.utils.terminal_conditions.common_conditions import *
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym.utils.reward_functions.common_rewards.misc_rewards import *
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import *
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import *
from rlgym.utils.reward_functions.common_rewards.conditional_rewards import *
from rlgym.utils.reward_functions import CombinedReward
from rlgym_tools.sb3_utils.sb3_log_reward import * # line 21 = if len(returns) > 0: # my own fix


parent_directory = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_directory)

from utils.advanced_padder import AdvancedObsPadder
from utils.discrete_act import DiscreteAction
from utils.util_classes import *
from utils.modified_states import *

MAX_INSTANCES_NO_PAGING = 5
WAIT_TIME_NO_PAGING = 20
WAIT_TIME_PAGING = 35
INSTANCE_SETUP_TIME = 50 #for safety

total_num_instances = 10
kickoff_instances = total_num_instances // 3
match_instances = total_num_instances - kickoff_instances
models: List = [["kickoff", kickoff_instances], ["match", match_instances]]
models: List = [["match", total_num_instances]]
#models: List = [["kickoff", total_num_instances]]

data_location = parent_directory + "/data/"

paging = False
if total_num_instances > MAX_INSTANCES_NO_PAGING:
    paging = True
wait_time=WAIT_TIME_NO_PAGING
if paging:
    wait_time=WAIT_TIME_PAGING

def start_training(send_messages: multiprocessing.Queue, model_args: List):
    global paging, wait_time, total_num_instances
    name = model_args[0]
    num_instances = model_args[1]
    frame_skip = 12          # Number of ticks to repeat an action
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    reward_log_file = data_location + "logs/" + name + "/reward_" + str(datetime.now().hour) + "-" + str(datetime.now().minute)

    fps = 120/frame_skip #120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
    team_size = 3
    self_play = True
    if self_play:
        agents_per_match = 2*team_size
    else:
        agents_per_match = team_size
    n_env = agents_per_match * num_instances
    print(">>>Wait time:        ", wait_time)
    print(">>># of instances:   ", num_instances)
    print(">>># of trainers:    ", n_env)
    print(">>>Paging:           ", paging)
    print(">>># of env:         ", n_env)
    target_steps = int(1_000_000)#*(num_instances/total_num_instances))
    steps = target_steps//n_env #making sure the experience counts line up properly
    print(">>>Steps:            ", steps)
    batch_size = (100_000//(steps))*(steps) #getting the batch size down to something more manageable - 80k in this case at 5 instances, but 25k at 16 instances
    if batch_size == 0:
        batch_size = steps
    print(">>>Batch size:       ", batch_size)
    training_interval = int(25_000_000)#*(num_instances/total_num_instances))
    print(">>>Training interval:", training_interval)
    mmr_save_frequency = 50_000_000
    print(">>>MMR frequency:    ", mmr_save_frequency)
    send_messages.put(1)
    attackRewards = SB3CombinedLogReward(
        (
            VelocityPlayerToBallReward(),
            LiuDistancePlayerToBallReward(),
            RewardIfTouchedLast(LiuDistanceBallToGoalReward()),
            RewardIfTouchedLast(VelocityBallToGoalReward()),
            RewardIfClosestToBall(AlignBallGoal(0,1), True),
        ),
        (2.0, 0.2, 1.0, 1.0, 0.8),
        reward_log_file + "_attack")

    defendRewards = SB3CombinedLogReward(
        (
            VelocityPlayerToBallReward(),
            LiuDistancePlayerToBallReward(),
            RewardIfTouchedLast(LiuDistanceBallToGoalReward()),
            RewardIfTouchedLast(VelocityBallToGoalReward()),
            AlignBallGoal(1,0)
        ),
        (1.0, 0.2, 1.0, 1.0, 1.5),
        reward_log_file + "_defend")

    lastManRewards = SB3CombinedLogReward(
        (
            RewardIfTouchedLast(VelocityBallToGoalReward()),
            AlignBallGoal(1,0),
            LiuDistancePlayerToGoalReward(),
            ConstantReward()
        ),
        (2.0, 1.0, 0.6, 0.2),
        reward_log_file + "_last_man")

    kickoffRewards = SB3CombinedLogReward(
        (
            RewardIfClosestToBall(
                CombinedReward(
                    (
                        VelocityPlayerToBallReward(),
                        AlignBallGoal(0,1),
                        LiuDistancePlayerToBallReward(),
                        FlipReward(),
                        useBoost()
                    ),
                    (20.0, 1.0, 2.0, 2.0, 5.0)
                ),
                team_only=True
            ),
            RewardIfMidFromBall(defendRewards),
            RewardIfFurthestFromBall(lastManRewards),
            TeamSpacingReward(),
            pickupBoost()
        ),
        (2.0, 1.0, 1.5, 1.0, 0.4),
        reward_log_file + "_kickoff")

    def exit_save(model, name: str):
        model.save(data_location + "models/" + name + "/exit_save")

    def load_save(name: str, env, steps, batch_size, MlpPolicy, gamma):
        try:
            folder_path = data_location + 'models/' + name
            file_type = r'\*.zip'
            files = glob.glob(folder_path + file_type)
            newest_kickoff_model = max(files, key=os.path.getctime)[0:-4]
            model = PPO.load(
                newest_kickoff_model,
                env,
                device="auto",
                #custom_objects={"n_envs": env.num_envs}, #automatically adjusts to users changing instance count, may encounter shaping error otherwise
                #If you need to adjust parameters mid training, you can use the below example as a guide
                custom_objects={"n_envs": env.num_envs, "n_steps": steps, "batch_size": batch_size, "_last_obs": None, "tensorboard_log": data_location + "logs/" + name}
            )
            print(">>>Loaded previous exit save.")
            return model
        except:
            print(">>>No saved model found, creating new model.")
            from torch.nn import Tanh
            policy_kwargs = dict(
                activation_fn=Tanh,
                net_arch=[512, 512, dict(pi=[256, 256, 256], vf=[256, 256, 256])],
            )

            return PPO(
                MlpPolicy,
                env,
                n_epochs=10,                 # PPO calls for multiple epochs
                policy_kwargs=policy_kwargs,
                learning_rate=5e-5,          # Around this is fairly common for PPO
                ent_coef=0.01,               # From PPO Atari
                vf_coef=1.,                  # From PPO Atari
                gamma=gamma,                 # Gamma as calculated using half-life
                verbose=3,                   # Print out all the info as we're going
                batch_size=batch_size,             # Batch size as high as possible within reason
                n_steps=steps,                # Number of steps to perform before optimizing network
                tensorboard_log=data_location + "/logs/" + name,  # `tensorboard --logdir out/logs` in terminal to see graphs
                device="auto"                # Uses GPU if available
            )

    def get_base_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=team_size, #amount of bots per team
            tick_skip=frame_skip,
            self_play=self_play, #play against its self
            obs_builder=AdvancedObsPadder(3),  # Not that advanced, good default
            action_parser=DiscreteAction(),  # Discrete > Continuous don't @ me
            reward_function= CombinedReward((), ()),
            terminal_conditions = [TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],
            state_setter = RandomState()  # Resets to random
        )

    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        match: Match = get_base_match()
        match._reward_fn = SB3CombinedLogReward(
            (
                RewardIfAttacking(attackRewards),
                RewardIfDefending(defendRewards),
                RewardIfLastMan(lastManRewards),
                VelocityReward(),
                FaceBallReward(),
                EventReward(
                    team_goal=100.0,
                    goal=10.0 * team_size,
                    concede=-100.0 + (10.0 * team_size),
                    shot=10.0,
                    save=30.0,
                    demo=12.0,
                ),
                JumpTouchReward(),
                TouchBallReward(1.2),
                TeamSpacingReward(1500),
                FlipReward(),
                SaveBoostReward(),
                pickupBoost(),
                useBoost()
            ),
            #(1.0, 0.2, 0.5, 1.0, 1.0, 1.0, 1.5, 1.0, 5.0, 0.6, 0.2, 1.6, 10.0),
            #(0.17, 0.20, 0.15, 0.24, 0.14, 3.92, 54.70, 6.07, 0.37, 0.19, 0.12, 0.60, 37.27), checkpoint 1
            #(0.17, 0.20, 0.15, 0.24, 0.14, 3.92, 54.70, 6.07, 0.37, 0.19, 0.12, 0.60, 37.27),
            (0.14, 0.25, 0.32, 0.19, 0.13, 3.33, 16.66, 4.36, 0.23, 0.22, 0.35, 1.06, 71.35),
            reward_log_file)
        match._terminal_conditions = [TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()]
        match._state_setter = RandomState()  # Resets to random
        return match

    def get_kickoff():  # Need to use a function so that each instance can call it and produce their own objects
        match: Match = get_base_match()
        match._reward_fn = SB3CombinedLogReward(
            (
                kickoffRewards,
                VelocityReward(),
                FaceBallReward(),
                EventReward(
                    team_goal=100.0,
                    goal=10.0 * team_size,
                    concede=-100.0 + (10.0 * team_size),
                    shot=10.0,
                    save=30.0,
                    demo=12.0,
                ),
                JumpTouchReward(),
                TouchBallReward(1.2),
                TeamSpacingReward(1500),
                FlipReward(),
                SaveBoostReward(),
                pickupBoost(),
                useBoost(),
            ),
            (0.08, 0.52, 0.82, 27.56, 108.22, 41.05, 0.2, 0.1, 1.0, 3.81, 319.16),
            reward_log_file)
        #time out after 12 seconds encourage kickoff
        match._terminal_conditions = [TimeoutCondition(fps * 12), GoalScoredCondition()]
        match._state_setter = ModifiedState()  # Resets to kickoff position
        return match
    
    if name == "kickoff":
        match = get_kickoff
    else:
        match = get_match
    env = SB3MultipleInstanceEnv(match, model_args[1], force_paging=paging, wait_time=wait_time) #or 40            # Start instances, waiting 60 seconds between each
    env = VecCheckNan(env)                                # Optional
    env = VecMonitor(env)                                 # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

    model = load_save(model_args[0], env, steps, batch_size, MlpPolicy, gamma)

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    checkpoint_callback = CheckpointCallback(round(1_000_000*(num_instances/total_num_instances) / env.num_envs), save_path=data_location + "models/" + name, name_prefix="rl_model") # backup every 5 mins
    
    rewardList = [checkpoint_callback]

    attack_names = ["AttVel", 'AttDisToBall', "AttBallToGoalDis", "AttBallToGoalVel", "AttAlignGoalOff"]
    defend_names = ["DefVel", 'DefDisToBall', "DefBallToGoalDis", "DefBallToGoalVel", "DefAlignGoalDef"]
    last_man_names = ["LMBallToGoalVel", "LMAlignGoalDef", "LMDisToGoal", 'LMConstant']
    kickoff_names = ["KOCombination", "KODef", "KOLastman", "KOSpacing", "KOPickupBoost"]
    reward_graph_callback_attacks = SB3CombinedLogRewardCallback(
        attack_names,
        reward_log_file + "_attack")
    reward_graph_callback_defend = SB3CombinedLogRewardCallback(
        defend_names,
        reward_log_file + "_defend")
    reward_graph_callback_last_man = SB3CombinedLogRewardCallback(
        last_man_names,
        reward_log_file + "_last_man")
    reward_graph_callback_kickoff = SB3CombinedLogRewardCallback(
        kickoff_names,
        reward_log_file + "_kickoff")
    if name == "kickoff":
        reward_names = ["KickKickoff", "KickVel", 'KickFaceball', "KickEvent", "KickJumptouch", "KickTouch", "KickSpacing", "KickFlip", "KickSaveboost", "KickPickupboost", "KickUseboost"]
    else:
        reward_names = ["MatchAtt", "MatchDef", "MatchLastman", "MatchVel", 'MatchFaceball', "MatchEvent", "MatchJumptouch", "MatchTouch", "MatchSpacing", "MatchFlip", "MatchSaveboost", "MatchPickupboost", "MatchUseboost"]
    reward_graph_callback = SB3CombinedLogRewardCallback(
        reward_names,
        reward_log_file)
    if name == "kickoff":
        rewardList.extend([reward_graph_callback, reward_graph_callback_kickoff])
    else:
        rewardList.extend([reward_graph_callback, reward_graph_callback_attacks, reward_graph_callback_defend, reward_graph_callback_last_man])
    #1mill per 8/9 minutes at 5 instances?

    mmr_model_target_count = model.num_timesteps + (mmr_save_frequency - (model.num_timesteps % mmr_save_frequency)) #current steps + remaing steps until mmr model
    closed_messages = False
    try:
        while True:
            new_training_interval = training_interval - (model.num_timesteps % training_interval) # remaining steps to train interval
            print(">>>Training for %s timesteps" % new_training_interval)
            if not closed_messages:
                send_messages.put(2)
                send_messages.close()
                closed_messages = True
            #may need to reset timesteps when you're running a different number of instances than when you saved the model
            #subprocess learning
            model.learn(new_training_interval, callback=CallbackList(rewardList), reset_num_timesteps=False) #can ignore callback if training_interval < callback target
            exit_save(model, name)
            if model.num_timesteps >= mmr_model_target_count:
                model.save(f"{data_location}mmr_models/{model_args[0]}/{model.num_timesteps}")
                mmr_model_target_count += mmr_save_frequency

    except KeyboardInterrupt:
        print(">>>Exiting training")
    if not closed_messages:
        send_messages.close()
    print(">>>Saving model")
    exit_save(model, name)
    print(">>>Save complete")

def trainingMonitor(send_messages: multiprocessing.Queue, model_args):
    global wait_time, paging
    instances = model_args[1]
    done = False
    trainers_RLPIDs = []
    initial_RLPIDs = getRLInstances()
    print(">>Initial instances:", len(initial_RLPIDs))
    receive_messages = multiprocessing.Queue()
    trainer = multiprocessing.Process(target=start_training, args=[receive_messages, model_args])
    trainer.start()
    try:
        count = 0
        #Wait until setup is printed
        while receive_messages.empty() and trainer.is_alive():
            sleep(1)
        receive_messages.get()
        start = time()
        minimise = 0
        instance_crashed = False
        while count < instances  and trainer.is_alive() and not instance_crashed:
            print(">>Parsing instance:" + str(count + 1) + "+" + str(len(initial_RLPIDs)))
            new_instance = False
            if ((time() - start) // INSTANCE_SETUP_TIME > count):
                print(">>Instance took too long")
                break
            while (time() - start) // INSTANCE_SETUP_TIME <= count and not instance_crashed and not new_instance:
                sleep(0.2)
                curr_PIDs = getRLInstances()
                #clean initial instances
                to_remove = []
                for pid in initial_RLPIDs:
                    if pid not in curr_PIDs:
                        to_remove.append(pid) #store to remove later
                for pid in to_remove:
                    initial_RLPIDs.remove(pid)
                #Check for new instance
                for pid in curr_PIDs:
                    if pid not in trainers_RLPIDs and pid not in initial_RLPIDs:
                        count +=1
                        new_instance = True
                        print(">>Instances found:" + str(count) + "+" + str(len(initial_RLPIDs)))
                        trainers_RLPIDs.append(pid)
                        break
                #Check if instance died
                for pid in trainers_RLPIDs:
                    if pid not in curr_PIDs:
                        #trainer instance was closed
                        instance_crashed = True
                        break
            #minimise done windows
            #if (time() - start) // INSTANCE_SETUP_TIME > minimise:
            #    minimiseRL([trainers_RLPIDs[minimise]])
            #    minimise = (time() - start) // INSTANCE_SETUP_TIME
        if instance_crashed:
            print(">>Instance Died")
        done = False
        if count == instances:
            print(">>Waiting to start")
            try:
                start = time()
                while (time() - start) < INSTANCE_SETUP_TIME * 2 and trainer.is_alive():
                    if not receive_messages.empty():
                        break
                    sleep(0.1)
                if not receive_messages.empty() and trainer.is_alive():
                    if receive_messages.get() == 2:
                        done = True
                receive_messages.close()
            except KeyboardInterrupt:
                killRL(trainers_RLPIDs)
                exit()
        if count != instances or not done:
            print(">>Killing trainer: " + model_args[0])
            trainer.terminate()
        else:
            print(">>Finished parsing trainer: " + model_args[0])
            send_messages.put(1)
            send_messages.put(trainers_RLPIDs)
            send_messages.close()
            sleep(10)
            minimiseRL(trainers_RLPIDs)
            #trainer exit process and restart
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(">>Error: with trainer parser")
        #Continue to wait for trainer to die, then exit
    #trainer will shut down and save, please wait
    #after it has closed, RL instances will be killed
    while trainer.is_alive():
        sleep(1)
    killRL(trainers_RLPIDs)
    receive_messages.close()
    send_messages.close()
    

def start_starter(messages: Dict[str, multiprocessing.Queue], monitors: Dict[str, multiprocessing.Process], model_args, initial_instances, all_instances):
    name = model_args[0]
    instances = model_args[1]
    print(">Starting trainer: " + model_args[0])
    blacklist = initial_instances.copy()
    blacklist.append(all_instances)
    while True:
        messages[name] = multiprocessing.Queue()
        monitors[name] = multiprocessing.Process(target=trainingMonitor, args=[messages[name], model_args])
        monitors[name].start()
        #wait to open RL instances
        start = time()
        while (time() - start) < INSTANCE_SETUP_TIME * (instances + 2) and monitors[name].is_alive():
            if not messages[name].empty():
                break
        if not messages[name].empty():
            if messages[name].get() == 1:
                print(">Training started: " + name)
                return messages[name].get()
        print(">Restarting trainer: " + name)
        monitors[name].terminate()
        while monitors[name].is_alive():
                sleep(0.1)
        killRL(blacklist=blacklist)
        sleep(5)

if __name__ == "__main__":
    messages: Dict[str, multiprocessing.Queue] = {}
    monitors: Dict[str, multiprocessing.Process] = {}
    model_instances: Dict[str, List[int]]  = {}
    models_used: Dict[str, List]  = {}
    all_instances = []
    initial_instances = getRLInstances()
    #no try and catch is needed during startup as the starters will clean themselves up
    #RLGym can't have reopened a RL instance yet
    for model_args in models:
        model_instances[model_args[0]] = start_starter(messages, monitors, model_args, initial_instances, all_instances)
        all_instances.extend(model_instances[model_args[0]])
        models_used[model_args[0]] = model_args
    print(">Finished starting trainers")
    try:
        while True:
            for key in monitors:
                monitor = monitors[key]
                if monitor.is_alive():
                    sleep(1)
                else:
                    print(">Trainer crashed")
                    #Kill instances that weren't present before and weren't reported by trainer
                    blacklist = initial_instances.copy()
                    blacklist.append(all_instances)
                    killRL(blacklist=blacklist)
                    model_args = models_used[key]
                    #kill trainer's instances
                    killRL(model_instances[key])
                    #add logic to detect not all were killed and to search for extra instances, if they match kill them
                    # if an instance crashes RLgym restarts it
                    messages[key].close()
                    sleep(1)
                    model_instances[key] = start_starter(messages, monitors, model_args, initial_instances, all_instances)
                #trainer died restart loop
    except KeyboardInterrupt:
        for key in monitors:
            monitor = monitors[key]
            #trainers will shut down and save, please wait
            while monitor.is_alive():
                sleep(0.1)
            messages[key].close()
            #kill instances reported
            #killRL(all_instances)
            #Kill instances that weren't present before
            killRL(blacklist=initial_instances)