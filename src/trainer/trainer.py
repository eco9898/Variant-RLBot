import sys, multiprocessing, pathlib, os, pickle, torch, torch.jit, wandb
from time import sleep, time
from typing import Dict
import numpy as np
from rlgym.envs import Match
from redis import Redis
from torch.nn import Linear, Sequential, ReLU
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker
from rlgym.utils.state_setters import RandomState
from rlgym.utils.terminal_conditions.common_conditions import *
from rlgym.utils.reward_functions.common_rewards.misc_rewards import *
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import *
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import *
from rlgym.utils.reward_functions.common_rewards.conditional_rewards import *
from rlgym.utils.reward_functions import CombinedReward
from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import SplitLayer
#Comment line 69-72 of wandb.sdk.internal.stats
parent_directory = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_directory)
from utils.discrete_act import DiscreteAction
from utils.util_classes import *
from utils.modified_states import *

MAX_INSTANCES_NO_PAGING = 5
WAIT_TIME_NO_PAGING = 20
WAIT_TIME_PAGING = 35
INSTANCE_SETUP_TIME = 50 #for safety

total_num_instances = 3
run_learner = False
run_workers = True
run_workers = not run_learner
clear_redis = False
kickoff_instances = total_num_instances // 3
match_instances = total_num_instances - kickoff_instances
models: List = [["kickoff", kickoff_instances], ["match", match_instances]]
models: List = [["match", total_num_instances]]
#models: List = [["kickoff", total_num_instances]]

data_location = parent_directory + "/data/"
pickle_directory = parent_directory + "/trainer/"

pickleData = pickle.load(open(pickle_directory + "pickleData.obj","rb"))

paging = False
if total_num_instances > MAX_INSTANCES_NO_PAGING:
    paging = True
wait_time=WAIT_TIME_NO_PAGING
if paging:
    wait_time=WAIT_TIME_PAGING

def runWorker(match):
    global pickleData
    """
    Starts up a rocket-learn worker process, which plays out a game, sends back game data to the 
    learner, and receives updated model parameters when available
    """    
    # OPTIONAL ADDITION: LIMIT TORCH THREADS TO 1 ON THE WORKERS TO LIMIT TOTAL RESOURCE USAGE
    torch.set_num_threads(1)

    

    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    r = Redis(host="127.0.0.1", password=pickleData["REDIS"])
    print("Redis init")


    # LAUNCH ROCKET LEAGUE AND BEGIN TRAINING
    # -past_version_prob SPECIFIES HOW OFTEN OLD VERSIONS WILL BE RANDOMLY SELECTED AND TRAINED AGAINST
    RedisRolloutWorker(r, "Variant", match, 
        past_version_prob=0,#.2, 
        evaluation_prob=0,#.01,
        sigma_target=1,
        streamer_mode=False, 
        send_gamestates=False, 
        pretrained_agents=None, 
        human_agent=None,
        deterministic_old_prob=0#.5
    ).run()

def runLearner(send_messages: multiprocessing.Queue, save_dir, rewards):
    global pickleData
    print("Learner instance started")
    """
    
    Starts up a rocket-learn learner process, which ingests incoming data, updates parameters
    based on results, and sends updated model parameters out to the workers
    
    """
    #os.environ["WANDB_MODE"]="offline"

    # ROCKET-LEARN USES WANDB WHICH REQUIRES A LOGIN TO USE. YOU CAN SET AN ENVIRONMENTAL VARIABLE
    # OR HARDCODE IT IF YOU ARE NOT SHARING YOUR SOURCE FILES
    wandb.login(key=pickleData["WANDB_KEY"])
    logger = wandb.init(project="Variant", entity=pickleData["ENTITY"])
    logger.name = "Variant"
    print("Wandb init")

    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    redis = Redis(password=pickleData["REDIS"])
    print("Redis init")

    # THE ROLLOUT GENERATOR CAPTURES INCOMING DATA THROUGH REDIS AND PASSES IT TO THE LEARNER.
    # -save_every SPECIFIES HOW OFTEN OLD VERSIONS ARE SAVED TO REDIS. THESE ARE USED FOR TRUESKILL
    # COMPARISON AND TRAINING AGAINST PREVIOUS VERSIONS
    # -clear DELETE REDIS ENTRIES WHEN STARTING UP (SET TO FALSE TO CONTINUE WITH OLD AGENTS)
    rollout_gen = RedisRolloutGenerator(redis, ExpandAdvancedObs, rewards, DiscreteAction,
                                        logger=logger,
                                        save_every=1,
                                        clear=False)
    print("Rollout init")

    # ROCKET-LEARN EXPECTS A SET OF DISTRIBUTIONS FOR EACH ACTION FROM THE NETWORK, NOT
    # THE ACTIONS THEMSELVES. SEE network_setup.readme.txt FOR MORE INFORMATION
    split = (3, 3, 3, 3, 3, 2, 2, 2)
    total_output = sum(split)

    # TOTAL SIZE OF THE INPUT DATA
    state_dim = 231

    critic = Sequential(
        Linear(state_dim, 512),
        ReLU(),
        Linear(512, 512),
        ReLU(),
        Linear(512, 1)
    )

    actor = DiscretePolicy(Sequential(
        Linear(state_dim, 512),
        ReLU(),
        Linear(512, 512),
        ReLU(),
        Linear(512, total_output),
        SplitLayer(splits=split)
    ), split)

    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": 5e-5},
        {"params": critic.parameters(), "lr": 5e-5}
    ])

    # PPO REQUIRES AN ACTOR/CRITIC AGENT
    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=0.01,
        n_steps=1_000_000,
        batch_size=20_000,
        minibatch_size=10_000,
        epochs=10,
        gamma=599/600,
        clip_range=0.2,
        gae_lambda=0.95,
        vf_coef=1,
        max_grad_norm=0.5,
        logger=logger,
        device="cuda"
    )

    print("PPO init")
    # BEGIN TRAINING. IT WILL CONTINUE UNTIL MANUALLY STOPPED
    # -iterations_per_save SPECIFIES HOW OFTEN CHECKPOINTS ARE SAVED
    # -save_dir SPECIFIES WHERE
    send_messages.put(1)
    alg.run(iterations_per_save=1, save_dir=save_dir)

def startTraining(send_messages: multiprocessing.Queue, model_args: List):
    global paging, wait_time, total_num_instances, run_learner, run_workers
    name = model_args[0]
    num_instances = model_args[1]

    team_size = 3
    if run_workers:
        print(">>>Wait time:        ", wait_time)
        print(">>># of instances:   ", num_instances)
        print(">>># of trainers:    ", 2 * team_size * num_instances)
        print(">>>Paging:           ", paging)
    attackRewards = CombinedReward(
        (
            VelocityPlayerToBallReward(),
            LiuDistancePlayerToBallReward(),
            RewardIfTouchedLast(LiuDistanceBallToGoalReward()),
            RewardIfTouchedLast(VelocityBallToGoalReward()),
            RewardIfClosestToBall(AlignBallGoal(0,1), True),
        ),
        (2.0, 0.2, 1.0, 1.0, 0.8))

    defendRewards = CombinedReward(
        (
            VelocityPlayerToBallReward(),
            LiuDistancePlayerToBallReward(),
            RewardIfTouchedLast(LiuDistanceBallToGoalReward()),
            RewardIfTouchedLast(VelocityBallToGoalReward()),
            AlignBallGoal(1,0)
        ),
        (1.0, 0.2, 1.0, 1.0, 1.5))

    lastManRewards = CombinedReward(
        (
            RewardIfTouchedLast(VelocityBallToGoalReward()),
            AlignBallGoal(1,0),
            LiuDistancePlayerToGoalReward(),
            ConstantReward()
        ),
        (2.0, 1.0, 0.6, 0.2))

    kickoffRewards = CombinedReward(
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
        (2.0, 1.0, 1.5, 1.0, 0.4))

    def get_base_match():
        return Match(
            team_size=team_size,
            game_speed=100,
            self_play=True,
            obs_builder=ExpandAdvancedObs(3),
            action_parser=DiscreteAction(),
            reward_function= CombinedReward((), ()),
            terminal_conditions = [NoTouchTimeoutCondition(20000), GoalScoredCondition()],
            state_setter = RandomState()  # Resets to random
        )

    def get_match():
        match: Match = get_base_match()
        match._reward_fn = CombinedReward(
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
            (0.14, 0.25, 0.32, 0.19, 0.13, 3.33, 16.66, 4.36, 0.23, 0.22, 0.35, 1.06, 71.35))
        match._terminal_conditions = [NoTouchTimeoutCondition(20000), GoalScoredCondition()]
        match._state_setter = RandomState()  # Resets to random
        return match

    def get_kickoff():
        match: Match = get_base_match()
        match._reward_fn = CombinedReward(
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
            (0.08, 0.52, 0.82, 27.56, 108.22, 41.05, 0.2, 0.1, 1.0, 3.81, 319.16))
        match._terminal_conditions = [TimeoutCondition(2000), GoalScoredCondition()]
        match._state_setter = ModifiedState()  # Resets to kickoff position
        return match
    
    if name == "kickoff":
        match = get_kickoff
    else:
        match = get_match
    if run_learner:
        print(">>>Starting learner")
        receive_messages = multiprocessing.Queue()
        learner = multiprocessing.Process(target=runLearner, args=[receive_messages, data_location + "models/" + name, match()._reward_fn])
        learner.start()
        #Give it time to start
        while receive_messages.qsize() == 0:
            sleep(1)
        receive_messages.get()
        if run_workers:
            sleep(15)
    else:
        print(">>>Skipping learner")
    send_messages.put(1)
    sleep(1)
    while not send_messages.empty():
        sleep(1)
    if run_workers:
        workers: List[multiprocessing.Process]= []
        workersPID: List[int]= []
        for i in range(num_instances):
            print(">>>Starting worker:", i + 1)
            worker = multiprocessing.Process(target=runWorker, args=[match()])
            worker.start()
            workers.append(worker)
            while send_messages.empty():
                sleep(0.5)
            workersPID.append(send_messages.get())
            sleep(wait_time)
    send_messages.put(2)
    send_messages.close()
    try:
        while True:
            if run_learner and not learner.is_alive():
                print(">>>Restarting learner")
                learner = multiprocessing.Process(target=runLearner, args=[receive_messages, data_location + "models/" + name, match()._reward_fn])
                learner.start()
                while receive_messages.qsize() == 0:
                    sleep(1)
                receive_messages.get()
            if run_workers:
                for i in range(num_instances):
                    if workers[i].is_alive():
                        sleep(1)
                    else:
                        print(">>>Restarting worker")
                        old_pid =  workersPID[i]
                        workersPIDCopy = workersPID.copy()
                        workersPIDCopy.remove(old_pid)
                        killRL([old_pid])
                        workers[i] = multiprocessing.Process(target=runWorker, args=[match()])
                        workers[i].start()
                        while workersPID[i] == old_pid:
                            sleep(1)
                            curr_PIDs = getRLInstances()
                            #Check for new instance
                            for pid in curr_PIDs:
                                if pid not in workersPID:
                                    new_instance = True
                                    print(">>>Instance found")
                                    workersPID[i] = pid
                                    break
                        sleep(INSTANCE_SETUP_TIME)
                        killRL(blacklist=workersPID)
                        minimiseRL([workersPID[i]])
            if not run_workers and not run_learner:
                print(">>>Nothing to open")
                sleep(100)

    except KeyboardInterrupt:
        pass
    #Wait for workers to die
    if run_workers:
        for worker in workers:
            worker.kill()
        for worker in workers:
            while worker.is_alive():
                sleep(1)
    #Wait for learner to die
    if run_learner:
        learner.terminate()
        while learner.is_alive():
            sleep(1)

def trainingMonitor(send_messages: multiprocessing.Queue, model_args):
    global wait_time, paging
    instances = model_args[1]
    done = False
    trainers_RLPIDs = []
    initial_RLPIDs = getRLInstances()
    print(">>Initial instances:", len(initial_RLPIDs))
    receive_messages = multiprocessing.Queue()
    trainer = multiprocessing.Process(target=startTraining, args=[receive_messages, model_args])
    trainer.start()
    if run_workers:
        try:
            count = 0
            #Wait until setup is printed
            while receive_messages.empty() and trainer.is_alive():
                sleep(0.5)
            if receive_messages.empty():
                exit()
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
                            receive_messages.put(pid)
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
                sleep(1)
                while not receive_messages.empty() and trainer.is_alive():
                    #wait for trainer to read pid
                    sleep(1)
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
                sleep(INSTANCE_SETUP_TIME + 10)
                minimiseRL(trainers_RLPIDs)
                #trainer exit process and restart
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(">>Error: with trainer parser")
            #Continue to wait for trainer to die, then exit
    #trainer will shut down and save, please wait
    #after it has closed, RL instances will be killed
    else:
        print(">>Not monitoring instances")
        while receive_messages.empty():
            sleep(1)
        receive_messages.get()
        while receive_messages.empty():
            sleep(1)
        receive_messages.get()
        send_messages.put(1)
        send_messages.put([])
        send_messages.close()
    while trainer.is_alive():
        sleep(1)
    if run_workers:
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
    #start redis
    print(">Starting redis")
    os.system("wsl sudo redis-server /etc/redis/redis.conf --daemonize yes")
    r = Redis(host="127.0.0.1", password=pickleData["REDIS"])
    if clear_redis:
        r.delete("save-freq", "model-latest", "model-version", "qualities", "num-updates", "opponent-models", "worker-ids")
    try:
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
    except:
        pass
    finally:
        #kill redis
        r.shutdown()