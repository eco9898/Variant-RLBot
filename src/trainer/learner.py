import sys, pathlib, pickle, torch, torch.jit, wandb, os, glob
from redis import Redis
from time import sleep
from torch.nn import Linear, Sequential, ReLU
from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import SplitLayer
parent_directory = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_directory)
from utils.discrete_act import DiscreteAction
from utils.util_classes import get_kickoff, get_match, ExpandAdvancedObs

if __name__ == "__main__":
    print("Learner instance started")
    if len(sys.argv) == 4:
        save_dir = sys.argv[1]
        name = sys.argv[2]
        team_size = int(sys.argv[3])
    else:
        save_dir = "src/data/models/match"
        name = "match"
        team_size = 3
    pickle_directory = parent_directory + "/trainer/"
    pickleData = pickle.load(open(pickle_directory + "pickleData.obj","rb"))
    """
    
    Starts up a rocket-learn learner process, which ingests incoming data, updates parameters
    based on results, and sends updated model parameters out to the workers
    
    """
    #os.environ["WANDB_MODE"]="offline"

    config = {
        "actor_lr":5e-5,
        "critic_lr":5e-5,
        "n_steps":1_000_000,
        "batch_size":200_000,
        "minibatch_size":20_000,
        "epochs":30,
        "gamma":0.9975,
        "iterations_per_save":1,
        "ent_coef":0.01,
        "clip_range":0.2
    }

    # ROCKET-LEARN USES WANDB WHICH REQUIRES A LOGIN TO USE. YOU CAN SET AN ENVIRONMENTAL VARIABLE
    # OR HARDCODE IT IF YOU ARE NOT SHARING YOUR SOURCE FILES
    wandb_id = "1lummxlt" #resume run
    wandb.login(key=pickleData["WANDB_KEY"])
    logger = wandb.init(name = name, project="Variant", entity=pickleData["ENTITY"], id=wandb_id, config=config, resume=wandb_id is not None)
    print("Wandb init")

    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    try:
        redis = Redis(password=pickleData["REDIS"])
        redis.get("*")
    except:
        os.system("wsl sudo redis-server /etc/redis/redis.conf --daemonize yes")
        sleep(1)
        redis = Redis(password=pickleData["REDIS"])
    print("Redis init")

    if name == "kickoff":
        match = get_kickoff(team_size)
    else:
        match = get_match(team_size)
    rewards = match._reward_fn

    # THE ROLLOUT GENERATOR CAPTURES INCOMING DATA THROUGH REDIS AND PASSES IT TO THE LEARNER.
    # -save_every SPECIFIES HOW OFTEN OLD VERSIONS ARE SAVED TO REDIS. THESE ARE USED FOR TRUESKILL
    # COMPARISON AND TRAINING AGAINST PREVIOUS VERSIONS
    # -clear DELETE REDIS ENTRIES WHEN STARTING UP (SET TO FALSE TO CONTINUE WITH OLD AGENTS)
    rollout_gen = RedisRolloutGenerator(redis, ExpandAdvancedObs, rewards, DiscreteAction,
                                        logger=logger,
                                        save_every=config["iterations_per_save"],
                                        clear=wandb_id is None)
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
        {"params": actor.parameters(), "lr": config["actor_lr"]},
        {"params": critic.parameters(), "lr": config["critic_lr"]}
    ])

    # PPO REQUIRES AN ACTOR/CRITIC AGENT
    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=config["ent_coef"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        minibatch_size=config["minibatch_size"],
        epochs=config["epochs"],
        gamma=config["gamma"],
        clip_range=config["clip_range"],
        vf_coef=1,
        max_grad_norm=0.5,
        logger=logger,
        device="cuda"
    )
    if wandb_id is not None:
        checkpint_path = save_dir + "/**.pt"
        files = glob.glob(checkpint_path, recursive=True)
        newest_model = max(files, key=os.path.getctime)[0:-4]
        alg.load(newest_model)
        alg.agent.optimizer.param_groups[0]["lr"] = config["actor_lr"]
        alg.agent.optimizer.param_groups[1]["lr"] = config["critic_lr"]
    print("PPO init")
    # BEGIN TRAINING. IT WILL CONTINUE UNTIL MANUALLY STOPPED
    # -iterations_per_save SPECIFIES HOW OFTEN CHECKPOINTS ARE SAVED
    # -save_dir SPECIFIES WHERE
    alg.run(iterations_per_save=config["iterations_per_save"], save_dir=save_dir)