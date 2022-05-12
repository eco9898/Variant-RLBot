import sys, pathlib, pickle, torch, torch.jit, wandb
from redis import Redis
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
    save_dir = sys.argv[1]
    name = sys.argv[2]
    team_size = int(sys.argv[3])
    pickle_directory = parent_directory + "/trainer/"
    pickleData = pickle.load(open(pickle_directory + "pickleData.obj","rb"))
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
                                        save_every=10,
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
    alg.run(iterations_per_save=10, save_dir=save_dir)