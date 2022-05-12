import sys, pathlib, pickle, torch, torch.jit, socket, os
from redis import Redis
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker
parent_directory = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_directory)
from utils.util_classes import get_kickoff, get_match

if __name__ == "__main__":
    print ("Worker instance started")
    os.system("title Variant_Worker")
    name = sys.argv[1]
    team_size = int(sys.argv[2])
    if name == "kickoff":
        match = get_kickoff(team_size)
    else:
        match = get_match(team_size)
    pickle_directory = parent_directory + "/trainer/"

    pickleData = pickle.load(open(pickle_directory + "pickleData.obj","rb"))
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
    RedisRolloutWorker(r, socket.gethostname(), match, 
        past_version_prob=0,#.2, 
        evaluation_prob=0,#.01,
        sigma_target=1,
        streamer_mode=False, 
        send_gamestates=False, 
        pretrained_agents=None, 
        human_agent=None,
        deterministic_old_prob=0#.5
    ).run()