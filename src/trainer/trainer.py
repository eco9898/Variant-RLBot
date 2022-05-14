import subprocess, sys, multiprocessing, pathlib, os, pickle, shutil
from datetime import datetime
from time import sleep, time
from typing import Dict
from redis import Redis
#Comment line 69-72 of wandb.sdk.internal.stats
parent_directory = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_directory)
from utils.util_classes import *

MAX_INSTANCES_NO_PAGING = 5
WAIT_TIME_NO_PAGING = 20
WAIT_TIME_PAGING = 35
INSTANCE_SETUP_TIME = 50 #for safety
PYTHON_EXE = os.getenv("localappdata") + "\\RLBotGUIX\\Python37\\python.exe"

total_num_instances = 3
run_learner = True
run_workers = True
#run_workers = not run_learner
kickoff_instances = total_num_instances // 3
match_instances = total_num_instances - kickoff_instances
models: List = [["kickoff", kickoff_instances], ["match", match_instances]]
models: List = [["match", total_num_instances]]
#models: List = [["kickoff", total_num_instances]]

data_location = parent_directory + "/data/"
trainer_directory = parent_directory + "/trainer/"

pickleData = pickle.load(open(trainer_directory + "pickleData.obj","rb"))

paging = False
if total_num_instances > MAX_INSTANCES_NO_PAGING:
    paging = True
wait_time=WAIT_TIME_NO_PAGING
if paging:
    wait_time=WAIT_TIME_PAGING

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

    if run_learner:
        print(">>>Starting learner")
        learner = subprocess.Popen([PYTHON_EXE, trainer_directory + "learner.py", data_location + "models/" + name, name, str(team_size)])#, creationflags=subprocess.CREATE_NEW_CONSOLE)
        if run_workers:
            sleep(30)
    else:
        print(">>>Skipping learner")
    send_messages.put(1)
    sleep(1)
    while not send_messages.empty():
        sleep(1)
    workers: List[subprocess.Popen]= []
    if run_workers:
        workersPID: List[int]= []
        os.makedirs(data_location + "worker-logs/" + name, exist_ok=True)
        for i in range(num_instances):
            print(">>>Starting worker:", i + 1)
            with open(data_location + "worker-logs/" + name + "/worker-" + str(datetime.now()).replace(":", "-") + ".txt", "wb") as f:
                worker = subprocess.Popen([PYTHON_EXE, trainer_directory + "worker.py", name, str(team_size)], stdout=f, stderr=subprocess.STDOUT)
            workers.append(worker)
            while send_messages.empty():
                sleep(0.5)
            workersPID.append(send_messages.get())
            sleep(wait_time)
    send_messages.put(2)
    send_messages.close()
    try:
        while True:
            if run_learner and learner.poll() != None:
                print(">>>Restarting learner")
                learner = subprocess.Popen([PYTHON_EXE, trainer_directory + "learner.py", data_location + "models/" + name, name, str(team_size)])#, creationflags=subprocess.CREATE_NEW_CONSOLE)
            if run_workers:
                for i in range(num_instances):
                    if workers[i].poll() == None:
                        sleep(1)
                    else:
                        print(">>>Restarting worker")
                        old_pid =  workersPID[i]
                        workersPIDCopy = workersPID.copy()
                        workersPIDCopy.remove(old_pid)
                        killRL([old_pid])
                        with open(data_location + "worker-logs/" + name + "/worker-" + str(datetime.now()).replace(":", "-") + ".txt", "wb") as f:
                            workers[i] = subprocess.Popen([PYTHON_EXE, trainer_directory + "worker.py", name, str(team_size)], stdout=f, stderr=subprocess.STDOUT)
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
                        getRLInstances() # kills EOSOverlay
            if not run_workers and not run_learner:
                print(">>>Nothing to open")
                sleep(100)

    except KeyboardInterrupt:
        pass
    #Wait for workers to die
    if run_workers:
        for worker in workers:
            worker.terminate()
        for worker in workers:
            while worker.poll() == None:
                sleep(1)
    #Wait for learner to die
    if run_learner:
        learner.terminate()
        while learner.poll() == None:
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
    #Clear worker logs
    shutil.rmtree(data_location + "worker-logs", ignore_errors=True)
    print(">Cleared worker logs")
    initial_instances = getRLInstances()
    try:
        messages: Dict[str, multiprocessing.Queue] = {}
        monitors: Dict[str, multiprocessing.Process] = {}
        model_instances: Dict[str, List[int]]  = {}
        models_used: Dict[str, List]  = {}
        all_instances = []
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
    except:
        print(">Exiting")
    #kill instances reported
    #killRL(all_instances)
    #Kill instances that weren't present before
    killRL(blacklist=initial_instances)
    #kill redis
    r.shutdown()