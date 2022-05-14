from time import sleep
import numpy as np
import psutil, pathlib, sys
import win32gui, win32con, win32process, os, signal
from rlgym.envs import Match
from rlgym.utils.common_values import ORANGE_TEAM, BLUE_TEAM, ORANGE_GOAL_BACK, BLUE_GOAL_BACK, ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER, BACK_WALL_Y, CAR_MAX_SPEED, BALL_MAX_SPEED
from rlgym.utils.reward_functions.common_rewards.conditional_rewards import ConditionalRewardFunction
from rlgym.utils import RewardFunction, math
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.state_setters import RandomState
from rlgym.utils.terminal_conditions.common_conditions import *
from rlgym.utils.reward_functions.common_rewards.misc_rewards import *
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import *
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import *
from rlgym.utils.reward_functions.common_rewards.conditional_rewards import *
from rlgym.utils.reward_functions import CombinedReward
from typing import List, Tuple, Optional, Any

parent_directory = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_directory)
from utils.advanced_padder import AdvancedObsPadder
from utils.discrete_act import DiscreteAction
from utils.modified_states import *

# ROCKET-LEARN ALWAYS EXPECTS A BATCH DIMENSION IN THE BUILT OBSERVATION
class ExpandAdvancedObs(AdvancedObsPadder):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return np.expand_dims(obs, 0)

def getRLInstances():
    '''
    Get a list of all the PIDs of a all the running process whose name is RocketLeague.exe
    '''
    #minimisedPIDs = []
    #while len(minimisedPIDs < instanceCount):
    listOfPIDs = []
    #Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'name'])
            # Check if process name contains the given name string.
            if "RocketLeague.exe".lower() == pinfo['name'].lower() :
                listOfPIDs.append(pinfo["pid"])
            if "EOSOverlayRenderer-Win64-Shipping.exe".lower() == pinfo['name'].lower():
                #print ("Killed EOS:", pinfo["pid"])
                os.kill(pinfo["pid"], signal.SIGTERM)
        except:# (psutil.NoSuchProcess, psutil.AccessDenied , psutil.ZombieProcess):
            pass
    return listOfPIDs
        #for proc in listOfProcessObjects:
        #    if not proc in minimisedPIDs:
        #        win32gui.FindWindow("RocketLeague.exe", None)

toplist = []
winlist = []
def enum_callback(hwnd, results):
    winlist.append((hwnd, win32gui.GetWindowText(hwnd)))

def minimiseRL(targets: List = []):
    toplist.clear()
    winlist.clear()
    win32gui.EnumWindows(enum_callback, toplist)
    Rl = [(hwnd, title) for hwnd, title in winlist if 'Rocket League (64-bit, DX11, Cooked)'.lower() in title.lower()]
    #worker = [(hwnd, title) for hwnd, title in winlist if '_Worker'.lower() in title.lower()]
    # just grab the first window that matches
    for win in Rl:
        # use the window handle to set focus
        #win32gui.SetForegroundWindow(win[0])
        pid = win32process.GetWindowThreadProcessId(win[0])[1]
        #print("Found", pid)
        #print("targets:", targets)
        if pid in targets:
            print ("Minimising RL:", pid)
            win32gui.ShowWindow(win[0], win32con.SW_MINIMIZE)
            sleep(1)
    #for win in worker:
    #    pid = win32process.GetWindowThreadProcessId(win[0])[1]
    #    print ("Minimising Worker:", pid)
    #    win32gui.ShowWindow(win[0], win32con.SW_MINIMIZE)
    #    sleep(0.2)


def killRL(targets: List = [], blacklist: List = []):
    PIDs = getRLInstances()
    while len(PIDs) > 0:
        pid = PIDs.pop()
        if len(blacklist) > 0:
            if pid in blacklist:
                continue
            else:
                targets.append(pid)
        if pid in targets:
            print("xxKilling RL instance", pid)
            try:
                os.kill(pid, signal.SIGTERM)
            except:
                print("xxFailed")

class TeamSpacingReward(RewardFunction):
    def __init__(self, min_spacing: float = 1000) -> None:
        super().__init__()
        self.min_spacing = min_spacing

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        for p in state.players:
            if p.team_num == player.team_num and p.car_id != player.car_id and not player.is_demoed and not p.is_demoed:
                separation = np.linalg.norm(player.car_data.position - p.car_data.position)
                if separation < self.min_spacing:
                    reward -= 1-(separation / self.min_spacing)
        return reward

class FlipReward(RewardFunction): #multiply by speed?
    def __init__(self) -> None:
        super().__init__()
        self.rewarded = False

    def reset(self, initial_state: GameState):
        self.rewarded = False

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if not player.has_flip and not self.rewarded:
            self.rewarded = True
            return 1
        elif player.has_flip:
            self.rewarded = False
        elif not player.on_ground:
            return 0.1
        return 0

class pickupBoost(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self.lastBoost = 100

    def reset(self, initial_state: GameState):
        self.lastBoost = 100

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.boost_amount > self.lastBoost + 12:
            self.lastBoost = player.boost_amount
            return 1
        elif player.boost_amount > self.lastBoost:
            self.lastBoost = player.boost_amount
            return 0.2
        self.lastBoost = player.boost_amount
        return 0

class useBoost(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self.lastBoost = 0

    def reset(self, initial_state: GameState):
        self.lastBoost = 0

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.boost_amount < self.lastBoost:
            reward = (self.lastBoost - player.boost_amount)/100
            self.lastBoost = player.boost_amount
            return reward
        self.lastBoost = player.boost_amount
        return 0

class LiuDistancePlayerToGoalReward(RewardFunction):
    def __init__(self, own_goal=True):
        super().__init__()
        self.own_goal = own_goal

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_CENTER)
        else:
            objective = np.array(BLUE_GOAL_CENTER)

        # Compensate for moving objective to back of net
        dist = np.linalg.norm(player.car_data.position - objective)
        return np.exp(-0.5 * dist / CAR_MAX_SPEED)

def playerCrossedHalfWay(player: PlayerData):
    playerY = player.car_data.position[1]
    if player.team_num == ORANGE_TEAM:
        playerY = -playerY
    return playerY > 0

def ballCrossedHalfWay(player: PlayerData, state: GameState):
    ballY = state.ball.position[1]
    if player.team_num == ORANGE_TEAM:
        ballY = -ballY
    return ballY > 0

def playerAtGoal(player: PlayerData, own_goal: bool=False):
    playerY = player.car_data.position[1]
    if player.team_num == ORANGE_TEAM:
        playerY = -playerY
    if own_goal:
        playerY = -playerY
    return playerY > BACK_WALL_Y/2

def ballAtGoal(player: PlayerData, state: GameState, own_goal: bool=False):
    ballY = state.ball.position[1]
    if player.team_num == ORANGE_TEAM:
        ballY = -ballY
    if own_goal:
        ballY = -ballY
    return ballY > BACK_WALL_Y/2

def playerApproachingGoal(player: PlayerData, own_goal: bool=False):
    if player.team_num == BLUE_TEAM and not own_goal \
            or player.team_num == ORANGE_TEAM and own_goal:
        objective = np.array(ORANGE_GOAL_BACK)
    else:
        objective = np.array(BLUE_GOAL_BACK)

    vel = player.car_data.linear_velocity
    pos_diff = objective - player.car_data.position
    # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
    # Max value should be max_speed / ball_radius = 2300 / 94 = 24.5
    # Used to guide the agent towards the ball
    return math.scalar_projection(vel, pos_diff) > 0
    """else:
        # Regular component velocity
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        vel /= CAR_MAX_SPEED
        return float(np.dot(norm_pos_diff, vel))"""

def ballApproachingGoal(state: GameState, team: int):
    if team == ORANGE_TEAM:
        objective = np.array(ORANGE_GOAL_BACK)
    else:
        objective = np.array(BLUE_GOAL_BACK)

    vel = state.ball.linear_velocity
    pos_diff = objective - state.ball.position
    # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
    # Max value should be max_speed / ball_radius = 2300 / 94 = 24.5
    # Used to guide the agent towards the ball
    return math.scalar_projection(vel, pos_diff) > 0
    """else:
        # Regular component velocity
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        vel /= BALL_MAX_SPEED
        return float(np.dot(norm_pos_diff, vel))"""

def isAttacking(player: PlayerData, state: GameState):
    if player.is_demoed:
        return False
    return playerCrossedHalfWay(player) and (playerApproachingGoal(player) or ballApproachingGoal(state, player.team_num)
            or ballAtGoal(player, state))


class RewardIfAttacking(ConditionalRewardFunction):
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        return isAttacking(player, state)

def defending(player: PlayerData, state: GameState):
    if player.is_demoed:
        return False
    return not playerCrossedHalfWay(player) and (not (playerApproachingGoal(player) and ballApproachingGoal(state, player.team_num))
            or ballAtGoal(player, state, True))

class RewardIfDefending(ConditionalRewardFunction):
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        return defending(player, state)

class RewardIfLastMan(ConditionalRewardFunction):
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        if not ballAtGoal(player, state) and (playerCrossedHalfWay(player) or (ballCrossedHalfWay(player, state) and not ballApproachingGoal(state, player.team_num))):
            return False
        teammates = 0
        teammatesAttacking = 0
        for p in state.players:
            if p.team_num == player.team_num and p.car_id != player.car_id and not p.is_demoed:
                teammates += 1
                if isAttacking(p, state):
                    teammatesAttacking += 1
        return teammatesAttacking == teammates - 1

def isKickoff(state: GameState) -> bool:
    return state.ball.position[0] == 0 and state.ball.position[1] == 0 and np.linalg.norm(state.ball.linear_velocity) == 0

class RewardIfKickoff(ConditionalRewardFunction):
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        return isKickoff(state)

class RewardIfFurthestFromBall(ConditionalRewardFunction):
    def __init__(self, reward_func: RewardFunction, team_only=True):
        super().__init__(reward_func)
        self.team_only = team_only

    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        dist = np.linalg.norm(player.car_data.position - state.ball.position)
        if len(state.players) > 2:
            for player2 in state.players:
                if not self.team_only or player2.team_num == player.team_num:
                    dist2 = np.linalg.norm(player2.car_data.position - state.ball.position)
                    if dist2 > dist:
                        return False
            return True
        return False

class RewardIfMidFromBall(ConditionalRewardFunction):
    def __init__(self, reward_func: RewardFunction, team_only=True):
        super().__init__(reward_func)
        self.team_only = team_only

    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        dist = np.linalg.norm(player.car_data.position - state.ball.position)
        min = max = dist
        for player2 in state.players:
            if not self.team_only or player2.team_num == player.team_num:
                dist2 = np.linalg.norm(player2.car_data.position - state.ball.position)
                if dist2 > max:
                    max = dist2
                if dist2 < min:
                    min = dist2
        return dist != max and dist != min

class JumpTouchReward(RewardFunction):
    """
    a ball touch reward that only triggers when the agent's wheels aren't in contact with the floor
    adjust minimum ball height required for reward with 'min_height' as well as reward scaling with 'exp'
    """
    
    def __init__(self, min_height=92, exp=0.2):
        self.min_height = min_height
        self.exp = exp

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and not player.on_ground and state.ball.position[2] >= self.min_height:
            return ((state.ball.position[2] - 92) ** self.exp)-1

        return 0

class RLCombinedLogReward(CombinedReward):

    def __init__(
            self,
            redis: Any,
            logger: Any,
            reward_names: List[str],
            reward_functions: Tuple[RewardFunction, ...],
            reward_weights: Optional[Tuple[float, ...]] = None,
            name_prefix: str = "",
            repeated_calls: int = 1,
            relative_count: bool = True
    ):
        """
        Creates the combined reward using multiple rewards, and a potential set
        of weights for each reward. Will also log the weighted rewards to
        the WandB logger.
        :param redis: The redis instance
        :param logger: WandB logger
        :param reward_names: The list of reward names
        :param reward_functions: Each individual reward function.
        :param reward_weights: The weights for each reward.
        """
        super().__init__(reward_functions, reward_weights)

        self.redis = redis
        self.reward_names = reward_names
        self.logger = logger

        # Initiates the array that will store the episode totals
        self.returns = [0] * len(self.reward_functions)
        self.rewards_given = 0
        self.name_prefix = name_prefix
        self.upd_num = 0
        self.repeated_calls = repeated_calls
        self.relative_count = relative_count

    def reset(self, initial_state: GameState):
        self.returns = [0] * len(self.reward_functions)
        self.rewards_given = 0
        super().reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rewards = [
            func.get_reward(player, state, previous_action)
            for func in self.reward_functions
        ]

        for n, value in enumerate(rewards):
            self.returns[n] += value * self.reward_weights[n] # store the rewards
        self.rewards_given += 1

        return float(np.dot(self.reward_weights, rewards))

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rewards = [
            func.get_final_reward(player, state, previous_action)
            for func in self.reward_functions
        ]

        for n, value in enumerate(rewards):
            self.returns[n] += value * self.reward_weights[n] # store the rewards
        reward_dict = dict()
        for n, name in enumerate(self.reward_names):
            reward_dict[self.name_prefix + "_" + name] = self.returns[n]
            reward_dict[self.name_prefix + "_" + name + "_avg"] = self.returns[n] / self.rewards_given
        reward_dict[self.name_prefix + "_count"] = self.rewards_given
        if self.upd_num % self.repeated_calls == 0:
            #for some reason it is called 6 times per run
            if self.relative_count:
                self.logger.log(reward_dict, step=int(self.upd_num/self.repeated_calls), commit=False)
            else:
                self.logger.log(reward_dict, step=int(self.redis.get("num-updates")), commit=False)
            print("Rewards given:", self.rewards_given)
            print("Update number:", self.upd_num)
            print(self.returns)
            print(reward_dict)
        self.upd_num += 1
        return float(np.dot(self.reward_weights, rewards))

attackRewards = [
    ["AlignGoalOff", "BallToGoalDis", "BallToGoalVel", 'DisToBall', "Vel"],
    (
        RewardIfClosestToBall(AlignBallGoal(0,1), True),
        RewardIfTouchedLast(LiuDistanceBallToGoalReward()),
        RewardIfTouchedLast(VelocityBallToGoalReward()),
        LiuDistancePlayerToBallReward(),
        VelocityPlayerToBallReward(),
    ),
    (0.00014702504, 0.05220883534, 1.00000000000, 0.00001697613, 0.00145454545)]

defendRewards =[
    ["Vel", 'DisToBall', "BallToGoalDis", "BallToGoalVel", "AlignGoalDef"],
    (
        VelocityPlayerToBallReward(),
        LiuDistancePlayerToBallReward(),
        RewardIfTouchedLast(LiuDistanceBallToGoalReward()),
        RewardIfTouchedLast(VelocityBallToGoalReward()),
        AlignBallGoal(1,0)
    ),
    (1.0, 0.2, 1.0, 1.0, 1.5)]

lastManRewards = [
    ["BallToGoalVel", "AlignGoalDef", "DisToGoal", 'Constant'],
    (
        RewardIfTouchedLast(VelocityBallToGoalReward()),
        AlignBallGoal(1,0),
        LiuDistancePlayerToGoalReward(),
        ConstantReward()
    ),
    (2.0, 1.0, 0.6, 0.2)]

kickoffRewards = [
    ["Combination", "Def", "Lastman", "Spacing", "PickupBoost"],
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
    (0.15, 52.21, 1000.00, 0.02, 1.45)]

def get_base_match(team_size):
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

def get_match(team_size, redis=None, wandb=None):
    match: Match = get_base_match(team_size)
    if redis != None and wandb != None: 
        match._reward_fn = RLCombinedLogReward(
            redis, wandb,
            [
                "Att",
                "Event",
                #"Flip",
                "Jumptouch",
                "MatchFaceball",
                #"Pickupboost",
                #"Saveboost",
                #"Spacing",
                "Touch",
                #"Useboost",
                "Vel"
            ],
            (
                RLCombinedLogReward(redis, wandb, attackRewards[0], attackRewards[1], attackRewards[2], "Att", 6), #RewardIfAttacking()
                EventReward(
                    team_goal=100.0,
                    goal=10.0 * team_size,
                    concede=-100.0 + (10.0 * team_size),
                    shot=10.0,
                    save=30.0,
                    demo=12.0,
                ),
                #FlipReward(),
                JumpTouchReward(),
                FaceBallReward(),
                #pickupBoost(),
                #SaveBoostReward(),
                #TeamSpacingReward(1500),
                TouchBallReward(1.2),
                #useBoost(),
                VelocityReward()
            ),
            #(0.14, 0.25, 0.32, 0.19, 0.13, 3.33, 16.66, 4.36, 0.23, 0.22, 0.35, 1.06, 71.35))
            #(0.14, 0.25, 0.32, 0.19, 0.13, 3.33, 16.66, 4.36, 0.23, 0.01, 0.35, 1.06, 71.35)) V0-57
            #(0.08, 20.41, 0.03, 380.38, 0.55, 0.77, 0.32, 0.37, 70.12, 50.45, 0.14),
            (
                0.00010157155,
                0.00308826355,
                #0.00000099052,
                0.03008629028,
                0.00029356597,
                #0.00005005627,
                #0.00002602214,
                #0.00005849864,
                0.00469220744,
                #0.00348264214,
                0.00002059851

            ), "Match", 6)
    else:
        match._reward_fn = CombinedReward(
            (
                CombinedReward(attackRewards[1], attackRewards[2]), #RewardIfAttacking()
                EventReward(
                    team_goal=100.0,
                    goal=10.0 * team_size,
                    concede=-100.0 + (10.0 * team_size),
                    shot=10.0,
                    save=30.0,
                    demo=12.0,
                ),
                FlipReward(),
                JumpTouchReward(),
                FaceBallReward(),
                pickupBoost(),
                SaveBoostReward(),
                TeamSpacingReward(1500),
                TouchBallReward(1.2),
                useBoost(),
                VelocityReward()
            ),
            #(0.14, 0.25, 0.32, 0.19, 0.13, 3.33, 16.66, 4.36, 0.23, 0.22, 0.35, 1.06, 71.35))
            #(0.14, 0.25, 0.32, 0.19, 0.13, 3.33, 16.66, 4.36, 0.23, 0.01, 0.35, 1.06, 71.35)) V0-57
            (0.20, 15.0, 0.0, 10.0, 0.1, 0.2, 1.0, 0.2, 4.0, 2.0, 0.2))
    match._terminal_conditions = [NoTouchTimeoutCondition(20000), GoalScoredCondition()]
    match._state_setter = RandomState()  # Resets to random
    return match

def get_kickoff(team_size, redis=None, wandb=None):
    match: Match = get_base_match(team_size)
    if redis != None and wandb != None: 
        match._reward_fn = RLCombinedLogReward(
            redis, wandb,
            ["Kickoff", "Vel", 'KickFaceball', "Event", "Jumptouch", "Touch", "Spacing", "Flip", "Saveboost", "Pickupboost", "Useboost"],
            (
                RLCombinedLogReward(redis, wandb, kickoffRewards[0], kickoffRewards[1], kickoffRewards[2], "KO"),
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
            "Kick")
    else:
        match._reward_fn = CombinedReward(
            (
                CombinedReward(kickoffRewards[1], kickoffRewards[2]),
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
