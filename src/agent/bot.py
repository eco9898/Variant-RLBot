from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

import numpy as np
import pathlib, sys
from rlgym_compat import GameState


from agent import Agent
parent_directory = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(parent_directory)
from utils.advanced_padder import AdvancedObsPadder

class StrainRLGymBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        self.obs_builder = AdvancedObsPadder(3)
        self.agent = Agent()
        self.tick_skip = 12
        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.ticks = 0
        self.prev_time = 0
        self.observed = False
        self.acted = False
        self.current_obs = None
        self.team_size = 3
        print(f'{self.name} Ready - Index:', self.index)


    def initialize_agent(self):
        print(f'{self.name} Initialising agent:', self.index)
        # Initialize the rlgym GameState object now that the game is active and the info is available
        self.game_state = GameState(self.get_field_info())
        self.ticks = self.tick_skip
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.tick_multi = 120

    def ignore_players(self):
        """removes unexpected players from game_state"""
        opponents = [p for p in self.game_state.players if p.team_num != self.team]
        allies = [p for p in self.game_state.players if p.team_num == self.team and p.car_id != self.index]
        while len(allies) > self.team_size - 1:
            furthest_al = max(allies, key=lambda p: np.linalg.norm(self.game_state.ball.position - p.car_data.position))
            self.game_state.players.remove(furthest_al)
            allies.remove(furthest_al)
        while len(opponents) > self.team_size:
            furthest_op = max(opponents, key=lambda p: np.linalg.norm(self.game_state.ball.position - p.car_data.position))
            self.game_state.players.remove(furthest_op)
            opponents.remove(furthest_op)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time
        ticks_elapsed = self.ticks * self.tick_multi
        self.ticks += delta

        if not self.observed:
            self.game_state.decode(packet, ticks_elapsed)
            self.ignore_players()
            self.current_obs = self.obs_builder.build_obs(self.game_state.players[self.index], self.game_state, self.action)
            self.observed = True

        elif ticks_elapsed >= self.tick_skip-2:
            if not self.acted:
                self.action = self.agent.act(self.game_state.players[self.index], self.current_obs, self.game_state)
                self.update_controls(self.action)
                self.acted = True

        if ticks_elapsed >= self.tick_skip-1:
            self.ticks = 0
            self.observed = False
            self.acted = False

        return self.controls


    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0
        
if __name__ == "__main__":
    print("You're doing it wrong.")

