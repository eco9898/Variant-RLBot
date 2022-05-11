from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.state_setters import DefaultState
import random


class ModifiedState(DefaultState):
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies state_wrapper values to emulate a randomly selected default kickoff.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        # possible kickoff indices are shuffled
        spawn_inds = [0, 1, 2, 3, 4]
        #30% chance to use chosen state all 3 at back
        if random.randint(1,100) > 50:
            spawn_inds = [2, 3, 4]

        random.shuffle(spawn_inds)
    
        blue_count = 0
        orange_count = 0
        for car in state_wrapper.cars:
            pos = [0,0,0]
            yaw = 0
            # team_num = 0 = blue team
            if car.team_num == 0:
                # select a unique spawn state from pre-determined values
                pos = self.SPAWN_BLUE_POS[spawn_inds[blue_count]]
                yaw = self.SPAWN_BLUE_YAW[spawn_inds[blue_count]]
                blue_count += 1
            # team_num = 1 = orange team
            elif car.team_num == 1:
                # select a unique spawn state from pre-determined values
                pos = self.SPAWN_ORANGE_POS[spawn_inds[orange_count]]
                yaw = self.SPAWN_ORANGE_YAW[spawn_inds[orange_count]]
                orange_count += 1
            # set car state values
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = 0.33
