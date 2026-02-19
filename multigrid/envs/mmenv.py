from __future__ import annotations

from multigrid.core.constants import Color, Direction, Type, COLOR_NAMES, DIR_TO_VEC
from multigrid.core.mission import MissionSpace, Mission
from multigrid.core.roomgrid import RoomGrid
from multigrid.core.world_object import Ball, Key, Box


class MMEnv(RoomGrid):

    def __init__(
        self,
        room_size: int = 8,
        numObjs: int = 2,
        max_steps: int | None = 100,
        joint_reward: bool = True,
        **kwargs,
    ):

        self.numObjs = numObjs
        self.obj_types = ["key", "ball", "box"]

        assert room_size >= 4
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[
                COLOR_NAMES,
                self.obj_types,
                COLOR_NAMES,
                self.obj_types,
            ],
        )
        super().__init__(
            room_size,
            num_rows=1,
            num_cols=1,
            max_steps=max_steps,
            joint_reward=joint_reward,
            success_termination_mode="any",
            failure_termination_mode="any",
            **kwargs,
        )

    @staticmethod
    def _gen_mission(
        move_color: str, move_type: str, target_color: str, target_type: str
    ):
        return f"put the {move_color} {move_type} near the {target_color} {target_type}"

    def _gen_grid(self, width, height):

        super()._gen_grid(width, height)

        types = ["key", "ball", "box"]

        objs = []
        objPos = []

        def near_obj(env, p1):
            for p2 in objPos:
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                if abs(dx) <= 1 and abs(dy) <= 1:
                    return True
            return False

        # Until we have generated all the objects
        while len(objs) < self.numObjs:
            objType = self._rand_elem(types)
            objColor = self._rand_elem(COLOR_NAMES)

            # If this object already exists, try again
            if (objType, objColor) in objs:
                continue

            if objType == "key":
                obj = Key(objColor)
            elif objType == "ball":
                obj = Ball(objColor)
            elif objType == "box":
                obj = Box(objColor)
            else:
                raise ValueError(
                    "{} object type given. Object type can only be of values key, ball and box.".format(
                        objType
                    )
                )

            pos = self.place_obj(obj, reject_fn=near_obj)

            objs.append((objType, objColor))
            objPos.append(pos)

        # Place agents in the left room
        for agent in self.agents:
            self.place_agent(agent, 0, 0)

        # Choose a random object to be moved
        objIdx = self._rand_int(0, len(objs))
        self.move_type, self.moveColor = objs[objIdx]
        self.move_pos = objPos[objIdx]

        # Choose a target object (to put the first object next to)
        while True:
            targetIdx = self._rand_int(0, len(objs))
            if targetIdx != objIdx:
                break
        self.target_type, self.target_color = objs[targetIdx]
        self.target_pos = objPos[targetIdx]

        self.mission = "put the {} {} near the {} {}".format(
            self.moveColor.name,
            self.move_type,
            self.target_color.name,
            self.target_type,
        )

        for agent in self.agents:
            agent.mission = Mission(self.mission)

    def step(self, actions):

        preCarrying = []
        for agent in self.agents:
            preCarrying.append(agent.carrying)

        obs, reward, terminated, truncated, info = super().step(actions)

        for agent_id, action in actions.items():
            agent = self.agents[agent_id]
            u, v = DIR_TO_VEC[agent.dir]
            ox, oy = (agent.pos[0] + u, agent.pos[1] + v)
            tx, ty = self.target_pos

            # If we picked up the wrong object, terminate the episode
            if action == self.actions.pickup and agent.carrying:
                if (
                    agent.carrying.type != self.move_type
                    or agent.carrying.color != self.moveColor
                ):
                    self.on_failure(agent, reward, terminated)

            # If successfully dropping an object near the target
            if action == self.actions.drop and preCarrying[agent_id]:
                if self.grid.get(ox, oy) is preCarrying[agent_id]:
                    if abs(ox - tx) <= 1 and abs(oy - ty) <= 1:
                        self.on_success(agent, reward, terminated)
                terminated = dict(enumerate(self.agent_states.terminated))

        return obs, reward, terminated, truncated, info
