import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self):
        world = World()

        self.num_predators = 4  # N
        self.num_preys = 9  # M
        self.capture_requirement = 2  # C
        self.capture_radius = 0.3
        self.reward_value = 1
        world.dim_c = 2
        world.collaborative = True

        world.agents = [Agent() for _ in range(self.num_predators)]
        for i, agent in enumerate(world.agents):
            agent.name = f"predator_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.accel = 3.0
            agent.max_speed = 1.0
            agent.color = np.array([0.35, 0.35, 0.85])

        world.landmarks = [Landmark() for _ in range(self.num_preys)]
        for i, prey in enumerate(world.landmarks):
            prey.name = f"prey_{i}"
            prey.collide = False
            prey.movable = False
            prey.size = 0.05
            prey.color = np.array([0.25, 0.25, 0.25])
            prey.captured = False

        self.reset_world(world)

        return world

    def reset_world(self, world):

        self.reward_value = 1
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1.5, 1.5, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        self.spawn_preys(world)

    def spawn_preys(self, world):

        prey_positions = [
            np.array([-0.9, 0.9]),  # top-left
            np.array([0.0, 0.9]),  # top-center
            np.array([0.9, 0.9]),  # top-right
            np.array([-0.9, 0.0]),  # middle-left
            np.array([0.0, 0.0]),  # center
            np.array([0.9, 0.0]),  # middle-right
            np.array([-0.9, -0.9]),  # bottom-left
            np.array([0.0, -0.9]),  # bottom-center
            np.array([0.9, -0.9]),  # bottom-right
        ]
        assert self.num_preys <= len(prey_positions)

        for i, prey in enumerate(world.landmarks):

            prey.state.p_pos = prey_positions[i].copy()
            prey.state.p_vel = np.zeros(world.dim_p)

            prey.captured = False
            prey.color = np.array([0.25, 0.25, 0.25])

    def predator_close_to_prey(self, predator, prey):

        delta = predator.state.p_pos - prey.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta)))

        return dist < self.capture_radius

    def reward(self, agent, world):

        rew = 0

        # check captures
        for prey in world.landmarks:

            # already captured
            if prey.captured:
                continue

            nearby_predators = 0
            for predator in world.agents:
                if self.predator_close_to_prey(predator, prey):
                    nearby_predators += 1

            # prey captured
            if nearby_predators >= self.capture_requirement:
                prey.captured = True
                prey.color = np.array([0.85, 0.25, 0.25])
                rew += self.reward_value

        all_captured = all(prey.captured for prey in world.landmarks)

        if all_captured:

            self.reward_value += 1
            self.spawn_preys(world)

        return rew

    def observation(self, agent, world):

        obs = []

        # own velocity and position
        obs.append(agent.state.p_vel)
        obs.append(agent.state.p_pos)

        # predator positions
        for other in world.agents:

            if other is agent:
                continue

            rel_pos = other.state.p_pos - agent.state.p_pos
            obs.append(rel_pos)

        # prey positions
        for prey in world.landmarks:

            rel_pos = prey.state.p_pos - agent.state.p_pos

            captured_flag = np.array([1.0 if prey.captured else 0.0])

            obs.append(rel_pos)
            obs.append(captured_flag)

        return np.concatenate(obs)
