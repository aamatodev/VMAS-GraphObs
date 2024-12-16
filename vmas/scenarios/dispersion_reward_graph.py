#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.pop("n_agents", 4)
        self.share_reward = kwargs.pop("share_rew", False)
        self.penalise_by_time = kwargs.pop("penalise_by_time", False)
        self.food_radius = kwargs.pop("food_radius", 0.05)
        self.pos_range = kwargs.pop("pos_range", 1.0)
        n_food = kwargs.pop("n_food", n_agents)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.pos_range,
            y_semidim=self.pos_range,
        )
        # Add agents
        for i in range(n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=False,
                shape=Sphere(radius=0.035),
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(n_food):
            food = Landmark(
                name=f"food_{i}",
                collide=False,
                shape=Sphere(radius=self.food_radius),
                color=Color.GREEN,
            )
            world.add_landmark(food)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros(
                    self.world.dim_p,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                batch_index=env_index,
            )
        for landmark in self.world.landmarks:
            landmark.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -self.pos_range,
                    self.pos_range,
                ),
                batch_index=env_index,
            )
            if env_index is None:
                landmark.eaten = torch.full(
                    (self.world.batch_dim,), False, device=self.world.device
                )
                landmark.just_eaten = torch.full(
                    (self.world.batch_dim,), False, device=self.world.device
                )
                landmark.reset_render()
            else:
                landmark.eaten[env_index] = False
                landmark.just_eaten[env_index] = False
                landmark.is_rendering[env_index] = True

    def reward(self, agent: Agent):

        closer_nodes = []
        for landmark in self.world.landmarks:
            current_land = torch.min(torch.stack([torch.linalg.vector_norm(a.state.pos - landmark.state.pos, dim=1)
                                                  for a in self.world.agents], dim=1), dim=1).values
            closer_nodes.append(current_land)

        rews = 2 - torch.sum(torch.stack(closer_nodes), dim=-2)
        # if self.penalise_by_time:
        #     rews[rews == 0] = -0.01

        return rews

    def observation(self, agent: Agent):

        obs = dict()
        obs["agent_pos"] = agent.state.pos
        obs["agent_index"] = torch.tensor(int(agent.name.split("_")[1])).unsqueeze(-1).repeat(agent.batch_dim).reshape(-1, 1).to(self.world.device)
        obs["agent_vel"] = agent.state.vel
        obs["relative_landmark_pos"] = dict()

        relative_landmark_pos = []
        for landmark in self.world.landmarks:
            relative_landmark_pos.append(
                torch.cat(
                    [
                        landmark.state.pos - agent.state.pos,
                        landmark.eaten.to(torch.int).unsqueeze(-1),
                    ],
                    dim=-1,
                )
            )
        relative_landmark_pos = torch.cat(relative_landmark_pos, dim=-1)
        obs["relative_landmark_pos"] = relative_landmark_pos

        # agent_and_landmark_pos = [agent.state.pos]
        agent_and_landmark_pos = []
        for idx, landmark in enumerate(self.world.landmarks):
            agent_and_landmark_pos.append(landmark.state.pos)

        landmark_pos = torch.cat(agent_and_landmark_pos, dim=-1)

        obs["landmark_pos"] = landmark_pos

        landmark_eaten = [torch.full((landmark.eaten.shape[0], 1), -1).to(self.world.device)]
        for idx, landmark in enumerate(self.world.landmarks):
            landmark_eaten.append(landmark.eaten.to(torch.int).unsqueeze(-1))

        landmark_eaten = torch.cat(landmark_eaten, dim=-1)

        obs["landmark_eaten"] = landmark_eaten

        return obs

    def info(self, agent: Agent):
        info = dict()

        out = []
        for idx, landmark in enumerate(self.world.landmarks):
            out.append(landmark.state.pos)
        out = torch.stack(out, dim=1)

        info["landmark_pos"] = out
        return info

    def done(self):
        return torch.all(
            torch.stack(
                [landmark.eaten for landmark in self.world.landmarks],
                dim=1,
            ),
            dim=-1,
        )


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        n_agents=4,
        share_reward=False,
        penalise_by_tim=False,
    )
