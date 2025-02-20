#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import math

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


def equally_spaced_points_on_circle(centers, r, num_points):
    """
    Generates equally spaced points on a circle using parametric equations.

    Args:
        centers: tensor of shape (batch_dim, 2) with the center of the circle.
        r: Radius of the circle.
        num_points: The number of equally spaced points to generate.

    Returns:
        A list of tuples, where each tuple is an (x, y) point on the circle.
    """

    points = []
    for env_index in range(centers.shape[0]):
        env_points = []
        for i in range(num_points):
            angle_degrees = (360.0 * i) / num_points  # Calculate angle for even spacing
            angle_radians = math.radians(angle_degrees)
            x = centers[env_index][0] + r * math.cos(angle_radians)
            y = centers[env_index][1] + r * math.sin(angle_radians)
            env_points.append((x, y))
        points.append(torch.tensor(env_points, device=centers.device, dtype=torch.float32))
    return torch.stack(points)


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.pop("n_agents", 4)
        self.share_reward = kwargs.pop("share_rew", False)
        self.penalise_by_time = kwargs.pop("penalise_by_time", False)
        self.landmark_radius = kwargs.pop("landmark_radius", 0.02)
        self.pos_range = kwargs.pop("pos_range", 1.0)
        self.circle_radius = kwargs.pop("circle_radius", 0.4)
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
                color=Color.BLUE,
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(n_food):
            landmark = Landmark(
                name=f"food_{i}",
                collide=False,
                shape=Sphere(radius=self.landmark_radius),
                color=Color.BLACK,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):

        centers = torch.zeros((self.world.batch_dim, self.world.dim_p),
                              device=self.world.device,
                              dtype=torch.float32).uniform_(-1 + self.circle_radius, 1 - self.circle_radius)

        landmarks = equally_spaced_points_on_circle(centers, self.circle_radius, len(self.world.landmarks))

        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros(
                    self.world.dim_p,
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0)
                ,
                batch_index=env_index,
            )

        for idx, landmark in enumerate(self.world.landmarks):
            if env_index is None:
                landmark.set_pos(
                    landmarks[:, idx, :],
                    batch_index=env_index,
                )
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

        is_first = agent == self.world.agents[0]
        if is_first:
            # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
            self.rews = torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
            )
            for single_agent in self.world.agents:
                for landmark in self.world.landmarks:
                    closest = torch.min(
                        torch.stack(
                            [
                                torch.linalg.vector_norm(
                                    a.state.pos - landmark.state.pos, dim=1
                                )
                                for a in self.world.agents
                            ],
                            dim=-1,
                        ),
                        dim=-1,
                    )[0]
                    self.rews -= closest

                if single_agent.collide:
                    for a in self.world.agents:
                        if a != single_agent:
                            self.rews[self.world.is_overlapping(a, single_agent)] -= 1

        return self.rews

    def observation(self, agent: Agent):
        # get positions of all landmarks in this agent's reference frame
        obs = dict()
        relative_landmark_pos = []
        for landmark in self.world.landmarks:  # world.entities:
            relative_landmark_pos.append(landmark.state.pos - agent.state.pos)
        obs["relative_landmark_pos"] = torch.cat(relative_landmark_pos, dim=-1)

        landmark_pos = []
        for landmark in self.world.landmarks:  # world.entities:
            landmark_pos.append(landmark.state.pos)
        obs["landmark_pos"] = torch.cat(landmark_pos, dim=-1)

        # distance to all other agents
        other_pos = []
        for other in self.world.agents:
            if other != agent:
                other_pos.append(other.state.pos - agent.state.pos)

        obs["other_pos"] = torch.cat(other_pos, dim=-1)
        obs["agent_pos"] = agent.state.pos
        obs["agent_vel"] = agent.state.vel

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
        n_agents=3,
        share_reward=False,
        penalise_by_tim=False,
    )
