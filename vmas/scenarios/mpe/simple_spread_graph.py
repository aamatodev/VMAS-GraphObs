#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from vmas.simulator.heuristic_policy import BaseHeuristicPolicy

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        num_agents = kwargs.pop("n_agents", 3)
        obs_agents = kwargs.pop("obs_agents", True)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        x_semidim = kwargs.pop("x_semidim", None)
        y_semidim = kwargs.pop("y_semidim", None)
        self.obs_agents = obs_agents

        world = World(batch_dim=batch_dim, device=device, x_semidim=x_semidim, y_semidim=y_semidim)
        # set any world properties first
        num_landmarks = num_agents
        # Add agents
        for i in range(num_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                shape=Sphere(radius=0.05),
                color=Color.BLUE,
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=False,
                color=Color.BLACK,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
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
                    -1.0,
                    1.0,
                ),
                batch_index=env_index,
            )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
            self.rew = torch.zeros(
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
                    self.rew -= closest

                if single_agent.collide:
                    for a in self.world.agents:
                        if a != single_agent:
                            self.rew[self.world.is_overlapping(a, single_agent)] -= 1

        return self.rew

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

        obs["other_pos"] = torch.cat(other_pos, dim=-1) if self.obs_agents else None
        obs["agent_pos"] = agent.state.pos
        obs["agent_vel"] = agent.state.vel

        return obs


def assign_agents_to_landmarks(cost_matrix):
    # Use the Hungarian algorithm (linear_sum_assignment)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Return the optimal assignments
    assignments = list(zip(row_indices, col_indices))
    return assignments


class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, obs: torch.Tensor, u_range: float):

        agent_pos = obs["agent_pos"].view(-1, 1, 2)
        landmark_pos = obs["relative_landmark_pos"].view(-1, 3, 2)
        other_pos = obs["other_pos"].view(-1, 2, 2)

        batch_size = agent_pos.shape[0]

        agents_pos = torch.cat([agent_pos, other_pos], dim=1)

        distances_to_landmark = []
        for env in range(batch_size):
            m = []
            for i in range(agents_pos.shape[1]):
                # Compute the distance from agent i to all landmarks
                m.append(torch.linalg.vector_norm(agents_pos[env][i] - landmark_pos[env], dim=1))
            distances_to_landmark.append(torch.stack(m, dim=0))

        # Compute the direction to the closest landmark
        assignments = []
        for env in range(batch_size):
            assignments.append(assign_agents_to_landmarks(np.array(distances_to_landmark[env])))

        selected_landmark = []
        for env in range(batch_size):
            selected_landmark.append(landmark_pos[env][assignments[env][0][1]])

        direction_to_landmark = torch.from_numpy(np.array(selected_landmark))
        # Normalize the direction
        direction_to_landmark /= torch.linalg.vector_norm(direction_to_landmark)
        # Compute the action
        action = direction_to_landmark * 5

        action = torch.clamp(action, min=-u_range, max=u_range)

        return action

if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
