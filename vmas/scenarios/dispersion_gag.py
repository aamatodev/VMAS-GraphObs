#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import numpy as np
import torch
from scipy.optimize._lsap import linear_sum_assignment

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, AGENT_INFO_TYPE


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.pop("n_agents", 4)
        self.share_reward = kwargs.pop("share_reward", False)
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
                ).uniform_(
                    -self.pos_range,
                    self.pos_range,
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
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        rews = torch.zeros(self.world.batch_dim, device=self.world.device)

        for landmark in self.world.landmarks:
            if is_first:
                landmark.how_many_on_food = torch.stack(
                    [
                        torch.linalg.vector_norm(
                            a.state.pos - landmark.state.pos, dim=1
                        )
                        < a.shape.radius + landmark.shape.radius
                        for a in self.world.agents
                    ],
                    dim=1,
                ).sum(-1)
                landmark.anyone_on_food = landmark.how_many_on_food > 0
                landmark.just_eaten[landmark.anyone_on_food] = True

            assert (landmark.how_many_on_food <= len(self.world.agents)).all()

            if self.share_reward:
                rews[landmark.just_eaten * ~landmark.eaten] += 1
            else:
                on_food = (
                        torch.linalg.vector_norm(
                            agent.state.pos - landmark.state.pos, dim=1
                        )
                        < agent.shape.radius + landmark.shape.radius
                )
                eating_rew = landmark.how_many_on_food.reciprocal().nan_to_num(
                    posinf=0, neginf=0
                )
                rews[on_food * ~landmark.eaten] += eating_rew[on_food * ~landmark.eaten]

            if is_last:
                landmark.eaten += landmark.just_eaten
                landmark.just_eaten[:] = False
                landmark.is_rendering[landmark.eaten] = False

        if self.penalise_by_time:
            rews[rews == 0] = -0.01
        return rews

    def observation(self, agent: Agent):
        obs = []
        for landmark in self.world.landmarks:
            obs.append(
                torch.cat(
                    [
                        landmark.state.pos - agent.state.pos,
                        landmark.eaten.to(torch.int).unsqueeze(-1),
                    ],
                    dim=-1,
                )
            )

        landmark_pos = []
        for idx, landmark in enumerate(self.world.landmarks):
            landmark_pos.append(landmark.state.pos)

        agent_pos = []
        for idx, agent_p in enumerate(self.world.agents):
            agent_pos.append(agent_p.state.pos)

        landmark_pos = torch.cat(landmark_pos, dim=-1)
        agent_pos = torch.cat(agent_pos, dim=-1)

        return {
            "id": torch.tensor(int(agent.name.split("_")[1])).repeat(agent.batch_dim),
            "pos": agent.state.pos,
            "vel": agent.state.vel,
            "relative_landmarks": torch.cat(obs, dim=1),
            "absolute_positions": agent_pos,
            "absolute_landmarks": landmark_pos,
        }

    # def info(self, agent: Agent) -> AGENT_INFO_TYPE:
    #     landmark_pos = []
    #     for idx, landmark in enumerate(self.world.landmarks):
    #         landmark_pos.append(landmark.state.pos)
    #
    #     agent_pos = []
    #     for idx, agent in enumerate(self.world.landmarks):
    #         landmark_pos.append(agent.state.pos)
    #
    #     landmark_pos = torch.cat(landmark_pos, dim=-1)
    #
    #     return {
    #             "absolute_positions": agent_pos,
    #             "absolute_landmarks": landmark_pos,
    #         }

    def done(self):
        return torch.all(
            torch.stack(
                [landmark.eaten for landmark in self.world.landmarks],
                dim=1,
            ),
            dim=-1,
        )


class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, n_agents, **kwargs):
        super().__init__(**kwargs)
        self.n_agents = n_agents

    def assign_agents_to_landmarks(self, cost_matrix):
        # Use the Hungarian algorithm (linear_sum_assignment)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Return the optimal assignments
        assignments = list(zip(row_indices, col_indices))
        return assignments

    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        id = observation["id"].view(-1, 1)
        agents_pos = observation["absolute_positions"].view(-1, self.n_agents, 2)
        landmark_pos = observation["absolute_landmarks"].view(-1, self.n_agents, 2)

        batch_size = agents_pos.shape[0]

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
            assignments.append(self.assign_agents_to_landmarks(np.array(distances_to_landmark[env])))

        selected_landmark = []
        for env in range(batch_size):
            selected_landmark.append(landmark_pos[env][assignments[env][id[0][0].item()][1]])

        direction_to_landmark = torch.from_numpy(np.array(torch.stack(selected_landmark) - agents_pos[:, id[0][0].item()]))
        # Normalize the direction
        direction_to_landmark /= torch.linalg.vector_norm(direction_to_landmark)
        # Compute the action
        action = direction_to_landmark * 5

        action = torch.clamp(action, min=-u_range, max=u_range)

        return action


class RandomPolicy(BaseHeuristicPolicy):
    def __init__(self, n_agents, env, **kwargs):
        super().__init__(**kwargs)
        self.n_agents = n_agents
        self.env = env

    # def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
    #     return self.env.get_random_action(agent)


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=False,
        n_agents=4,
        share_reward=True,
        penalise_by_tim=False,
        pos_range=1.0,
    )
