import gym
import os
from safe_ppo_agent import Agent
from train import Train

from arguments import get_args
import os
import sys
sys.path.append("..")
import easycarla
from os.path import dirname
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ppo_utilities.seeds import set_seeds
from ppo_utilities.evaluation import evaluate_model

TRAIN_FLAG = True

carla_params = {
    'number_of_vehicles': 8,
    'number_of_walkers': 0,
    'dt': 0.1,  # time interval between two frames
    'ego_vehicle_filter': 'vehicle.tesla.model3',  # filter for defining ego vehicle
    'surrounding_vehicle_spawned_randomly': True, # Whether surrounding vehicles are spawned randomly (True) or set manually (False)
    'port': 2000,  # connection port
    'town': 'Town10HD_Opt',  # which town to simulate
    'max_time_episode': 200,  # maximum timesteps per episode
    'max_waypoints': 12,  # maximum number of waypoints
    'visualize_waypoints': True,  # Whether to visualize waypoints (default: True)
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'view_mode' : 'top',  # 'top' for bird's-eye view, 'follow' for third-person view
    'traffic': 'off',  # 'on' for normal traffic lights, 'off' for always green and frozen
    'lidar_max_range': 50.0,  # Maximum LIDAR perception range (meters)
    'max_nearby_vehicles': 8,  # Maximum number of nearby vehicles to observe
}

if __name__ == "__main__":
    args = get_args()

    ENV_NAME = args.env_name
    #test_env = gym.make(ENV_NAME, params=carla_params)

    # define the params for constructing the agent


    env = gym.make(ENV_NAME, params=carla_params)
    #env.seed(args.seed)
    #test_env.seed(args.seed + 1)
    n_states = 9 + 2 + 240 + 36 + 4 * carla_params['max_nearby_vehicles']
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]
    n_actions = env.action_space.shape[0]
    n_iterations = args.n_iterations
    lr = args.lr
    device = args.device

    print(f"number of states:{n_states}\n"
          f"action bounds:{action_bounds}\n"
          f"number of actions:{n_actions}")

    set_seeds(args)

    agent = Agent(n_states=n_states,
                  n_iter=n_iterations,
                  env_name=ENV_NAME,
                  action_bounds=action_bounds,
                  n_actions=n_actions,
                  lr=lr,
                  device=device)
    if TRAIN_FLAG:
        trainer = Train(env=env,
                        test_env=env,
                        agent=agent,
                        args=args)
        trainer.step()
    else:
        try:
            agent.load_weights()
        except:
            pass
        eval_rew, eval_cost = evaluate_model(agent, env, render=True)
        print(eval_rew, eval_cost)
    # player = Play(env, agent, ENV_NAME)
    # player.evaluate()

    env.close()