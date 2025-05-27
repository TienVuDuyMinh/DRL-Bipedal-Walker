import gymnasium as gym
from gymnasium import Env
from typing import Optional, List, Any

def make_env(env_name: str = "BipedalWalker-v3",
             seed: Optional[int] = None,
             render_mode: Optional[str] = None) -> Env:
    """
    Tạo môi trường Gym với tùy chọn seed và chế độ hiển thị.
    
    Parameters:
        env_name (str): Tên môi trường (ví dụ: "BipedalWalker-v3")
        seed (int, optional): Seed ngẫu nhiên cho môi trường.
        render_mode (str, optional): 'human' để hiển thị, hoặc None để huấn luyện.

    Returns:
        env (Env): Môi trường đã được khởi tạo.
    """
    env = gym.make(env_name, render_mode=render_mode)
    
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    
    return env


def evaluate_agent(env: Env, agent: Any, episodes: int = 5, render: bool = False) -> List[float]:
    """
    Chạy agent trong môi trường để đánh giá kết quả trung bình.

    Parameters:
        env (Env): Môi trường Gym.
        agent (object): Agent đã huấn luyện, cần có phương thức act(state, deterministic=True).
        episodes (int): Số episode để đánh giá.
        render (bool): Có hiển thị không.

    Returns:
        rewards (List[float]): Tổng reward mỗi episode.
    """
    rewards = []
    try:
        for ep in range(episodes):
            reset_result = env.reset()
            # Gym mới trả về tuple (obs, info), gym cũ trả về obs
            if isinstance(reset_result, tuple):
                state, _ = reset_result
            else:
                state = reset_result

            done = False
            total_reward = 0

            while not done:
                if render:
                    env.render()

                action = agent.act(state, deterministic=True)
                # Nếu action là tensor hoặc kiểu khác convert sang numpy
                if hasattr(action, 'numpy'):
                    action = action.numpy()

                step_result = env.step(action)
                # Gym mới trả về 5 giá trị, cũ trả về 4
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_result

                state = next_state
                total_reward += reward

            rewards.append(total_reward)
            print(f"Episode {ep + 1}: Reward = {total_reward:.2f}")

    finally:
        env.close()

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")
    return rewards
