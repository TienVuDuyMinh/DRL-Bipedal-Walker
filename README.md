# 🦿 BipedalWalker-v3: Deep Reinforcement Learning Agents

<p align="center">
  <img src="https://gymnasium.farama.org/_images/bipedal-walker.gif" width="400" alt="BipedalWalker Demo">
</p>

## 📌 Mô tả dự án

Dự án này áp dụng các thuật toán **Deep Reinforcement Learning (DRL)** để huấn luyện một agent học cách đi bộ và giữ thăng bằng trong môi trường `BipedalWalker-v3` thuộc OpenAI Gym. Agent sẽ học điều khiển một robot hai chân vượt qua địa hình ngẫu nhiên mà không bị ngã.

---

## 🧠 Mục tiêu

- Hiểu và triển khai các thuật toán DRL tiêu biểu: DQN, Double DQN, REINFORCE, PPO.
- So sánh hiệu quả các thuật toán trên cùng một môi trường.
- Ghi lại kết quả, vẽ biểu đồ học, và lưu video mô phỏng agent sau khi huấn luyện.

---

## 🗂️ Cấu trúc thư mục

```plaintext
BipedalWalker-DRL/
├── Env/             # Định nghĩa môi trường, wrapper, seed, preprocessing
├── Dqn/             # Thuật toán DQN, Double DQN
├── Policy/          # Thuật toán chính sách: PPO
├── Training/        # File train cho từng thuật toán
├── Result/          # Lưu kết quả đánh giá, biểu đồ học
├── requirements.txt # Thư viện cần cài đặt
└── README.md        # Mô tả dự án

