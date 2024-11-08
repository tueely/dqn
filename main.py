import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
import sys
import threading
import time

# ============================ #
#        Configuration         #
# ============================ #

# Screen dimensions
WIDTH, HEIGHT = 800, 600  # Increased for better layout
GAME_AREA_WIDTH = WIDTH * 2 // 3
PLOT_AREA_WIDTH = WIDTH // 3

# Colors
WHITE = (255, 255, 255)
BLACK = (18, 18, 18)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 200)
GREY = (50, 50, 50)
LIGHT_GREY = (200, 200, 200)
YELLOW = (255, 255, 0)

# Frame rate
FPS = 60
CLOCK = pygame.time.Clock()

# Agent properties
AGENT_SIZE = 20
AGENT_SPEED = 5

# Obstacle properties
OBSTACLE_SIZE = 20
OBSTACLE_SPEED = 3
OBSTACLE_SPAWN_RATE = 30  # Increased for better pacing
MAX_OBSTACLES = 50  # Reduced for better performance

# DQN parameters
STATE_SIZE = 8  # [agent_x, agent_y, closest_obs_x, closest_obs_y, second_closest_obs_x, second_closest_obs_y, third_closest_obs_x, third_closest_obs_y]
ACTION_SIZE = 4  # Up, Down, Left, Right
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
LEARNING_RATE = 1e-3

# Rendering optimization
RENDER_EVERY = 10  # Render every N episodes

# Training configuration
TOTAL_EPISODES = 100  # Reduced from 1000 to 100

# ============================ #
#        Initialize Pygame      #
# ============================ #

pygame.init()
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Survival Arena with AI")
FONT = pygame.font.SysFont('Arial', 16)

# ============================ #
#       Global Variables        #
# ============================ #

average_survival_times = []
lock = threading.Lock()
training_complete = False
shared_obstacles = []
shared_agent = {'rect': pygame.Rect(GAME_AREA_WIDTH//2, HEIGHT//2, AGENT_SIZE, AGENT_SIZE), 'color': GREEN}

# ============================ #
#          Classes              #
# ============================ #

class Agent:
    def __init__(self):
        self.rect = pygame.Rect(GAME_AREA_WIDTH//2, HEIGHT//2, AGENT_SIZE, AGENT_SIZE)
        self.color = GREEN

    def move(self, action):
        if action == 0:  # Up
            self.rect.y -= AGENT_SPEED
        elif action == 1:  # Down
            self.rect.y += AGENT_SPEED
        elif action == 2:  # Left
            self.rect.x -= AGENT_SPEED
        elif action == 3:  # Right
            self.rect.x += AGENT_SPEED

        # Wrap around horizontally
        self.rect.x %= GAME_AREA_WIDTH
        # Wrap around vertically
        self.rect.y %= HEIGHT

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

class Obstacle:
    def __init__(self):
        self.rect = pygame.Rect(random.randint(0, GAME_AREA_WIDTH - OBSTACLE_SIZE),
                                random.randint(0, HEIGHT - OBSTACLE_SIZE),
                                OBSTACLE_SIZE, OBSTACLE_SIZE)
        self.color = RED
        # Random movement direction
        self.dx = random.choice([-OBSTACLE_SPEED, OBSTACLE_SPEED])
        self.dy = random.choice([-OBSTACLE_SPEED, OBSTACLE_SPEED])

    def move(self):
        self.rect.x += self.dx
        self.rect.y += self.dy

        # Bounce off the walls
        if self.rect.left < 0 or self.rect.right > GAME_AREA_WIDTH:
            self.dx *= -1
        if self.rect.top < 0 or self.rect.bottom > HEIGHT:
            self.dy *= -1

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class AgentAI:
    def __init__(self):
        self.policy_net = DQN(STATE_SIZE, ACTION_SIZE)
        self.target_net = DQN(STATE_SIZE, ACTION_SIZE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0
        self.epsilon = EPS_START

    def select_action(self, state):
        # Epsilon-greedy strategy
        self.epsilon = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))

        state_batch = torch.FloatTensor(batch[0])
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch[2]).unsqueeze(1)
        next_state_batch = torch.FloatTensor(batch[3])
        done_batch = torch.FloatTensor(batch[4]).unsqueeze(1)

        # Compute current Q values
        current_q = self.policy_net(state_batch).gather(1, action_batch)

        # Compute next Q values from target network
        next_q = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)
        target_q = reward_batch + (GAMMA * next_q * (1 - done_batch))

        # Compute loss
        loss = F.mse_loss(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ============================ #
#        Helper Functions       #
# ============================ #

def calculate_distance(rect1, rect2):
    """Calculate the Euclidean distance between the centers of two pygame.Rect objects."""
    dx = rect1.centerx - rect2.centerx
    dy = rect1.centery - rect2.centery
    return np.sqrt(dx ** 2 + dy ** 2)

def get_state(agent, obstacles):
    # State Representation:
    # [agent_x, agent_y, closest_obs_x, closest_obs_y, second_closest_obs_x, second_closest_obs_y, third_closest_obs_x, third_closest_obs_y]
    state = [agent.rect.x / GAME_AREA_WIDTH,
             agent.rect.y / HEIGHT]

    if obstacles:
        # Calculate distances to all obstacles
        distances = [calculate_distance(agent.rect, obs.rect) for obs in obstacles]
        # Get indices of the three closest obstacles
        closest_indices = np.argsort(distances)[:3]
        for idx in closest_indices:
            closest = obstacles[idx]
            # Append normalized positions
            state.append(closest.rect.x / GAME_AREA_WIDTH)
            state.append(closest.rect.y / HEIGHT)
        # If fewer than 3 obstacles, pad with zeros
        while len(state) < STATE_SIZE:
            state.extend([0.0, 0.0])
    else:
        # No obstacles; pad the remaining state with zeros
        state.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return state[:STATE_SIZE]

def draw_text(screen, text, position, color=WHITE):
    """Render text on the Pygame screen."""
    text_surface = FONT.render(text, True, color)
    screen.blit(text_surface, position)

def plot_survival_time(screen, survival_times):
    """Plot the average survival times within the Pygame window."""
    # Define plot area
    plot_x_start = GAME_AREA_WIDTH + 10
    plot_y_start = 50
    plot_width = PLOT_AREA_WIDTH - 20
    plot_height = HEIGHT - 60

    # Background for plot area
    pygame.draw.rect(screen, GREY, (GAME_AREA_WIDTH, 0, PLOT_AREA_WIDTH, HEIGHT))
    
    # Draw border
    pygame.draw.rect(screen, LIGHT_GREY, (GAME_AREA_WIDTH, 0, PLOT_AREA_WIDTH, HEIGHT), 2)

    if len(survival_times) < 2:
        return

    # Determine scaling based on maximum survival time
    max_frames = max(survival_times) if survival_times else 1
    max_frames = max(max_frames, 100)  # Minimum scale to prevent overly compressed plots

    # Draw grid lines
    grid_color = LIGHT_GREY
    for i in range(5):
        # Horizontal lines
        y = plot_y_start + i * (plot_height / 4)
        pygame.draw.line(screen, grid_color, (GAME_AREA_WIDTH + 10, y), (WIDTH - 10, y), 1)
        # Vertical lines
        x = plot_x_start + i * (plot_width / 4)
        pygame.draw.line(screen, grid_color, (x, plot_y_start), (x, HEIGHT - 10), 1)

    # Define points
    points = []
    for i, survival in enumerate(survival_times):
        x = plot_x_start + (i / TOTAL_EPISODES) * plot_width
        y = plot_y_start + plot_height - (survival / max_frames) * plot_height
        points.append((x, y))

    # Draw lines
    pygame.draw.lines(screen, YELLOW, False, points, 2)

    # Draw axes labels
    label_font = pygame.font.SysFont('Arial', 14)
    # Y-axis label
    y_label = label_font.render("Avg Survival", True, WHITE)
    screen.blit(y_label, (GAME_AREA_WIDTH + 10, 10))
    # X-axis label
    x_label = label_font.render("Episode", True, WHITE)
    screen.blit(x_label, (plot_x_start + plot_width // 2 - 20, HEIGHT - 30))

    # Draw current average survival time
    if survival_times:
        current_average = survival_times[-1]
        avg_text = label_font.render(f"{current_average:.2f}", True, WHITE)
        screen.blit(avg_text, (GAME_AREA_WIDTH + plot_width + 5, plot_y_start + plot_height - (current_average / max_frames) * plot_height - 10))

# ============================ #
#        Training Function      #
# ============================ #

def train_ai():
    global training_complete
    agent = Agent()
    ai = AgentAI()
    obstacles = []
    frame_count = 0
    episode = 0
    total_survival_time = 0
    average_survival_time = 0

    while episode < TOTAL_EPISODES:
        obstacles = []
        agent = Agent()
        done = False
        survival_time = 0

        # Initialize shared agent position
        with lock:
            shared_agent['rect'] = agent.rect.copy()

        while not done:
            # Spawn obstacles
            if frame_count % OBSTACLE_SPAWN_RATE == 0 and len(obstacles) < MAX_OBSTACLES:
                obs = Obstacle()
                obstacles.append(obs)
                with lock:
                    shared_obstacles.append(obs)

            # Move obstacles
            for obs in obstacles:
                obs.move()

            # Update shared obstacles
            with lock:
                # Clear and update shared_obstacles to reflect current positions
                shared_obstacles.clear()
                shared_obstacles.extend(obstacles)

            # Get current state
            state = get_state(agent, obstacles)

            # Select and perform action
            action = ai.select_action(state)
            agent.move(action)

            # Update shared agent position
            with lock:
                shared_agent['rect'] = agent.rect.copy()

            # Check for collisions
            collision = False
            for obs in obstacles:
                if agent.rect.colliderect(obs.rect):
                    collision = True
                    break

            # Define reward
            if collision:
                reward = -1.0
                done = True
            else:
                reward = 0.01  # Small reward for survival

            # Get next state
            next_state = get_state(agent, obstacles)

            # Store transition in memory
            ai.memory.push((state, action, reward, next_state, collision))

            # Optimize the model
            ai.optimize_model()

            # Update target network
            if frame_count % TARGET_UPDATE == 0:
                ai.target_net.load_state_dict(ai.policy_net.state_dict())

            # Update survival time and frame count
            survival_time += 1
            frame_count += 1

        episode += 1
        total_survival_time += survival_time
        average_survival_time = total_survival_time / episode

        # Append to global list with thread safety
        with lock:
            average_survival_times.append(average_survival_time)

        # Log progress
        print(f"Episode {episode}/{TOTAL_EPISODES} - Survival Time: {survival_time} frames - Average Survival Time: {average_survival_time:.2f} frames - Epsilon: {ai.epsilon:.2f}")

    training_complete = True

# ============================ #
#           Main Loop           #
# ============================ #

def main():
    # Start training in a separate thread
    training_thread = threading.Thread(target=train_ai)
    training_thread.start()

    while not training_complete or len(average_survival_times) < TOTAL_EPISODES:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Fill background
        SCREEN.fill(BLACK)

        # Draw Game Area
        with lock:
            current_obstacles = shared_obstacles.copy()
            agent_rect = shared_agent['rect'].copy()
            agent_color = shared_agent['color']

        # Draw Agent
        pygame.draw.rect(SCREEN, agent_color, agent_rect)

        # Draw Obstacles
        for obs in current_obstacles:
            pygame.draw.rect(SCREEN, obs.color, obs.rect)

        # Draw Plot Area
        with lock:
            plot_times = average_survival_times.copy()
        plot_survival_time(SCREEN, plot_times)

        # Draw HUD
        current_average = plot_times[-1] if plot_times else 0
        with lock:
            current_episode = len(plot_times)
        draw_text(SCREEN, f"Episode: {current_episode}/{TOTAL_EPISODES}", (10, 10))
        draw_text(SCREEN, f"Avg Survival Time: {current_average:.2f}", (10, 40))
        # Optionally, display epsilon if accessible
        # draw_text(SCREEN, f"Epsilon: {ai.epsilon:.2f}", (10, 70))

        pygame.display.flip()

        CLOCK.tick(FPS)

    training_thread.join()
    pygame.quit()
    print("Training complete.")

    # Optional: Save the trained model
    # torch.save(ai.policy_net.state_dict(), "dqn_survival_arena.pth")
    # print("Model saved to dqn_survival_arena.pth")

if __name__ == "__main__":
    main()
