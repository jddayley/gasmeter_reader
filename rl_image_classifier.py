import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from collections import deque
import numpy as np
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights

# Define your pre-trained model loading function
def load_model(start_from_scratch=True, checkpoint_path='best_checkpoint.pth.tar'):
    # Choose whether to start from scratch or load the checkpoint
    if start_from_scratch:
        # Start training from scratch
        model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 10)  # Adjust the number of classes if different
    else:
        # Load the existing checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Handle the case where the checkpoint file doesn't exist
            print(f"Checkpoint file '{checkpoint_path}' not found. Starting from scratch.")
            model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 10)  # Adjust the number of classes if different

    model.eval()
    return model

# Define ThresholdTransform class
class ThresholdTransform:
    def __init__(self, thr_255):
        self.thr_255 = thr_255

    def __call__(self, img):
        # Convert PIL Image to NumPy array
        img_array = np.array(img)
        # Apply threshold
        thresholded_array = (img_array > self.thr_255) * 255
        # Convert back to PIL Image and return
        return Image.fromarray(thresholded_array.astype(np.uint8))

# Define image preprocessing transforms
def image_transforms(state):
    if isinstance(state, tuple) and len(state) == 2:
        image, _ = state
    elif isinstance(state, Image.Image):
        image = state
    else:
        raise ValueError("Unsupported input type for image_transforms")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        ThresholdTransform(thr_255=75),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).float()






# Define actions for your RL agent
def apply_action(image, action):
    # Define actions (e.g., rotate, flip). Modify as per your task requirements
    # Example actions:
    if action == 0:
        return transforms.functional.rotate(image, 25)
    elif action == 1:
        return transforms.functional.hflip(image)
    return image

# Classification function using pre-trained model
def classify(model, image):
    image = image_transforms(image).float()
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

# Define RL environment
class ImageEnvironment:
    def __init__(self, dataset_paths, model):
        # Load datasets from provided paths
        self.datasets = [self.load_dataset(path) for path in dataset_paths]
        self.current_class = 0
        self.current_image_index = 0
        self.model = model

    def load_dataset(self, path):
        # Load images from a dataset directory
        images = []
        for filename in os.listdir(path):
            if filename.endswith(".jpg"):  # Modify as per your file types
                image_path = os.path.join(path, filename)
                image = Image.open(image_path).convert('RGB')
                images.append(image)
        return images

    def reset(self):
        # Reset environment to a new state
        self.current_class = (self.current_class + 1) % len(self.datasets)
        self.current_image_index = 0
        return self.get_state()

    # def get_state(self):
    #     # Get current state (image and class)
    #     image = self.datasets[self.current_class][self.current_image_index]
    #     return image, self.current_class

    def step(self, action):
        # Apply action and get next state, reward, and done flag
        image = self.datasets[self.current_class][self.current_image_index]
        processed_image = apply_action(image, action)
        prediction = classify(self.model, processed_image)
        reward = 1 if prediction == self.current_class else -1
        self.current_image_index = (self.current_image_index + 1) % len(self.datasets[self.current_class])
        done = self.current_image_index == 0
        return self.get_state(), reward, done
    def get_state(self):
        # Get current state (image and class)
        image = self.datasets[self.current_class][self.current_image_index]
        class_label = self.current_class
        return (image, class_label)


# Define DQN model
class DQN(nn.Module):
    # Define your DQN architecture here
    # Modify according to your task requirements
    def __init__(self, input_channels, hidden_dim, output_dim):
        super(DQN, self).__init__()
        # Example layers (modify as needed):
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        conv1_output_size = ((224 - 8) // 4) + 1
        conv2_output_size = ((conv1_output_size - 4) // 2) + 1
        self.fc1_input_size = 64 * conv2_output_size * conv2_output_size
        self.fc1 = nn.Linear(self.fc1_input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Forward pass through the network
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define DQN agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate):
        self.model = DQN(state_dim, hidden_dim, action_dim)
        self.action_dim = action_dim
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def select_action(self, state):
        # Random action selection for now
        # Modify to implement more sophisticated policy (e.g., epsilon-greedy)
        return random.randint(0, self.action_dim - 1)

    def learn(self, states, actions, rewards, next_states, dones):
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]

            # Handling the state format
            if isinstance(state, tuple) and len(state) == 2:
                state_image, _ = state
            elif isinstance(state, Image.Image):
                state_image = state
            else:
                raise ValueError("Unsupported input type for state")

            # Similar handling for next_state
            if isinstance(next_state, tuple) and len(next_state) == 2:
                next_state_image, _ = next_state
            elif isinstance(next_state, Image.Image):
                next_state_image = next_state
            else:
                raise ValueError("Unsupported input type for next state")

            # Rest of the learning logic
            state_image = image_transforms(state_image).unsqueeze(0)
            next_state_image = image_transforms(next_state_image).unsqueeze(0)

            action = torch.LongTensor([action])
            reward = torch.FloatTensor([reward])
            done = torch.FloatTensor([done])

            # Compute Q values and loss
            q_values = self.model(state_image)
            current_q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            next_q_values = self.model(next_state_image).max(1)[0]
            expected_q_value = reward + 0.99 * next_q_values * (1 - done)  # Discount factor
            loss = F.mse_loss(current_q_value, expected_q_value.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()




# Define replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Main training loop
def save_checkpoint(state, filename="best_checkpoint.pth.tar"):
    # Save model state to a file
    torch.save(state, filename)

def main():
    # Define paths to your image datasets (modify as needed)
    dataset_paths = [f"/Users/ddayley/Desktop/gas/data/images_copy/{i}" for i in range(0, 10)]

    model = load_model()
    env = ImageEnvironment(dataset_paths, model)
    state_dim = 3  # Modify as per your state representation
    action_dim = 2  # Modify as per your number of actions
    hidden_dim = 128
    learning_rate = 0.001

    agent = DQNAgent(state_dim, action_dim, hidden_dim, learning_rate)
    replay_buffer = ReplayBuffer(1000)
    num_episodes = 100
    batch_size = 32
    best_total_reward = -float('inf')

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                # Unpack and learn
                states, actions, rewards, next_states, dones = zip(*batch)
                agent.learn(states, actions, rewards, next_states, dones)

            state = next_state
            total_reward += reward

        # Check for best model
        if total_reward > best_total_reward:
                best_total_reward = total_reward
                save_checkpoint({
                    'episode': episode,
                    'state_dict': agent.model.state_dict(),
                    'best_total_reward': best_total_reward,
                    'optimizer': agent.optimizer.state_dict(),
                }, filename="best_checkpoint.pth.tar")  # Save as the best_checkpoint

        print(f"Episode: {episode}, Total Reward: {total_reward}")

if __name__ == '__main__':
    main()
