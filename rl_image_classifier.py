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
def load_model(start_from_scratch=True, checkpoint_path='best_checkpoint.pth.tar', device=None):
    # Choose whether to start from scratch or load the checkpoint
    if start_from_scratch:
        model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 10)  # Adjust the number of classes if different
    else:
        # Load the existing checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print(f"Checkpoint file '{checkpoint_path}' not found. Starting from scratch.")
            model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 10)  # Adjust the number of classes if different

    model = model.to(device)  # Move the model to the appropriate device
    return model

class ThresholdTransform:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, img):
        # No need to convert to tensor, it should already be one
        # Check if the input is a tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Input should be a torch.Tensor. Got {type(img)}")

        # Visualize before thresholding
       # self.visualize_tensor(img, title="Before Thresholding")

        # Apply thresholding
        thresholded_tensor = (img > self.threshold / 255.0).float()

        # Visualize after thresholding
        #self.visualize_tensor(thresholded_tensor, title="After Thresholding")
        return thresholded_tensor

    def visualize_tensor(self, tensor, title):
        plt.figure()
        # Assuming the tensor is in the range [0, 1]
        plt.imshow(tensor.squeeze().permute(1, 2, 0), cmap='gray' if tensor.size(0) == 1 else None)
        plt.title(title)
        plt.axis('off')
        plt.show()

def image_transforms(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),  # Converts image to tensor and scales it to [0, 1]
        ThresholdTransform(threshold=75),  # Apply thresholding on the tensor
        # Add normalization if needed
    ])
    transformed_image = transform(image)

    # Visualize the final transformed image
    # plt.figure()
    # plt.imshow(transformed_image.squeeze(0).permute(1, 2, 0), cmap='gray')
    # plt.title("Final Transformed Image")
    # plt.axis('off')
    # plt.show()

    return transformed_image




# Define actions for your RL agent
def apply_action(image, action):
    # Define actions (e.g., rotate, flip). Modify as per your task requirements
    # Example actions:
    # if action == 0:
    #     return transforms.functional.rotate(image, 25)
    # elif action == 1:
    #     return transforms.functional.hflip(image)
    return image
# Classification function using pre-trained model
# def classify(model, image, device):
#     image = image_transforms(image).float().to(device)
#     image = image.unsqueeze(0)
#     with torch.no_grad():
#         output = model(image)
#     _, predicted = torch.max(output.data, 1)
#     return predicted.item()
import matplotlib.pyplot as plt

# Classification function using pre-trained model with visualization
def classify(model, image, device):
    transformed_image = image_transforms(image).to(device)
    transformed_image = transformed_image.unsqueeze(0) if transformed_image.dim() == 3 else transformed_image

    with torch.no_grad():
        output = model(transformed_image)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()


# Define RL environment
class ImageEnvironment:
    def __init__(self, dataset_paths, model, device):  # Add 'device' as an argument
        self.datasets = [self.load_dataset(path) for path in dataset_paths]
        self.current_class = 0
        self.current_image_index = 0
        self.model = model
        self.device = device 

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
         # Debug message
        print(f"Debug - Current Class: {self.current_class}, Image Index: {self.current_image_index}, Action: {action}")

        # Apply action and get next state, reward, and done flag
        image = self.datasets[self.current_class][self.current_image_index]
        processed_image = apply_action(image, action)
        prediction = classify(self.model, processed_image, self.device)  # Pass the device here     
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
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate, device):
        self.model = DQN(state_dim, hidden_dim, action_dim).to(device)
        self.action_dim = action_dim
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = device 

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
           # print(f"Debug - State Type: {type(state)}, Content: {state}")
          #  print(f"Debug - Next State Type: {type(next_state)}, Content: {next_state}")

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


        # Transform and move images to device
            state_image = image_transforms(state_image).unsqueeze(0).to(self.device)
            next_state_image = image_transforms(next_state_image).unsqueeze(0).to(self.device)

            action = torch.LongTensor([action]).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            done = torch.FloatTensor([done]).to(self.device)

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
def set_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        print("Using Mac GPU")
        device = torch.device("mps")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    return device

def main():
    device = set_device()
    model = load_model(device=device)
    # ... rest of your code ...

    # Define paths to your image datasets (modify as needed)
    dataset_paths = [f"/Users/ddayley/Desktop/gas/data/images_copy/{i}" for i in range(0, 10)]
  
    env = ImageEnvironment(dataset_paths, model, device)
    state_dim = 3  # Modify as per your state representation
    action_dim = 2  # Modify as per your number of actions
    hidden_dim = 128
    learning_rate = 0.001

    agent = DQNAgent(state_dim, action_dim, hidden_dim, learning_rate, device)
    replay_buffer = ReplayBuffer(1000)
    num_episodes = 300
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
                #print(f"Debug - Batch Sample: {batch}")
                states, actions, rewards, next_states, dones = zip(*batch)
                agent.learn(states, actions, rewards, next_states, dones)

            state = next_state
            total_reward += reward
        print("Best : " + str(total_reward))
        # Check for best model
        if total_reward > best_total_reward:
                best_total_reward = total_reward
                print("Saving Checkpoint : " + str(total_reward))
                save_checkpoint({
                    'episode': episode,
                    'state_dict': agent.model.state_dict(),
                    'best_total_reward': best_total_reward,
                    'optimizer': agent.optimizer.state_dict(),
                }, filename="best_checkpoint.pth.tar")  # Save as the best_checkpoint

        print(f"Episode: {episode}, Total Reward: {total_reward}")
if __name__ == '__main__':
    main()
