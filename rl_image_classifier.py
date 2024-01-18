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
# Define your pre-trained model loading function
def load_model():
    model = models.densenet121(pretrained=True)  # Ensure this matches the model used for training
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 10)  # Adjust the number of classes if different

    # Load the state dictionary from the checkpoint, only extracting the model weights
    checkpoint = torch.load('best_checkpoint.pth.tar')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)  # If the file directly contains the state dict

    model.eval()
    return model

# Add ThresholdTransform to your transforms
class ThresholdTransform:
    def __init__(self, thr_255):
        self.thr_255 = thr_255

    def __call__(self, img):
        # Convert PIL Image to NumPy array
        img_array = np.array(img)
        
        # Apply threshold
        thresholded_array = (img_array > self.thr_255) * 255  # Element-wise comparison

        # Convert back to PIL Image and return
        return Image.fromarray(thresholded_array.astype(np.uint8))
def image_transforms(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        ThresholdTransform(thr_255=75),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image)

# Define actions
def apply_action(image, action):
    # if action == 0:
    #     return transforms.functional.rotate(image, 25)
    # elif action == 1:
    #     return transforms.functional.hflip(image)
    # # Add other actions based on your original code or new ideas
    return image

# Classification function using your pre-trained model
def classify(model, image):
    image = image_transforms(image).float()
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

# Environment
class ImageEnvironment:
    def __init__(self, dataset_paths, model):
        self.datasets = [self.load_dataset(path) for path in dataset_paths]
        self.current_class = 0
        self.current_image_index = 0
        self.model = model

    def load_dataset(self, path):
        images = []
        for filename in os.listdir(path):
            if filename.endswith(".jpg"):  # Assuming JPG format
                image_path = os.path.join(path, filename)
                image = Image.open(image_path).convert('RGB')
                images.append(image)
        return images

    def reset(self):
        self.current_class = (self.current_class + 1) % len(self.datasets)
        self.current_image_index = 0
        return self.get_state()

    def get_state(self):
        image = self.datasets[self.current_class][self.current_image_index]
        return image, self.current_class

    def step(self, action):
        image = self.datasets[self.current_class][self.current_image_index]
        processed_image = apply_action(image, action)
        prediction = classify(self.model, processed_image)
        reward = 1 if prediction == self.current_class else -1
        self.current_image_index = (self.current_image_index + 1) % len(self.datasets[self.current_class])
        done = self.current_image_index == 0
        return self.get_state(), reward, done

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(64 * 55 * 55, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate):
        self.model = DQN(state_dim, hidden_dim, action_dim)
        self.action_dim = action_dim  # Store action_dim as an attribute
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def select_action(self, state):
        # Use self.action_dim here
        return random.randint(0, self.action_dim - 1)

    def learn(self, state, action, reward, next_state, done):
        # Convert everything to tensors
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        # Get current Q estimate
        q_values = self.model(state)
        current_q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # Compute the expected Q values
        next_q_values = self.model(next_state).max(1)[0]
        expected_q_value = reward + 0.99 * next_q_values * (1 - done)  # 0.99 is the discount factor

        # Compute loss and update the model
        loss = F.mse_loss(current_q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
def set_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        dev = "cuda:0"
    elif torch.backends.mps.is_available():
     #   print("Using Mac GPU")
        dev = "mps"
    else:
        print("Using CPU")
        dev = "cpu"
# Main training loop
def save_checkpoint(state, filename="best_checkpoint.pth.tar"):
    torch.save(state, filename)

def main():
    dataset_paths = [f"/Users/ddayley/Desktop/gas/data/images_copy/{i}" for i in range(1, 11)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model()  
    model.to(device)

    env = ImageEnvironment(dataset_paths, model)
    state_dim = 3
    action_dim = 2
    hidden_dim = 128
    learning_rate = 0.001
    best_total_reward = -float('inf')

    agent = DQNAgent(state_dim, action_dim, hidden_dim, learning_rate)
    replay_buffer = ReplayBuffer(1000)
    num_episodes = 100
    batch_size = 32

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
                # Unpack batch and update DQN model

            state = next_state
            total_reward += reward

        # Save the model if it has improved
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            save_checkpoint({
                'episode': episode,
                'state_dict': agent.model.state_dict(),
                'best_total_reward': best_total_reward,
                'optimizer': agent.optimizer.state_dict(),
            })

        print(f"Episode: {episode}, Total Reward: {total_reward}, Class: {env.current_class}")

if __name__ == '__main__':
    main()
