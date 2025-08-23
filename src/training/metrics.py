import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any
from collections import deque
import pickle
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

class MetricsTracker:

    def __init__(self, log_dir: str, plot_dir: str, save_frequency: int = 100,
                 use_tensorboard: bool = False, tensorboard_log_dir: str = None,
                 moving_average_window: int = 100):

        self.log_dir = log_dir
        self.plot_dir = plot_dir
        self.save_frequency = save_frequency
        self.use_tensorboard = use_tensorboard
        
        if self.use_tensorboard:
            if tensorboard_log_dir is None:
                tensorboard_log_dir = os.path.join(log_dir, "tensorboard")
            
            self.tensorboard_log_dir = tensorboard_log_dir
            self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
            print(f"Initialized TensorBoard logging at: {tensorboard_log_dir}")
            print(f"Run 'tensorboard --logdir {tensorboard_log_dir}' to view logs")

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        
        self.episode_score_rewards: List[float] = []
        self.episode_custom_rewards: List[float] = []
        self.episode_total_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_losses: List[float] = []
        self.epsilon_values: List[float] = []
        self.learning_rates: List[float] = []
        

        self.moving_avg_score_rewards: List[float] = []
        self.moving_avg_custom_rewards: List[float] = []
        self.moving_avg_total_rewards: List[float] = []
        self.best_score_reward: float = float('-inf')
        self.best_custom_reward: float = float('-inf')
        self.best_total_reward: float = float('-inf')
        self.best_episode: int = 0
        self.total_steps: int = 0
        
        self.lives_lost: List[int] = []
        self.shots_fired: List[int] = []
        self.enemies_killed: List[int] = []
        

        self.gradient_norms: List[float] = []
        self.q_value_estimates: List[float] = []
        self.replay_buffer_size: List[int] = []
        

        self.episode_times: List[float] = []
        self.training_start_time = datetime.now()
        
        self.score_reward_window = deque(maxlen=moving_average_window)
        self.custom_reward_window = deque(maxlen=moving_average_window)
        self.total_reward_window = deque(maxlen=moving_average_window)
        self.loss_window = deque(maxlen=moving_average_window)

        self.hyperparams: Dict[str, Any] = {}
        self.checkpoints: List[Dict] = []
        
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        self.hyperparams.update(hyperparams)

        if self.use_tensorboard:
            for key, value in hyperparams.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"hyperparams/{key}", value, 0)
                else:
                    self.writer.add_text(f"hyperparams/{key}", str(value), 0)
        
    def save_progress(self, checkpoint_name: str = None) -> str:
        if checkpoint_name is None:
            checkpoint_name = f"progress_episode_{len(self.episode_total_rewards)}"
        
        checkpoint_path = os.path.join(self.log_dir, f"{checkpoint_name}.pkl")
        
        progress_data = {
            'metrics': {
                'episode_score_rewards': self.episode_score_rewards,
                'episode_custom_rewards': self.episode_custom_rewards,
                'episode_total_rewards': self.episode_total_rewards,
                'episode_lengths': self.episode_lengths,
                'episode_losses': self.episode_losses,
                'epsilon_values': self.epsilon_values,
                'learning_rates': self.learning_rates,
                'moving_avg_score_rewards': self.moving_avg_score_rewards,
                'moving_avg_custom_rewards': self.moving_avg_custom_rewards,
                'moving_avg_total_rewards': self.moving_avg_total_rewards,
                'lives_lost': self.lives_lost,
                'shots_fired': self.shots_fired,
                'enemies_killed': self.enemies_killed,
                'gradient_norms': self.gradient_norms,
                'q_value_estimates': self.q_value_estimates,
                'replay_buffer_size': self.replay_buffer_size,
                'episode_times': self.episode_times
            },
            'state': {
                'best_score_reward': self.best_score_reward,
                'best_custom_reward': self.best_custom_reward,
                'best_total_reward': self.best_total_reward,
                'best_episode': self.best_episode,
                'total_steps': self.total_steps,
                'training_start_time': self.training_start_time.isoformat(),
                'current_episode': len(self.episode_total_rewards),
                'moving_avg_window_size': self.score_reward_window.maxlen
            },
            'config': {
                'hyperparams': self.hyperparams,
                'log_dir': self.log_dir,
                'plot_dir': self.plot_dir,
                'save_frequency': self.save_frequency,
                'use_tensorboard': self.use_tensorboard,
                'tensorboard_log_dir': getattr(self, 'tensorboard_log_dir', None)
            },
            'checkpoints': self.checkpoints,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(progress_data, f)
        
        json_path = os.path.join(self.log_dir, f"{checkpoint_name}.json")
        progress_data_json = progress_data.copy()
        progress_data_json['metrics'] = {k: list(v) if isinstance(v, (list, np.ndarray)) else v 
                                       for k, v in progress_data['metrics'].items()}
        
        with open(json_path, 'w') as f:
            json.dump(progress_data_json, f, indent=2, default=str)
        
        print(f"Progress saved to: {checkpoint_path}")
        return checkpoint_path
    
    def load_progress(self, checkpoint_path: str = None, resume_tensorboard: bool = True) -> bool:
        """Load training progress from checkpoint"""
        if checkpoint_path is None:
            #finding latest checkpoint
            checkpoint_files = [f for f in os.listdir(self.log_dir) if f.startswith('progress_') and f.endswith('.pkl')]
            if not checkpoint_files:
                print("No progress checkpoints found")
                return False
            
            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.log_dir, x)))
            checkpoint_path = os.path.join(self.log_dir, checkpoint_files[-1])
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            with open(checkpoint_path, 'rb') as f:
                progress_data = pickle.load(f)
            
            # Restoring metrics
            for key, value in progress_data['metrics'].items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            # Restoring state
            state = progress_data['state']
            self.best_score_reward = state['best_score_reward']
            self.best_custom_reward = state['best_custom_reward']
            self.best_total_reward = state['best_total_reward']
            self.best_episode = state['best_episode']
            self.total_steps = state['total_steps']
            self.training_start_time = datetime.fromisoformat(state['training_start_time'])
            
            # Restoring moving average windows data
            window_size = state.get('moving_avg_window_size', 100)
            self.score_reward_window = deque(maxlen=window_size)
            self.custom_reward_window = deque(maxlen=window_size)
            self.total_reward_window = deque(maxlen=window_size)
            self.loss_window = deque(maxlen=window_size)
            
            # Filling windows with recent data
            if self.episode_score_rewards:
                recent_episodes = min(window_size, len(self.episode_score_rewards))
                self.score_reward_window.extend(self.episode_score_rewards[-recent_episodes:])
                self.custom_reward_window.extend(self.episode_custom_rewards[-recent_episodes:])
                self.total_reward_window.extend(self.episode_total_rewards[-recent_episodes:])
                self.loss_window.extend(self.episode_losses[-recent_episodes:])
            
            # Restoring config
            config = progress_data.get('config', {})
            self.hyperparams = config.get('hyperparams', {})
            self.checkpoints = progress_data.get('checkpoints', [])
            
            if resume_tensorboard and self.use_tensorboard:
                self._resume_tensorboard_logging()
            
            current_episode = state.get('current_episode', len(self.episode_total_rewards))
            print(f"Progress loaded from: {checkpoint_path}")
            print(f"Resumed from episode {current_episode} with {self.total_steps} total steps")
            return True
            
        except Exception as e:
            print(f"Error loading progress: {e}")
            return False
    
    def _resume_tensorboard_logging(self):
        if not self.use_tensorboard or not hasattr(self, 'writer'):
            return
        
        print("Resuming TensorBoard with historical data...")
        
        for i, (score_reward, custom_reward, total_reward, length, loss, epsilon, episode_time) in enumerate(
            zip(self.episode_score_rewards, self.episode_custom_rewards, self.episode_total_rewards,
                self.episode_lengths, self.episode_losses, self.epsilon_values, self.episode_times)):
            
            episode = i + 1
            self.writer.add_scalar("episode/score_reward", score_reward, episode)
            self.writer.add_scalar("episode/custom_reward", custom_reward, episode)
            self.writer.add_scalar("episode/total_reward", total_reward, episode)
            self.writer.add_scalar("episode/length", length, episode)
            self.writer.add_scalar("episode/loss", loss, episode)
            self.writer.add_scalar("episode/epsilon", epsilon, episode)
            self.writer.add_scalar("episode/time", episode_time, episode)
            
            if i < len(self.moving_avg_score_rewards):
                self.writer.add_scalar("moving_average/score_reward", self.moving_avg_score_rewards[i], episode)
                self.writer.add_scalar("moving_average/custom_reward", self.moving_avg_custom_rewards[i], episode)
                self.writer.add_scalar("moving_average/total_reward", self.moving_avg_total_rewards[i], episode)
            
            if i < len(self.lives_lost):
                self.writer.add_scalar("game_metrics/lives_lost", self.lives_lost[i], episode)
            if i < len(self.shots_fired):
                self.writer.add_scalar("game_metrics/shots_fired", self.shots_fired[i], episode)
            if i < len(self.enemies_killed):
                self.writer.add_scalar("game_metrics/enemies_killed", self.enemies_killed[i], episode)
            if i < len(self.learning_rates):
                self.writer.add_scalar("training/learning_rate", self.learning_rates[i], episode)
        
        for key, value in self.hyperparams.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"hyperparams/{key}", value, 0)
            else:
                self.writer.add_text(f"hyperparams/{key}", str(value), 0)
        
        print("TensorBoard logging resumed successfully")
    
    def auto_save_progress(self) -> None:
        if len(self.episode_total_rewards) % self.save_frequency == 0:
            episode = len(self.episode_total_rewards)
            self.save_progress(f"auto_save_episode_{episode}")
    
    def get_checkpoint_info(self) -> List[Dict]:
        checkpoint_files = [f for f in os.listdir(self.log_dir) if f.startswith('progress_') and f.endswith('.pkl')]
        
        checkpoints = []
        for file in checkpoint_files:
            file_path = os.path.join(self.log_dir, file)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                checkpoints.append({
                    'file': file,
                    'path': file_path,
                    'episode': data['state'].get('current_episode', 0),
                    'total_steps': data['state'].get('total_steps', 0),
                    'best_total_reward': data['state'].get('best_total_reward', 0),
                    'timestamp': data.get('timestamp', 'unknown'),
                    'size_mb': os.path.getsize(file_path) / (1024 * 1024)
                })
            except Exception as e:
                print(f"Error reading checkpoint {file}: {e}")
        
        return sorted(checkpoints, key=lambda x: x['episode'], reverse=True)
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> None:
        checkpoints = self.get_checkpoint_info()
        
        if len(checkpoints) <= keep_last_n:
            return
        
        to_delete = checkpoints[keep_last_n:]
        deleted_count = 0
        
        for checkpoint in to_delete:
            try:
                os.remove(checkpoint['path'])
                json_path = checkpoint['path'].replace('.pkl', '.json')
                if os.path.exists(json_path):
                    os.remove(json_path)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting checkpoint {checkpoint['file']}: {e}")
        
        if deleted_count > 0:
            print(f"Cleaned up {deleted_count} old checkpoint files")
    
    def log_episode(self, episode: int, score_reward: float, custom_reward: float, length: int, 
                   loss: float, epsilon: float, episode_time: float,
                   lives_lost: int = 0, shots_fired: int = 0, 
                   enemies_killed: int = 0, learning_rate: float = None) -> None:

        total_reward = score_reward + custom_reward

        self.episode_score_rewards.append(score_reward)
        self.episode_custom_rewards.append(custom_reward)
        self.episode_total_rewards.append(total_reward)
        self.episode_lengths.append(length)
        self.episode_losses.append(loss)
        self.epsilon_values.append(epsilon)
        self.episode_times.append(episode_time)
        
        self.lives_lost.append(lives_lost)
        self.shots_fired.append(shots_fired)
        self.enemies_killed.append(enemies_killed)
        
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        
        self.score_reward_window.append(score_reward)
        self.custom_reward_window.append(custom_reward)
        self.total_reward_window.append(total_reward)
        self.loss_window.append(loss)
        
        moving_avg_score = np.mean(self.score_reward_window)
        moving_avg_custom = np.mean(self.custom_reward_window)
        moving_avg_total = np.mean(self.total_reward_window)
        
        self.moving_avg_score_rewards.append(moving_avg_score)
        self.moving_avg_custom_rewards.append(moving_avg_custom)
        self.moving_avg_total_rewards.append(moving_avg_total)
        
        if score_reward > self.best_score_reward:
            self.best_score_reward = score_reward
        if custom_reward > self.best_custom_reward:
            self.best_custom_reward = custom_reward
        if total_reward > self.best_total_reward:
            self.best_total_reward = total_reward
            self.best_episode = episode
        
        self.total_steps += length

        if self.use_tensorboard:
            self.writer.add_scalar("episode/score_reward", score_reward, episode)
            self.writer.add_scalar("episode/custom_reward", custom_reward, episode)
            self.writer.add_scalar("episode/total_reward", total_reward, episode)
            self.writer.add_scalar("episode/length", length, episode)
            self.writer.add_scalar("episode/loss", loss, episode)
            self.writer.add_scalar("episode/epsilon", epsilon, episode)
            self.writer.add_scalar("episode/time", episode_time, episode)
            self.writer.add_scalar("moving_average/score_reward", moving_avg_score, episode)
            self.writer.add_scalar("moving_average/custom_reward", moving_avg_custom, episode)
            self.writer.add_scalar("moving_average/total_reward", moving_avg_total, episode)
            self.writer.add_scalar("best/score_reward", self.best_score_reward, episode)
            self.writer.add_scalar("best/custom_reward", self.best_custom_reward, episode)
            self.writer.add_scalar("best/total_reward", self.best_total_reward, episode)
            self.writer.add_scalar("total_steps", self.total_steps, episode)
            self.writer.add_scalar("game_metrics/lives_lost", lives_lost, episode)
            self.writer.add_scalar("game_metrics/shots_fired", shots_fired, episode)
            self.writer.add_scalar("game_metrics/enemies_killed", enemies_killed, episode)
            
            if learning_rate is not None:
                self.writer.add_scalar("training/learning_rate", learning_rate, episode)
        
        if episode % self.save_frequency == 0:
            self.save_metrics()
            self.auto_save_progress()
            
    def log_training_step(self, gradient_norm: float = None, 
                         q_value_estimate: float = None,
                         replay_buffer_size: int = None, step: int = None) -> None:
        
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
        if q_value_estimate is not None:
            self.q_value_estimates.append(q_value_estimate)
        if replay_buffer_size is not None:
            self.replay_buffer_size.append(replay_buffer_size)


        if self.use_tensorboard and step is not None:
            if gradient_norm is not None:
                self.writer.add_scalar("training/gradient_norm", gradient_norm, step)
            if q_value_estimate is not None:
                self.writer.add_scalar("training/q_value_estimate", q_value_estimate, step)
            if replay_buffer_size is not None:
                self.writer.add_scalar("training/replay_buffer_size", replay_buffer_size, step)
    
    def log_checkpoint(self, episode: int, model_path: str, 
                      total_reward: float, is_best: bool = False) -> None:
        
        checkpoint_info = {
            'episode': episode,
            'model_path': model_path,
            'total_reward': total_reward,
            'timestamp': datetime.now().isoformat(),
            'is_best': is_best
        }
        self.checkpoints.append(checkpoint_info)

        if self.use_tensorboard:
            self.writer.add_scalar("checkpoint/reward", total_reward, episode)
            self.writer.add_scalar("checkpoint/is_best", int(is_best), episode)
    
    def get_current_stats(self, window_size: int = 100) -> Dict[str, float]:

        if len(self.episode_total_rewards) == 0:
            return {}
        
        recent_score_rewards = self.episode_score_rewards[-window_size:]
        recent_custom_rewards = self.episode_custom_rewards[-window_size:]
        recent_total_rewards = self.episode_total_rewards[-window_size:]
        recent_lengths = self.episode_lengths[-window_size:]
        recent_losses = self.episode_losses[-window_size:]
        
        stats = {
            'total_episodes': len(self.episode_total_rewards),
            'total_steps': self.total_steps,
            'best_score_reward': self.best_score_reward,
            'best_custom_reward': self.best_custom_reward,
            'best_total_reward': self.best_total_reward,
            'best_episode': self.best_episode,
            'recent_mean_score_reward': np.mean(recent_score_rewards),
            'recent_std_score_reward': np.std(recent_score_rewards),
            'recent_mean_custom_reward': np.mean(recent_custom_rewards),
            'recent_std_custom_reward': np.std(recent_custom_rewards),
            'recent_mean_total_reward': np.mean(recent_total_rewards),
            'recent_std_total_reward': np.std(recent_total_rewards),
            'recent_mean_length': np.mean(recent_lengths),
            'recent_mean_loss': np.mean(recent_losses),
            'current_epsilon': self.epsilon_values[-1] if self.epsilon_values else 0,
            'training_time_hours': (datetime.now() - self.training_start_time).total_seconds() / 3600
        }
        
        return stats
    
    def save_metrics(self) -> None:

        metrics_dict = {
            'episode_score_rewards': self.episode_score_rewards,
            'episode_custom_rewards': self.episode_custom_rewards,
            'episode_total_rewards': self.episode_total_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_losses': self.episode_losses,
            'epsilon_values': self.epsilon_values,
            'learning_rates': self.learning_rates,
            'moving_avg_score_rewards': self.moving_avg_score_rewards,
            'moving_avg_custom_rewards': self.moving_avg_custom_rewards,
            'moving_avg_total_rewards': self.moving_avg_total_rewards,
            'lives_lost': self.lives_lost,
            'shots_fired': self.shots_fired,
            'enemies_killed': self.enemies_killed,
            'gradient_norms': self.gradient_norms,
            'q_value_estimates': self.q_value_estimates,
            'replay_buffer_size': self.replay_buffer_size,
            'episode_times': self.episode_times,
            'hyperparams': self.hyperparams,
            'checkpoints': self.checkpoints,
            'best_score_reward': self.best_score_reward,
            'best_custom_reward': self.best_custom_reward,
            'best_total_reward': self.best_total_reward,
            'best_episode': self.best_episode,
            'total_steps': self.total_steps,
            'training_start_time': self.training_start_time.isoformat()
        }
        
        with open(os.path.join(self.log_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        with open(os.path.join(self.log_dir, 'training_metrics.pkl'), 'wb') as f:
            pickle.dump(metrics_dict, f)
        
        df = pd.DataFrame({
            'episode': range(len(self.episode_total_rewards)),
            'score_reward': self.episode_score_rewards,
            'custom_reward': self.episode_custom_rewards,
            'total_reward': self.episode_total_rewards,
            'length': self.episode_lengths,
            'loss': self.episode_losses,
            'epsilon': self.epsilon_values,
            'moving_avg_score_reward': self.moving_avg_score_rewards,
            'moving_avg_custom_reward': self.moving_avg_custom_rewards,
            'moving_avg_total_reward': self.moving_avg_total_rewards,
            'episode_time': self.episode_times
        })
        df.to_csv(os.path.join(self.log_dir, 'training_data.csv'), index=False)
    
    def load_metrics(self, metrics_file: str = None) -> None:

        if metrics_file is None:
            metrics_file = os.path.join(self.log_dir, 'training_metrics.pkl')
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'rb') as f:
                metrics_dict = pickle.load(f)
            
            for key, value in metrics_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def plot_training_progress(self, save_path: str = None) -> None:
        
        if len(self.episode_total_rewards) == 0:
            print("No training data to plot")
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        episodes = np.arange(len(self.episode_total_rewards))
        
        axes[0, 0].plot(episodes, self.episode_score_rewards, alpha=0.6, label='Score Reward', color='blue')
        axes[0, 0].plot(episodes, self.moving_avg_score_rewards, 'b-', linewidth=2,
                       label=f'Moving Average ({len(self.score_reward_window)})')
        axes[0, 0].axhline(y=self.best_score_reward, color='darkblue', linestyle='--', 
                          label=f'Best: {self.best_score_reward:.1f}')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Score Reward')
        axes[0, 0].set_title('Score Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(episodes, self.episode_custom_rewards, alpha=0.6, label='Custom Reward', color='green')
        axes[0, 1].plot(episodes, self.moving_avg_custom_rewards, 'g-', linewidth=2,
                       label=f'Moving Average ({len(self.custom_reward_window)})')
        axes[0, 1].axhline(y=self.best_custom_reward, color='darkgreen', linestyle='--', 
                          label=f'Best: {self.best_custom_reward:.1f}')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Custom Reward')
        axes[0, 1].set_title('Custom Rewards')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[0, 2].plot(episodes, self.episode_total_rewards, alpha=0.6, label='Total Reward', color='red')
        axes[0, 2].plot(episodes, self.moving_avg_total_rewards, 'r-', linewidth=2,
                       label=f'Moving Average ({len(self.total_reward_window)})')
        axes[0, 2].axhline(y=self.best_total_reward, color='darkred', linestyle='--', 
                          label=f'Best: {self.best_total_reward:.1f}')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Total Reward')
        axes[0, 2].set_title('Total Rewards (Score + Custom)')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        axes[1, 0].plot(episodes, self.episode_lengths, alpha=0.6)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Episode Length')
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(episodes, self.episode_losses, alpha=0.6)
        if len(self.loss_window) > 0:
            moving_avg_loss = pd.Series(self.episode_losses).rolling(window=min(50, len(self.episode_losses))).mean()
            axes[1, 1].plot(episodes, moving_avg_loss, 'r-', label='Moving Average')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        axes[1, 2].plot(episodes, self.epsilon_values)
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Epsilon')
        axes[1, 2].set_title('Epsilon Decay')
        axes[1, 2].grid(True)
        
        if self.enemies_killed:
            axes[2, 0].plot(episodes, self.enemies_killed, label='Enemies Killed')
            axes[2, 0].plot(episodes, self.lives_lost, label='Lives Lost')
            axes[2, 0].set_xlabel('Episode')
            axes[2, 0].set_ylabel('Count')
            axes[2, 0].set_title('Game Metrics')
            axes[2, 0].legend()
            axes[2, 0].grid(True)
        
        if self.episode_times:
            axes[2, 1].plot(episodes, np.cumsum(self.episode_times) / 3600)  
            axes[2, 1].set_xlabel('Episode')
            axes[2, 1].set_ylabel('Cumulative Time (hours)')
            axes[2, 1].set_title('Training Time')
            axes[2, 1].grid(True)
        
        axes[2, 2].plot(episodes, self.episode_score_rewards, alpha=0.4, label='Score', color='blue')
        axes[2, 2].plot(episodes, self.episode_custom_rewards, alpha=0.4, label='Custom', color='green')
        axes[2, 2].plot(episodes, self.episode_total_rewards, alpha=0.6, label='Total', color='red')
        axes[2, 2].set_xlabel('Episode')
        axes[2, 2].set_ylabel('Reward')
        axes[2, 2].set_title('All Rewards Comparison')
        axes[2, 2].legend()
        axes[2, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.plot_dir, 'training_progress.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if self.use_tensorboard:
            img = plt.imread(save_path)
            self.writer.add_image("training_progress", img, len(self.episode_total_rewards), dataformats='HWC')
        
        plt.show()

    def log_system_metrics(self, episode: int) -> None:
        if self.use_tensorboard and episode % 100 == 0:
            self.writer.add_scalar("system/cpu_percent", psutil.cpu_percent(), episode)
            self.writer.add_scalar("system/memory_percent", psutil.virtual_memory().percent, episode)
            if torch.cuda.is_available():
                self.writer.add_scalar("system/gpu_memory_used_gb", torch.cuda.memory_allocated() / 1024**3, episode)

    def close_tensorboard(self):
        if self.use_tensorboard and hasattr(self, 'writer'):
            self.save_progress("final_progress")
            self.writer.close()
    
    def generate_training_report(self) -> str:
        stats = self.get_current_stats()
        
        report = f"""
        # Training Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        ## Training Summary
        - Total Episodes: {stats.get('total_episodes', 0)}
        - Total Steps: {stats.get('total_steps', 0)}
        - Training Time: {stats.get('training_time_hours', 0):.2f} hours
        - Best Score Reward: {stats.get('best_score_reward', 0):.2f}
        - Best Custom Reward: {stats.get('best_custom_reward', 0):.2f}
        - Best Total Reward: {stats.get('best_total_reward', 0):.2f} (Episode {stats.get('best_episode', 0)})

        ## Recent Performance (last 100 episodes)
        - Mean Score Reward: {stats.get('recent_mean_score_reward', 0):.2f} ± {stats.get('recent_std_score_reward', 0):.2f}
        - Mean Custom Reward: {stats.get('recent_mean_custom_reward', 0):.2f} ± {stats.get('recent_std_custom_reward', 0):.2f}
        - Mean Total Reward: {stats.get('recent_mean_total_reward', 0):.2f} ± {stats.get('recent_std_total_reward', 0):.2f}
        - Mean Episode Length: {stats.get('recent_mean_length', 0):.1f}
        - Mean Loss: {stats.get('recent_mean_loss', 0):.4f}
        - Current Epsilon: {stats.get('current_epsilon', 0):.4f}

        ## Hyperparameters
        """
        for key, value in self.hyperparams.items():
            report += f"- {key}: {value}\n"
        
        return report