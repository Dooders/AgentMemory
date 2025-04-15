import json
import os
import time
import tkinter as tk
from tkinter import scrolledtext, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from main_demo import MazeEnvironment, MemoryEnhancedAgent, convert_numpy_to_python

# Import the necessary components from your maze demo
from memory import (
    AgentMemorySystem,
    MemoryConfig,
    RedisIMConfig,
    RedisSTMConfig,
    SQLiteLTMConfig,
)
from memory.config import AutoencoderConfig


class TASMVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("TASM Memory System Visualizer")
        self.root.geometry("3800x2300")  # Almost full 4K resolution

        # Modern color scheme
        self.colors = {
            "bg": "#1a1a1a",  # Dark background
            "panel": "#262626",  # Slightly lighter panel background
            "text_bg": "#333333",  # Text area background
            "text": "#ffffff",  # White text
            "accent1": "#00ff9d",  # Bright green for success/target
            "accent2": "#00b8ff",  # Bright blue for agent
            "accent3": "#ff3860",  # Bright red for obstacles
            "grid": "#404040",  # Subtle grid lines
            "button": "#00d1b2",  # Teal buttons
            "button_hover": "#00e6c5",  # Lighter teal for hover
            "label_bg": "#2d2d2d",  # Slightly lighter than bg for labels
            "value_bg": "#3d3d3d",  # Even lighter for value labels
        }

        self.root.configure(bg=self.colors["bg"])

        # Set very large default font size for all widgets
        default_font = ("Segoe UI", 24)
        self.root.option_add("*Font", default_font)

        # Create a dark theme
        self.create_dark_theme()

        # Configure memory system
        self.setup_memory_system()

        # Create maze environment
        self.maze_size = 10
        self.obstacles = [(3, 3), (3, 4), (3, 5), (7, 7), (8, 7), (9, 7)]
        self.env = MazeEnvironment(
            size=self.maze_size, obstacles=self.obstacles, max_steps=100
        )

        # Create agent
        self.agent = self.create_agent()

        # Setup the UI
        self.create_ui()

        # State variables
        self.is_running = False
        self.current_step = 0
        self.episode = 0
        self.total_reward = 0
        self.episode_rewards = []
        self.episode_steps = []
        self.memory_contents = {"STM": [], "IM": [], "LTM": []}

        # Initialize maze
        self.reset_environment()
        self.draw_maze()

        # Configure window resizing behavior
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    def create_dark_theme(self):
        """Create a custom dark theme for ttk widgets"""
        style = ttk.Style()

        # Try to use a theme that supports customization
        try:
            style.theme_use("clam")  # 'clam' is usually more customizable
        except:
            pass  # If the theme doesn't exist, continue with default

        # Configure base styles
        style.configure("TFrame", background=self.colors["bg"])
        style.configure(
            "TLabel",
            background=self.colors["bg"],
            foreground=self.colors["text"],
            font=("Segoe UI", 24),
        )
        style.configure("TLabelframe", background=self.colors["panel"])
        style.configure(
            "TLabelframe.Label",
            font=("Segoe UI", 24, "bold"),
            foreground=self.colors["text"],
            background=self.colors["panel"],
        )
        style.configure(
            "TButton", font=("Segoe UI", 24, "bold"), background=self.colors["button"]
        )

        # Custom tabbed widget styling
        style.configure("TNotebook", background=self.colors["bg"])

        # Tab styling - no truncation
        style.configure(
            "TNotebook.Tab",
            font=("Segoe UI", 22, "bold"),
            padding=[20, 10],
            background=self.colors["bg"],
            foreground=self.colors["text"],
            width=25,  # Wider tabs
            borderwidth=0,
        )

        # Tab state-specific styles
        style.map(
            "TNotebook.Tab",
            background=[
                ("selected", self.colors["panel"]),
                ("active", self.colors["button"]),
                ("!active", self.colors["bg"]),
            ],
            foreground=[
                ("selected", self.colors["accent2"]),
                ("active", self.colors["text"]),
                ("!active", self.colors["text"]),
            ],
        )

        # Fix tab styling with global options
        self.root.option_add("*TNotebook.Tab*Foreground", self.colors["text"])
        self.root.option_add("*TNotebook*background", self.colors["bg"])

        # Fix scrollbar colors
        style.configure(
            "TScrollbar",
            background=self.colors["bg"],
            troughcolor=self.colors["panel"],
            bordercolor=self.colors["bg"],
            arrowcolor=self.colors["text"],
            arrowsize=24,
        )

        # Configure status label styles
        style.configure(
            "StatusLabel.TLabel",
            font=("Segoe UI", 28, "bold"),
            background=self.colors["bg"],
            foreground=self.colors["text"],
            padding=[20, 10],
        )

        style.configure(
            "ValueLabel.TLabel",
            font=("Segoe UI", 32, "bold"),
            background=self.colors["bg"],
            foreground=self.colors["accent2"],
            padding=[20, 10],
        )

        # Configure frame styles
        style.configure("MemoryFrame.TFrame", background=self.colors["panel"])

    def setup_memory_system(self):
        """Configure and initialize the memory system"""
        # Configure autoencoder
        autoencoder_config = AutoencoderConfig(
            use_neural_embeddings=False,
            epochs=1,
            batch_size=1,
            vector_similarity_threshold=0.6,
        )

        # Configure STM
        stm_config = RedisSTMConfig(
            ttl=120,
            memory_limit=500,
            use_mock=True,
        )

        # Configure IM
        im_config = RedisIMConfig(
            ttl=240,
            memory_limit=1000,
            compression_level=0,
            use_mock=True,
        )

        # Configure LTM
        db_path = "visual_memory_demo.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        ltm_config = SQLiteLTMConfig(
            compression_level=0,
            batch_size=20,
            db_path=db_path,
        )

        # Create main memory config
        self.memory_config = MemoryConfig(
            stm_config=stm_config,
            im_config=im_config,
            ltm_config=ltm_config,
            autoencoder_config=autoencoder_config,
            cleanup_interval=1000,
            enable_memory_hooks=False,
        )

        # Initialize memory system
        self.memory_system = AgentMemorySystem.get_instance(self.memory_config)

    def create_agent(self):
        """Create a memory-enhanced agent"""
        agent_id = f"visual_agent_{int(time.time())}"
        config = {"memory_config": self.memory_config}
        agent = MemoryEnhancedAgent(agent_id, config=config, action_space=4)
        return agent

    def create_ui(self):
        """Create the UI components"""
        # Main frame using grid instead of pack for better control
        main_frame = ttk.Frame(self.root)
        main_frame.grid(
            row=0, column=0, sticky="nsew", padx=5, pady=5
        )  # Reduced padding
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        # Left panel - Maze visualization
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=2)  # Reduced padding
        left_frame.grid_rowconfigure(0, weight=3)
        left_frame.grid_rowconfigure(1, weight=1)

        # Maze canvas with 4K-appropriate size
        self.maze_frame = ttk.LabelFrame(left_frame, text="Maze Environment")
        self.maze_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.cell_size = 100  # Reduced from 160 to make the maze smaller
        canvas_width = self.maze_size * self.cell_size
        canvas_height = self.maze_size * self.cell_size
        self.canvas = tk.Canvas(
            self.maze_frame,
            width=canvas_width,
            height=canvas_height,
            bg=self.colors["panel"],
            highlightthickness=0,
        )
        self.canvas.pack(padx=10, pady=10, expand=True)

        # Performance metrics with better proportions
        self.metrics_frame = ttk.LabelFrame(left_frame, text="Performance Metrics")
        self.metrics_frame.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        # Create a matplotlib figure with 4K-appropriate size and DPI
        plt.rcParams.update(
            {
                "font.size": 20,  # Larger font size
                "lines.linewidth": 4,  # Thicker lines
                "axes.labelsize": 22,  # Larger axis labels
                "axes.titlesize": 24,  # Larger titles
                "xtick.labelsize": 20,  # Larger tick labels
                "ytick.labelsize": 20,
                "axes.facecolor": self.colors["panel"],
                "figure.facecolor": self.colors["bg"],
                "text.color": self.colors["text"],
                "axes.labelcolor": self.colors["text"],
                "axes.edgecolor": self.colors["grid"],
                "xtick.color": self.colors["text"],
                "ytick.color": self.colors["text"],
                "grid.color": self.colors["grid"],
                "grid.alpha": 0.3,
            }
        )

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(30, 8), dpi=100)
        self.fig.patch.set_facecolor(self.colors["bg"])

        # Add the plot to the Tkinter window
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.metrics_frame)
        self.canvas_plot.draw()
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Right panel - Memory visualization with improved layout
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        right_frame.grid_rowconfigure(0, weight=4)
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Memory tiers display with improved tab labels
        self.memory_notebook = ttk.Notebook(right_frame)
        self.memory_notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Create memory tier frames
        self.stm_frame = ttk.Frame(self.memory_notebook, style="MemoryFrame.TFrame")
        self.im_frame = ttk.Frame(self.memory_notebook, style="MemoryFrame.TFrame")
        self.ltm_frame = ttk.Frame(self.memory_notebook, style="MemoryFrame.TFrame")

        # Configure frame weights
        for frame in [self.stm_frame, self.im_frame, self.ltm_frame]:
            frame.grid_rowconfigure(0, weight=1)
            frame.grid_columnconfigure(0, weight=1)

        # Add tabs with improved text
        self.memory_notebook.add(self.stm_frame, text="  STM  ")
        self.memory_notebook.add(self.im_frame, text="  ITM  ")
        self.memory_notebook.add(self.ltm_frame, text="  LTM  ")

        # Text widgets for memory contents with improved styling
        font_config = ("Consolas", 20)
        text_config = {
            "wrap": tk.WORD,
            "font": font_config,
            "bg": self.colors["text_bg"],
            "fg": self.colors["text"],
            "insertbackground": self.colors["text"],  # Cursor color
            "selectbackground": self.colors["button"],  # Selection background
            "selectforeground": self.colors["text"],  # Selection text color
            "relief": "flat",
            "borderwidth": 0,
            "padx": 10,
            "pady": 10,
        }

        # Create and grid the text widgets to fill their containers
        self.stm_text = scrolledtext.ScrolledText(self.stm_frame, **text_config)
        self.stm_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.im_text = scrolledtext.ScrolledText(self.im_frame, **text_config)
        self.im_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.ltm_text = scrolledtext.ScrolledText(self.ltm_frame, **text_config)
        self.ltm_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Current state display with improved styling
        self.state_frame = ttk.LabelFrame(right_frame, text="Current State")
        self.state_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.state_frame.grid_rowconfigure(0, weight=1)
        self.state_frame.grid_columnconfigure(0, weight=1)

        self.state_text = scrolledtext.ScrolledText(
            self.state_frame, height=8, **text_config
        )
        self.state_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Controls at bottom
        self.controls_frame = ttk.Frame(self.root)
        self.controls_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=2)

        # Status indicators with better spacing and visibility
        self.status_frame = ttk.Frame(self.controls_frame)
        self.status_frame.pack(
            side=tk.LEFT, fill=tk.X, expand=True, pady=20
        )  # Added vertical padding

        # Status labels with improved visibility
        label_width = 12

        # Episode counter with proper background
        episode_container = ttk.Frame(self.status_frame, style="TFrame")
        episode_container.grid(row=0, column=0, padx=20)
        episode_container.configure(
            style="TFrame"
        )  # Ensure frame has correct background

        episode_label = ttk.Label(
            episode_container, text="EPISODE", style="StatusLabel.TLabel"
        )
        episode_label.pack(pady=(0, 5))
        self.episode_label = ttk.Label(
            episode_container, text="0", style="ValueLabel.TLabel", width=label_width
        )
        self.episode_label.pack()

        # Step counter with proper background
        step_container = ttk.Frame(self.status_frame, style="TFrame")
        step_container.grid(row=0, column=1, padx=20)
        step_container.configure(style="TFrame")  # Ensure frame has correct background

        step_label = ttk.Label(step_container, text="STEP", style="StatusLabel.TLabel")
        step_label.pack(pady=(0, 5))
        self.step_label = ttk.Label(
            step_container, text="0", style="ValueLabel.TLabel", width=label_width
        )
        self.step_label.pack()

        # Reward counter with proper background
        reward_container = ttk.Frame(self.status_frame, style="TFrame")
        reward_container.grid(row=0, column=2, padx=20)
        reward_container.configure(
            style="TFrame"
        )  # Ensure frame has correct background

        reward_label = ttk.Label(
            reward_container, text="REWARD", style="StatusLabel.TLabel"
        )
        reward_label.pack(pady=(0, 5))
        self.reward_label = ttk.Label(
            reward_container, text="0.0", style="ValueLabel.TLabel", width=label_width
        )
        self.reward_label.pack()

        # Control buttons with 4K-appropriate width and font
        self.button_frame = ttk.Frame(self.controls_frame)
        self.button_frame.pack(side=tk.RIGHT)

        button_width = 20
        button_font = ("Segoe UI", 24, "bold")

        self.step_button = ttk.Button(
            self.button_frame,
            text="Step",
            width=button_width,
            command=self.step_simulation,
        )
        self.step_button.pack(side=tk.LEFT, padx=10)

        self.start_button = ttk.Button(
            self.button_frame,
            text="Start",
            width=button_width,
            command=self.toggle_simulation,
        )
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.reset_button = ttk.Button(
            self.button_frame,
            text="Reset",
            width=button_width,
            command=self.reset_simulation,
        )
        self.reset_button.pack(side=tk.LEFT, padx=10)

        # Speed control with 4K-appropriate scaling
        ttk.Label(self.button_frame, text="Speed:", font=button_font).pack(
            side=tk.LEFT, padx=20
        )
        self.speed_var = tk.DoubleVar(value=0.5)
        self.speed_scale = ttk.Scale(
            self.button_frame,
            from_=0.1,
            to=2.0,
            variable=self.speed_var,
            orient=tk.HORIZONTAL,
            length=500,
        )
        self.speed_scale.pack(side=tk.LEFT, padx=10)

    def draw_maze(self):
        """Draw the maze on canvas with updated colors"""
        self.canvas.delete("all")
        self.canvas.configure(bg=self.colors["panel"])

        # Draw cells
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                # Determine cell color
                if (i, j) == self.env.position:
                    color = self.colors["accent2"]  # Bright blue for agent
                elif (i, j) == self.env.target:
                    color = self.colors["accent1"]  # Bright green for target
                elif (i, j) in self.obstacles:
                    color = self.colors["accent3"]  # Bright red for obstacles
                else:
                    color = self.colors["panel"]  # Background color

                # Draw cells with thicker borders
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline=self.colors["grid"], width=4
                )

        # Add grid lines
        for i in range(self.maze_size + 1):
            self.canvas.create_line(
                0,
                i * self.cell_size,
                self.maze_size * self.cell_size,
                i * self.cell_size,
                fill=self.colors["grid"],
                width=4,
            )
            self.canvas.create_line(
                i * self.cell_size,
                0,
                i * self.cell_size,
                self.maze_size * self.cell_size,
                fill=self.colors["grid"],
                width=4,
            )

    def reset_environment(self):
        """Reset the environment to start a new episode"""
        self.observation = self.env.reset()
        self.agent.current_observation = convert_numpy_to_python(self.observation)
        self.current_step = 0
        self.total_reward = 0
        self.update_state_display()

    def update_state_display(self):
        """Update the current state display"""
        self.state_text.delete(1.0, tk.END)
        state_str = json.dumps(convert_numpy_to_python(self.observation), indent=2)
        self.state_text.insert(tk.END, state_str)

    def update_memory_displays(self):
        """Update the memory tier displays"""
        agent_id = self.agent.agent_id

        # Get STM memories
        try:
            memory_agent = self.memory_system.get_memory_agent(agent_id)
            stm_memories = memory_agent.stm_store.get_all(agent_id)
            self.memory_contents["STM"] = stm_memories
            self.stm_text.delete(1.0, tk.END)
            self.stm_text.insert(tk.END, f"STM Memories: {len(stm_memories)}\n\n")
            for i, mem in enumerate(stm_memories[:10]):  # Show top 10
                self.stm_text.insert(tk.END, f"Memory {i+1}:\n")
                self.stm_text.insert(tk.END, json.dumps(mem, indent=2) + "\n\n")
        except Exception as e:
            self.stm_text.delete(1.0, tk.END)
            self.stm_text.insert(tk.END, f"Error retrieving STM: {str(e)}")

        # Get IM memories
        try:
            im_memories = memory_agent.im_store.get_all(agent_id)
            self.memory_contents["IM"] = im_memories
            self.im_text.delete(1.0, tk.END)
            self.im_text.insert(tk.END, f"IM Memories: {len(im_memories)}\n\n")
            for i, mem in enumerate(im_memories[:10]):  # Show top 10
                self.im_text.insert(tk.END, f"Memory {i+1}:\n")
                self.im_text.insert(tk.END, json.dumps(mem, indent=2) + "\n\n")
        except Exception as e:
            self.im_text.delete(1.0, tk.END)
            self.im_text.insert(tk.END, f"Error retrieving IM: {str(e)}")

        # Get LTM memories
        try:
            ltm_memories = memory_agent.ltm_store.get_all(limit=1000)
            self.memory_contents["LTM"] = ltm_memories
            self.ltm_text.delete(1.0, tk.END)
            self.ltm_text.insert(tk.END, f"LTM Memories: {len(ltm_memories)}\n\n")
            for i, mem in enumerate(ltm_memories[:10]):  # Show top 10
                self.ltm_text.insert(tk.END, f"Memory {i+1}:\n")
                self.ltm_text.insert(tk.END, json.dumps(mem, indent=2) + "\n\n")
        except Exception as e:
            self.ltm_text.delete(1.0, tk.END)
            self.ltm_text.insert(tk.END, f"Error retrieving LTM: {str(e)}")

    def update_metrics(self):
        """Update performance metrics plots with better styling"""
        # Clear the axes
        self.ax1.clear()
        self.ax2.clear()

        # Plot episode rewards with improved styling
        if self.episode_rewards:
            self.ax1.plot(self.episode_rewards, "b-", linewidth=2)
            self.ax1.set_title("Rewards per Episode", pad=10, fontsize=10)
            self.ax1.set_xlabel("Episode", fontsize=9)
            self.ax1.set_ylabel("Total Reward", fontsize=9)
            self.ax1.grid(True, linestyle="--", alpha=0.7)

        # Plot episode steps with improved styling
        if self.episode_steps:
            self.ax2.plot(self.episode_steps, "r-", linewidth=2)
            self.ax2.set_title("Steps per Episode", pad=10, fontsize=10)
            self.ax2.set_xlabel("Episode", fontsize=9)
            self.ax2.set_ylabel("Steps", fontsize=9)
            self.ax2.grid(True, linestyle="--", alpha=0.7)

        # Update the canvas with proper spacing
        self.fig.tight_layout()
        self.canvas_plot.draw()

    def step_simulation(self):
        """Execute a single step in the simulation"""
        # Get action from agent
        epsilon = max(0.05, 0.5 - (self.episode / 100))
        action = self.agent.act(self.observation, epsilon)

        # Take the action in the environment
        next_observation, reward, done = self.env.step(action.params["direction"])

        # Update Q-values
        self.agent.update_q_value(
            self.observation, action.params["direction"], reward, next_observation, done
        )

        # Update state
        self.observation = next_observation
        self.agent.current_observation = convert_numpy_to_python(self.observation)
        self.current_step += 1
        self.total_reward += reward

        # Update UI
        self.update_labels()
        self.draw_maze()
        self.update_state_display()
        self.update_memory_displays()

        # Check if episode is done
        if done:
            print(
                f"Episode {self.episode+1} completed: steps={self.env.steps}, reward={self.total_reward:.1f}"
            )
            self.episode_rewards.append(self.total_reward)
            self.episode_steps.append(self.env.steps)
            self.update_metrics()
            self.episode += 1
            self.reset_environment()

        return done

    def update_labels(self):
        """Update status labels with better formatting"""
        self.episode_label.config(text=str(self.episode))
        self.step_label.config(text=str(self.current_step))
        self.reward_label.config(text=f"{self.total_reward:.1f}")

    def toggle_simulation(self):
        """Start or stop the continuous simulation"""
        self.is_running = not self.is_running
        self.start_button.config(text="Stop" if self.is_running else "Start")

        if self.is_running:
            self.run_simulation()

    def run_simulation(self):
        """Run the simulation continuously in a background thread"""
        if not self.is_running:
            return

        # Run one step
        done = self.step_simulation()

        # Schedule the next step
        delay = 1.0 / self.speed_var.get()  # Calculate delay based on speed
        self.root.after(int(delay * 1000), self.run_simulation)

    def reset_simulation(self):
        """Reset the entire simulation"""
        # Stop if running
        self.is_running = False
        self.start_button.config(text="Start")

        # Reset environment and agent
        self.reset_environment()
        self.draw_maze()

        # Reset metrics
        self.episode = 0
        self.episode_rewards = []
        self.episode_steps = []
        self.update_metrics()
        self.update_labels()

        # Clear memory
        try:
            self.memory_system.clear_memories(self.agent.agent_id)
        except:
            pass
        self.update_memory_displays()


# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = TASMVisualizer(root)
    root.mainloop()

    # Clean up
    if os.path.exists("visual_memory_demo.db"):
        try:
            os.remove("visual_memory_demo.db")
        except:
            pass
