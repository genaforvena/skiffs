# Start a new tmux session in current directory
tmux new-session -s skiffs -d

# Split the window into four panes
tmux split-window -h
tmux split-window -v
tmux select-pane -t 0

# Navigate to the setup script directory and activate virtual environment in each panes

for pane in {0..2}; do
	tmux select-pane -t $pane
	tmux send-keys "cd $(pwd)" C-m "source venv/bin/activate" C-m
done

# Start nvim in pane 0
tmux select-pane -t 0
tmux resize-pane -R 30
tmux send-keys "nvim" C-m

# Attach to the session
tmux attach-session -t skiffs
