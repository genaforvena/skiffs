# Check if directory argument is provided
if [ -z "$1" ]; then
	echo "Please provide a directory."
	exit 1
fi

# Start a new tmux session
tmux new-session -s skiffs -d

# Split the window into four panes
tmux split-window -h
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v

# Navigate to the specified directory and activate virtual environment in each pane
for pane in {0..3}; do
	tmux select-pane -t $pane
	tmux send-keys "cd ${1}" C-m "source venv/bin/activate" C-m
done

# Start nvim in pane 0
tmux select-pane -t 0
tmux send-keys "nvim" C-m

# Start OpenAI REPL in pane 1 (replace 'openai_repl_command' with the actual command)
tmux select-pane -t 1
tmux send-keys "openai repl" C-m

# Attach to the session
tmux attach-session -t skiffs
