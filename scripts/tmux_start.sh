#!/bin/bash

session="spomp"

tmux new-session -d -s $session

window=0
tmux rename-window -t $session:$window 'roscore'
tmux send-keys -t $session:$window 'roscore' C-m

window=1
tmux new-window -t $session:$window -n 'drivers'
tmux send-keys -t $session:$window 'roslaunch jackal_launch jackal_hw.launch ouster:=true realsense:=true' C-m

window=2
tmux new-window -t $session:$window -n 'spomp'
tmux send-keys -t $session:$window 'roslaunch spomp spomp.launch'

window=3
tmux new-window -t $session:$window -n 'db'
tmux send-keys -t $session:$window 'roslaunch comm_stack_launch jackal_comms.launch'

window=4
tmux new-window -t $session:$window -n 'goal_manager'
tmux send-keys -t $session:$window 'roslaunch spomp goal_manager.launch'

window=5
tmux new-window -t $session:$window -n 'detector'
tmux send-keys -t $session:$window 'roslaunch dcist_obstacle_distance target_distance_jackal.launch'

window=6
tmux new-window -t $session:$window -n 'scratch'

tmux attach-session -t $session

