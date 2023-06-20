while getopts m:n:f: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        n) number=${OPTARG};;
        f) fullname=${OPTARG};;
    esac
done

tmux new-session -d -s $model #"$1_$2"
tmux set -g mouse on
#declare -i num_train=50
#declare -i num_test=50
#declare -i max_actions=-1

for ((i=0; i<$number; i++)); do
    com="python main.py"
    echo $com
    tmux send "source /env/bin/activate" C-m
    tmux send "$com" C-m
    tmux new-window
    sleep 5
done
