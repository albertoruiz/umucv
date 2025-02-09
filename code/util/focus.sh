# sudo apt install v4l-utils
# ajuste manual de enfoque

FOCUS=$2
DEV=$1

[ -z "$2" ] && DEV='0' && FOCUS=$1

v4l2-ctl -d /dev/video$DEV -c focus_auto=0
v4l2-ctl -d /dev/video$DEV -c focus_absolute=$FOCUS
