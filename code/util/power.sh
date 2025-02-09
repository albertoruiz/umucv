# sudo apt install v4l-utils
# compensa la oscilación de la luz eléctrica

DEV=$1

[ -z "$1" ] && DEV='0'

v4l2-ctl -d /dev/video$DEV -c power_line_frequency=1

