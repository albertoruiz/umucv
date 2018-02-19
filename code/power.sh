# compensa la oscilación de la luz eléctrica
v4l2-ctl -d /dev/video0 -c power_line_frequency=1
