# based on
# https://gist.github.com/marc-hanheide/7b3557f487f1353b2b7c

PORT=${2:-8090}
echo $PORT

DISPLAY=:0 cvlc -q --no-audio $1 --sout "#transcode{vcodec=MJPG}:standard{access=http,mux=mpjpeg,dst=:$PORT}" --sout-http-mime="multipart/x-mixed-replace;boundary=--7b3cc56e5f51db803f790dad720ed50a"
