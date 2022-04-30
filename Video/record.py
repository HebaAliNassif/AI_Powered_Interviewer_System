import os
import cv2
import config

cap= cv2.VideoCapture(0)

width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs(os.path.split(config.VIDEO_PATH+"/")[0], exist_ok=True)

writer= cv2.VideoWriter(config.VIDEO_PATH + '/basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))


while True:
    ret,frame= cap.read()

    writer.write(frame)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
writer.release()
cv2.destroyAllWindows()