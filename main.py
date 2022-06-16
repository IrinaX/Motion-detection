import cv2

# load the video
video = cv2.VideoCapture("videos/traffic.mp4")
# get first two frames
_, frame1 = video.read()
grabbed, frame2 = video.read()

while video.isOpened():
    # if we have reached the end of the video we leave
    if not grabbed:
        break
    # find a difference between tho frames using absdiff
    diff = cv2.absdiff(frame1, frame2)
    # convert difference to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # apply smoothing to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # apply threshold
    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY)
    # expand the threshold to cover holes
    dilated = cv2.dilate(thresh, None, iterations=10)
    # find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Loop over each contour
    for contour in contours:
        # Calc bounding box
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 300:
            # Draw rectangle - bounding box
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("result", frame1)
    frame1 = frame2
    # get next frame
    grabbed, frame2 = video.read()

    # if esc is pressed
    if cv2.waitKey(30) == 27:
        break

cv2.destroyAllWindows()
video.release()
