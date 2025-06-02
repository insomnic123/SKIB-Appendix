# Full File Found in Overall Folder
min_contour_area = 50
min_circularity = 0.71
positions = []

cap = cv2.VideoCapture(video_path)
frame_width, frame_height = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_width is None or frame_height is None:
        frame_height, frame_width = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([ 5, 150, 140])
    upper_orange = np.array([ 20, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)

        if area > min_contour_area and circularity > min_circularity:
            (x, y), _ = cv2.minEnclosingCircle(cnt)
            positions.append((int(x), int(y)))

            cv2.circle(frame, (int(x), int(y)), min_contour_area, (0, 255, 0), 2)  # Green circle with radius 10

            break

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)


cap.release()