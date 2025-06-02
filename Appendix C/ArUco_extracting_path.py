# Full File Found in Overall Folder
### --- PART 1: Extract bot path from video --- ###
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

positions = []

cap = cv2.VideoCapture(video_path)
frame_width, frame_height = None, None

# defining dictionaries 
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()
aruco_image_length_pixels = None
aruco_image_length_real = 2.4

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_width is None or frame_height is None:
        frame_height, frame_width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, _ = detector.detectMarkers(gray)

    # mask f same size
    mask = np.zeros_like(frame)

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == 50:
                c = corners[i][0]  # four fours my word fham
                center_x = int(np.mean(c[:, 0]))
                center_y = int(np.mean(c[:, 1]))

                if aruco_image_length_pixels == None:
                    aruco_image_length_pixels = c[-1][0] - c[0][0]
                    print(aruco_image_length_pixels)

                positions.append((center_x, center_y))

                # draw dot
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                pts = np.array([c], dtype=np.int32)
                cv2.fillPoly(mask, pts, (0, 255, 0)) 

                break

    # blend frame and mask
    overlayed = cv2.addWeighted(frame, 1.0, mask, 0.5, 0)

    cv2.imshow("Masked Marker Tracking", overlayed)
    cv2.waitKey(5)

cap.release()
