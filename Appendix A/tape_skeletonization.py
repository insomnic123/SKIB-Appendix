# Full File Found in Overall Folder
### --- PART 2: Extract and align tape path (skeleton) --- ###
def skeletonize(img):
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open_img)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

tape_img = cv2.imread(tape_img_path)
resized_tape = cv2.resize(tape_img, (frame_width, frame_height))
gray = cv2.cvtColor(resized_tape, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
_, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
#copy = np.copy(resized_tape)
#cv2.drawContours(copy, filtered_contours, -1, (0, 255, 0), 2)
#cv2.imshow("skib", copy)

tape_points = []
for cnt in filtered_contours:
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    skeleton = skeletonize(mask)
    ys, xs = np.where(skeleton == 255)
    tape_points += list(zip(xs, ys))

tape_points = sorted(tape_points, key=lambda p: p[1])  # top-to-bottom
tape_offset = 40
tape_points = tape_points[tape_offset::]

if tape_points and positions:
    dx = positions[0][0] - tape_points[0][0]
    dy = positions[0][1] - tape_points[0][1]
    aligned_tape_points = [(x + dx, y + dy) for (x, y) in tape_points]
else:
    aligned_tape_points = tape_points

