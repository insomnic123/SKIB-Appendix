# Full File Found in Overall Folder
### --- PART 3: Draw both paths --- ###
canvas1 = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
canvas2 = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

for i in range(len(aligned_tape_points) - 1):
    cv2.line(canvas2, aligned_tape_points[i], aligned_tape_points[i+1], (0, 0, 0), 2)
for i in range(1, len(positions)):
    cv2.line(canvas1, positions[i-1], positions[i], (0, 0, 255), 2)

cv2.imwrite(output_image_red, canvas1)
cv2.imwrite(output_image_black, canvas2)

### --- PART 4: Deviation Analysis --- ###
def interpolate_path(points, num_points=2000):
    # interpolation
    from scipy.interpolate import interp1d

    points = sorted(points, key=lambda p: p[1])  # sort by Y
    xs, ys = zip(*points)
    t = np.linspace(0, 1, len(xs))
    t_interp = np.linspace(0, 1, num_points)
    fx = interp1d(t, xs, kind='linear', fill_value="extrapolate")
    fy = interp1d(t, ys, kind='linear', fill_value="extrapolate")
    interpolated = np.stack([fx(t_interp), fy(t_interp)], axis=-1)
    return interpolated.astype(np.int32)

def find_line_deviation(red_path, black_path, csv_output_path, output_image_path):
    image_red = cv2.imread(red_path)
    image_black = cv2.imread(black_path)
    hsv_red = cv2.cvtColor(image_red, cv2.COLOR_BGR2HSV)
    hsv_black = cv2.cvtColor(image_black, cv2.COLOR_BGR2HSV)

    # Detect red
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv_red, lower_red1, upper_red1) | cv2.inRange(hsv_red, lower_red2, upper_red2)

    # Detect black
    gray = cv2.cvtColor(image_black, cv2.COLOR_BGR2GRAY)
    _, black_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    #black_only_mask = cv2.bitwise_and(black_thresh, cv2.bitwise_not(red_mask)) DO NOT USE!!!
    black_only_mask = black_thresh

    #cv2.imshow("Black Mask", black_thresh)
    #cv2.waitKey(0)

    contours_black, _ = cv2.findContours(black_only_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x = image_red.copy()
    cv2.drawContours(x, contours_red, -1, (0, 255, 0), 2)
    cv2.imshow("hi", x)

    
    if not contours_black or not contours_red:
        raise ValueError("Could not detect both lines.")

    black_pts_raw = max(contours_black, key=cv2.contourArea).reshape(-1, 2)
    red_pts = max(contours_red, key=cv2.contourArea).reshape(-1, 2)

    # Fill gaps in black path
    black_pts_interp = interpolate_path(black_pts_raw, num_points=3000)
    
    lx = []
    ly = []
    for bruh in black_pts_interp:
        lx.append(float(bruh[0]))
        ly.append(float(bruh[1]))

    lxr = []
    lyr = []
    for bruh2 in red_pts:
        lxr.append(float(bruh2[0]))
        lyr.append(float(bruh2[1]))

    point_dict_black = {}
    for i in range(len(ly)):
        all_x = []
        for j in range(len(lx)):
            if ly[j] == ly[i]:
                all_x.append(lx[j])
        point_dict_black[ly[i]] = max(all_x)

    point_dict_red = {}
    for i in range(len(lyr)):
        all_x = []
        for j in range(len(lxr)):
            if lyr[j] == lyr[i]:
                all_x.append(lxr[j])
        point_dict_red[lyr[i]] = all_x[0]

    with open(csv_output_path, "w") as f:
        f.write("Red_X,Red_Y,Black_X,Black_Y,Deviation_X\n")

        black_y = list(point_dict_black.keys())
        black_x = list(point_dict_black.values())
        red_y = list(point_dict_red.keys())
        red_x = list(point_dict_red.values())

        for i in range(len(black_y)):
            if i < len(red_y):
                f.write(f"{red_x[i]},{red_y[i]},{black_x[i]},{black_y[i]},{abs(black_x[i]-red_x[i])}\n")

    fig, ax = plt.subplots()
    ax.scatter(list(point_dict_black.keys()), list(point_dict_black.values()), s = 0.1)
    ax.scatter(list(point_dict_red.keys()), list(point_dict_red.values()), s = 0.1)
    ax.set(xlim=(0, 1500), xticks=np.arange(0, 500, 100),
       ylim=(0, 1500), yticks=np.arange(0, 500, 100))
    plt.savefig(output_image_path)


def plot_deviation_area(csv_path):
    red_x, red_y, black_x, black_y = [], [], [], []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            red_x.append(float(row['Red_X']))
            red_y.append(float(row['Red_Y']))
            black_x.append(float(row['Black_X']))
            black_y.append(float(row['Black_Y']))

    # Sort points by red_y (along-path direction)
    combined = sorted(zip(red_y, red_x, black_x, black_y), key=lambda p: p[0])
    red_y_sorted, red_x_sorted, black_x_sorted, black_y_sorted = zip(*combined)

    plt.figure(figsize=(10, 6))

    # image_width = frame_width
    # image_height = frame_height

    # red_x_flipped = [image_width - x for x in red_x_sorted]
    # red_y_flipped = [image_height - y for y in red_y_sorted]

    # plt.plot(red_y_flipped, red_x_flipped, label='Red Path (Bot)', color='red')

    plt.plot(red_y_sorted, red_x_sorted, label='Red Path (Bot)', color='red')
    plt.plot(black_y_sorted, black_x_sorted, label='Black Path (Tape)', color='black')
    plt.fill_between(red_y_sorted, red_x_sorted, black_x_sorted, color='purple', alpha=0.3, label='Deviation Area')
    plt.xlabel('Y-coordinate (along path)')
    plt.ylabel('X-coordinate (position)')
    plt.title('Deviation Between Bot Path and Tape Path')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(deviation_output_graph)
    plt.show()

def calculate_values(csv_path, result_path):

    deviation_sum = 0
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            deviation_sum += float(row["Deviation_X"])

    x_deviation_sum = (deviation_sum/aruco_image_length_pixels)*aruco_image_length_real
    y_deviation = (1/aruco_image_length_pixels)*aruco_image_length_real
    scaled_deviation = x_deviation_sum*y_deviation
    
    with open(result_path, "w") as f:
        f.write(str(scaled_deviation))

### --- Run Analysis --- ###
find_line_deviation(output_image_red, output_image_black, deviation_csv, deviation_output_image)
plot_deviation_area(deviation_csv)
calculate_values(deviation_csv, deviation_result)