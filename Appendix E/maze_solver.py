# Full File Found in Overall Folder
import cv2
import numpy as np
from skimage.morphology import skeletonize
from collections import deque

start_point = None
end_point = None
points_clicked = []

def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, points_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        points_clicked.append((x, y))
        if len(points_clicked) == 1:
            start_point = (x, y)
            print(f"Start: {start_point}")
        elif len(points_clicked) == 2:
            end_point = (x, y)
            print(f"End: {end_point}")

def preprocess_maze(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def get_skeleton(binary_image):
    # treat 0-pixels (the corridors) as foreground
    path = binary_image == 0
    skeleton = skeletonize(path).astype(np.uint8) * 255
    return skeleton

def bfs_on_skeleton(skeleton, start, end):
    h, w = skeleton.shape
    visited = np.zeros((h, w), dtype=bool)
    prev = np.full((h, w, 2), -1, dtype=int)
    queue = deque()
    queue.append(start)
    visited[start[1], start[0]] = True

    directions = [
    (-1,  0), ( 1,  0), ( 0, -1), ( 0,  1),
    (-1, -1), (-1,  1), ( 1, -1), ( 1,  1)
]

    found = False

    while queue:
        x, y = queue.popleft()

        # allow fuzzy match to account for pixel offset
        if abs(x - end[0]) <= 1 and abs(y - end[1]) <= 1:
            end = (x, y)  # snap to reachable point
            found = True
            break

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if not visited[ny, nx] and skeleton[ny, nx] == 255:
                    queue.append((nx, ny))
                    visited[ny, nx] = True
                    prev[ny, nx] = (x, y)

    if not found:
        print("No path found to end.")
        return []

    path = []
    curr = end
    while curr != start:
        path.append(curr)
        prev_point = prev[curr[1], curr[0]]
        if prev_point[0] == -1:
            print("No skeleton path found (broken chain).")
            return []
        curr = tuple(prev_point)
    path.append(start)
    path.reverse()
    return path


def draw_centered_path(image, path, color=(0,0,255), thickness=2):
    for i in range(1, len(path)):
        pt1 = path[i-1]
        pt2 = path[i]
        cv2.line(image, pt1, pt2, color, thickness)

def main():
    global start_point, end_point

    image = cv2.imread("bigmaze.png")
    display = image.copy()

    cv2.namedWindow("Click Start and End Points")
    cv2.setMouseCallback("Click Start and End Points", mouse_callback)

    while len(points_clicked) < 2:
        cv2.imshow("Click Start and End Points", display)
        if cv2.waitKey(1) & 0xFF == 27:
            return
    cv2.destroyAllWindows()

    binary = preprocess_maze(image)
    skeleton = get_skeleton(binary)

    # ensure clicked points are on skeleton
    def nearest_skeleton_point(point, skeleton):
        # create mask of all white pixels
        skeleton_points = np.column_stack(np.where(skeleton == 255))

        if len(skeleton_points) == 0:
            print("Skeleton is empty.")
            return point

        distances = np.linalg.norm(skeleton_points - np.array(point)[::-1], axis=1)
        nearest_index = np.argmin(distances)
        nearest_yx = skeleton_points[nearest_index]
        return (nearest_yx[1], nearest_yx[0])  # (x, y)



    sk_start = nearest_skeleton_point(start_point, skeleton)
    sk_end = nearest_skeleton_point(end_point, skeleton)


    path = bfs_on_skeleton(skeleton, sk_start, sk_end)
    if not path:
        print("No path found on skeleton.")
        return

    draw_centered_path(image, path)
    cv2.imshow("Binary Maze", binary)
    cv2.imshow("Skeleton", skeleton)
    cv2.imshow("Solved Maze", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()