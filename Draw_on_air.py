import cv2
import numpy as np


def nothing(x):
    pass


def color_cube(col):
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 0:] = [col[0], col[1], col[2]]
    return img



cap = cv2.VideoCapture(0)
accumulator_frame = None
draw = True

# cv2.namedWindow("panel")
#
# cv2.createTrackbar("r", "panel", 0, 255, nothing)
# cv2.createTrackbar("g", "panel", 0, 255, nothing)
# cv2.createTrackbar("b", "panel", 0, 255, nothing)
# cv2.createTrackbar("R", "panel", 210, 255, nothing)
# cv2.createTrackbar("G", "panel", 64, 255, nothing)
# cv2.createTrackbar("B", "panel", 42, 255, nothing)

yellow = (0, 255, 255)
red = (0, 0, 255)
blue = (255, 0, 0)
green = (0, 255, 0)
black = (0, 0, 0)

color = yellow
cube = color_cube(yellow)


canvas = np.ones((640, 976, 3), dtype=np.uint8) * 255

while True:

    ret, frame = cap.read()

    shape = frame.shape  # (480, 640, 3)
    flip_frame = cv2.flip(frame, 1)
    # grey = cv2.cvtColor(flip_frame, cv2.COLOR_BGR2GRAY)
    # print(grey.shape)
    new_mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    
    # h = int(cv2.getTrackbarPos("r", "panel"))
    # H = int(cv2.getTrackbarPos("R", "panel"))
    # s = int(cv2.getTrackbarPos("g", "panel"))
    # S = int(cv2.getTrackbarPos("G", "panel"))
    # v = int(cv2.getTrackbarPos("b", "panel"))
    # V = int(cv2.getTrackbarPos("B", "panel"))

    mask = cv2.inRange(flip_frame, (0, 20, 126), (255, 76, 255))  # (h, s, v), (H, S, V))
    masked_input = cv2.bitwise_and(flip_frame, flip_frame, mask=mask)
    if accumulator_frame is None:
        accumulator_frame = np.zeros(shape, dtype=np.uint8)

    p = cv2.SimpleBlobDetector_Params()
    p.filterByColor = False
    p.filterByConvexity = False
    p.filterByArea = True
    p.minArea = 100
    detector = cv2.SimpleBlobDetector_create(p)
    key_points = detector.detect(mask)

    if key_points:
        pt = key_points[0].pt
        if pt[0] < 10 and pt[1] < 40:
            accumulator_frame[:, :, :] = 0
        if 620 <= pt[0] <= 640 and 80 <= pt[1] <= 102:
            color = yellow
            print("yellow")
            print(pt)

        if 620 <= pt[0] <= 640 and 158 <= pt[1] <= 178:
            color = red
            print("red")
            print(pt)

        if 620 <= pt[0] <= 640 and 230 <= pt[1] <= 250:
            color = blue
            print("blue")
            print(pt)

        if 620 <= pt[0] <= 640 and 310 <= pt[1] <= 330:
            color = green
            print(pt)

        if 620 <= pt[0] <= 640 and 370 <= pt[1] <= 410:
            color = black
            print(pt)

        # elif
        cv2.circle(new_mask, (round(pt[0]), round(pt[1])),  10, (255, 255, 255), -1)
        if draw:
            cv2.circle(accumulator_frame, (round(pt[0]), round(pt[1])),  10, color, -1)

    white_screen = np.ones(shape, dtype=np.uint8) * 255

    new_image = cv2.bitwise_and(white_screen, white_screen, mask=new_mask)
    ew_image = cv2.bitwise_and(accumulator_frame, accumulator_frame, mask=cv2.bitwise_not(new_mask))

    canvas[:640, :720, :] = cv2.resize(new_image + ew_image, (720, 640))
    # cv2.putText(canvas, '@', (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
    #             1, (120, 120, 120), 1, cv2.LINE_AA)
    # canvas[:32,:32,:] = cv2.resize(cv2.imread("reload-icon.png"), (32,32))
    canvas[100:132, 710:742, :] = color_cube(yellow)
    canvas[:32,:32,:] = cv2.resize(cv2.imread("reload-icon.jpg"), (32, 32))
    canvas[200:232, 710:742, :] = color_cube(red)
    canvas[300:332, 710:742, :] = color_cube(blue)
    canvas[400:432, 710:742, :] = color_cube(green)
    canvas[500:532, 710:742, :] = color_cube((50, 50, 50))
    canvas[-257:-1, -257:-1, :] = cv2.resize(flip_frame, (256, 256))
    cv2.putText(canvas, 'Press \' q \' to QUIT ', (780, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Press \' s \' to SAVE ', (780, 200), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.imshow("canvas", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        print("saved")
        cv2.imwrite("My_drawing.jpg", accumulator_frame)

cap.release()
cv2.destroyAllWindows()

# green = (49, 111, 32), (108, 255, 108)
# def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
#     im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
#     return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)
#
# im_tile_resize = concat_tile_resize([[im1],
#                                      [im1, im2, im1, im2, im1],
#                                      [im1, im2, im1]])
# cv2.imwrite('data/dst/opencv_concat_tile_resize.jpg', im_tile_resize)

# img = cv2.imread("photo.jpg")
# print(img[:100,:100].shape)
# img2 = cv2.resize(img, (100, 100))
# print(img2.shape)
# img[:100, :100, :] = img2[:100, :100, :]
# cv2.imshow("img", img)
# cv2.waitKey(0)
# (623.8963012695312, 87.14814758300781)
# (625.0070190429688, 86.06302642822266)
# (631.0201416015625, 86.24712371826172)
# 640, 480 =


# h = int(cv2.getTrackbarPos("r", "panel"))
# H = int(cv2.getTrackbarPos("R", "panel"))
# s = int(cv2.getTrackbarPos("g", "panel"))
# S = int(cv2.getTrackbarPos("G", "panel"))
# v = int(cv2.getTrackbarPos("b", "panel"))
# V = int(cv2.getTrackbarPos("B", "panel"))
# lower_green = np.array([h, s, v])
# higher_green = np.array([H, S, V])
# print(h,s,v,H,S,V)

# print(cube.shape)
# cv2.namedWindow("panel")
#
# cv2.createTrackbar("r", "panel", 0, 255, nothing)
# cv2.createTrackbar("g", "panel", 0, 255, nothing)
# cv2.createTrackbar("b", "panel", 0, 255, nothing)
# cv2.createTrackbar("R", "panel", 210, 255, nothing)
# cv2.createTrackbar("G", "panel", 64, 255, nothing)
# cv2.createTrackbar("B", "panel", 42, 255, nothing)
