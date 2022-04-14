import os
import numpy as np
import cv2
from skimage.transform import hough_line, hough_line_peaks, rotate, resize
from tensorflow.keras import models
from copy import deepcopy
from skimage import morphology
import warnings


def crop_digit(d):
    crop_value = 10
    return d[:, crop_value:-crop_value]


def center_digit(d):
    """Cuts out the largest contour of the image (presumably it's a digit) and centers it.

    Args:
        d (np.array): Image of a digit.

    Returns:
        np.array: Array of the size (100, 100) of the centered black digit with a white background.
    """
    digit = cv2.normalize(d, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    contours = cv2.findContours(~digit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    contour = contours[max_index]
    x, y, w, h = cv2.boundingRect(contour)
    result = np.ones((100, 100), dtype=d.dtype)
    result[50-int(np.floor(h/2)):50+int(np.ceil(h/2)), 50-int(np.floor(w/2)):50+int(np.ceil(w/2))] = d[y:y + h, x:x + w]
    return result


# patch - list of 9 numbers: row, column or prepared 3x3 subsquare of the board
def validate_sudoku_patch(patch):
    desired_count = 9 - patch.count(0)
    setted_row = set(patch)
    if 0 in setted_row:
        setted_row.remove(0)
    actual_count = len(setted_row)
    if desired_count != actual_count:
        return False
    return True


# validates board - list of 9 lists of 9 numbers
def validate_correctness_of_sudoku_board(sudoku_board: list):
    for row in sudoku_board:
        if not validate_sudoku_patch(row):
            return False
    for column in zip(*sudoku_board):
        if not validate_sudoku_patch(column):
            return False
    dummy_board = []
    for i in [0, 3, 6]:
        for j in [0, 3, 6]:
            dummy_line = []
            for k in range(3):
                for l in range(3):
                    dummy_line.append(sudoku_board[i + k][j + l])
            dummy_board.append(dummy_line)

    for sudoku_patch in dummy_board:
        if not validate_sudoku_patch(sudoku_patch):
            return False
    return True


def is_digit(d):
    """Determines whether the image is a blank space or a digit.
    Args:
        d (np.array): Image of a digit or a blank space.

    Returns:
        int: 0 if the image is a blank space, 1 otherwise.
    """
    test_digit = morphology.remove_small_holes(d, area_threshold=200, connectivity=5)
    if test_digit[20:-20, 20:-20].mean() > 0.95:
        return 0
    else:
        return 1


def read_digit(d):
    test_digit = morphology.remove_small_holes(d, area_threshold=200, connectivity=5)
    test_digit = test_digit.astype(np.float32)
    test_digit = center_digit(test_digit)
    test_digit = resize(test_digit, (28, 28, 1))
    result = model.predict(np.expand_dims(test_digit, 0))
    return np.argmax(result) + 1


def solve(bo):
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1, 10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i

            if solve(bo):
                return True

            bo[row][col] = 0

    return False


def valid(bo, num, pos):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if bo[i][j] == num and (i, j) != pos:
                return False

    return True


def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)  # row, col

    return None


warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
model = models.load_model('models/model2.h5')
cap = cv2.VideoCapture(0)
prev_digit_position = []
board = []
board_changed = True
is_sudoku_solvable = True

while True:
    ret, frame = cap.read()
    image = frame
    original_image = image.copy()

    image = cv2.medianBlur(image, 3)
    image = cv2.Canny(image, 50, 50)

    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    contour = contours[max_index]
    sudoku_with_contour = original_image.copy()
    cv2.drawContours(sudoku_with_contour, contour, -1, (0, 255, 0), 2)
    if cv2.contourArea(contour) < 30000:
        cv2.imshow('Sudoku solver', frame)
    else:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)

        if len(approx) != 4:
            cv2.imshow('Sudoku solver', frame)
        else:
            width = height = 100 * 9
            x = [[approx[1][0][0], approx[1][0][1]], [approx[0][0][0], approx[0][0][1]],
                 [approx[2][0][0], approx[2][0][1]], [approx[3][0][0], approx[3][0][1]]]
            x = np.float32(x)
            y = np.float32([[0, height], [0, 0], [width, height], [width, 0]])
            matrix = cv2.getPerspectiveTransform(x, y)
            perspective = cv2.warpPerspective(original_image, matrix, (width, height))

            sudoku_rotated = False
            angle = np.arctan((approx[0][0][1] - approx[1][0][1]) / (approx[0][0][0] - approx[1][0][0])) / np.pi * 180
            if angle > -45:
                sudoku_rotated = True
                perspective = cv2.rotate(perspective, cv2.ROTATE_90_CLOCKWISE)

            sudoku = cv2.cvtColor(perspective, cv2.COLOR_RGB2GRAY)
            sudoku = cv2.medianBlur(sudoku, 3)
            sudoku = cv2.adaptiveThreshold(sudoku, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 4)

            digit_positions = [[0 for _ in range(9)] for _ in range(9)]
            for i in range(9):
                for j in range(9):
                    digit_positions[i][j] = is_digit(crop_digit(sudoku[i*100:(i+1)*100, j*100:(j+1)*100]))

            if digit_positions != prev_digit_position or not is_sudoku_solvable:
                board_changed = True
                board = deepcopy(digit_positions)
                for i in range(9):
                    for j in range(9):
                        if board[i][j] == 1:
                            board[i][j] = read_digit(crop_digit(sudoku[i*100:(i+1)*100, j*100:(j+1)*100]))
            else:
                board_changed = False

            prev_digit_position = deepcopy(digit_positions)

            original_board = deepcopy(board)
            if validate_correctness_of_sudoku_board(board) or not board_changed:
                if board_changed:
                    is_sudoku_solvable = True
                    solve(board)
                x_offset = 20
                y_offset = 80
                solved_sudoku = np.zeros(perspective.shape, dtype=np.uint8)
                for i in range(9):
                    for j in range(9):
                        if digit_positions[i][j] == 0:
                            solved_sudoku = cv2.putText(solved_sudoku, str(board[i][j]),
                                                        (100 * j + x_offset, 100 * i + y_offset),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 3,
                                                        (255, 255, 255), 7, cv2.LINE_AA)

                if sudoku_rotated:
                    solved_sudoku = cv2.rotate(solved_sudoku, cv2.ROTATE_90_COUNTERCLOCKWISE)

                pts = x
                img_dest = sudoku_with_contour.copy()
                img_src = solved_sudoku.copy()

                pts_source = y
                h, status = cv2.findHomography(pts_source, pts)
                warped = cv2.warpPerspective(img_src, h, (img_dest.shape[1], img_dest.shape[0]))

                dst_img = cv2.subtract(img_dest, warped)
                cv2.imshow('Sudoku solver', dst_img)
            else:
                is_sudoku_solvable = False
                cv2.imshow('Sudoku solver', sudoku_with_contour)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
