# Importing Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read images
image = cv2.imread("wall3.jpg")
poster_image = cv2.imread("poster.jpg")


# Detecting and reading ArUco markers
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
parameters = cv2.aruco.DetectorParameters()
corners, ids, rejectedpoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)
aruco_image= cv2.aruco.drawDetectedMarkers(image, corners ,ids)
if ids is not None:
    print(f"Detected ArUco marker IDs: {ids.flatten()}")  # Print detected IDs
    # Draw detected markers on the image
    cv2.aruco.drawDetectedMarkers(image, corners, ids)



# Reshaping the corner points to get the correct format
corner_points = np.reshape(corners[0], (-1, 2))  
corner_points = corner_points.astype(int)
marker_center_x = np.mean(corner_points[:, 0])  # Average of x-coordinates
marker_center_y = np.mean(corner_points[:, 1])  # Average of y-coordinates
marker_center = (marker_center_x, marker_center_y) # Marker center



# Define the poster corners
height, width, channels = poster_image.shape
poster_corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
#center of Poster
poster_center_x, poster_center_y = center = np.mean(poster_corners, axis=0)
print(poster_center_x, poster_center_y)


# Scaling factor for poster size relative to marker
scaling_factor = 0.15 # Scale the poster to fit the marker
scaled_width = int(width * scaling_factor)
scaled_height = int(height * scaling_factor)

# Define the new corners of the scaled poster so its center aligns with the marker center
top_left_scaled = (poster_center_x - scaled_width // 2, poster_center_y - scaled_height // 2)
top_right_scaled = (poster_center_x + scaled_width // 2,poster_center_y - scaled_height // 2)
bottom_left_scaled = (poster_center_x - scaled_width // 2, poster_center_y + scaled_height // 2)
bottom_right_scaled = (poster_center_x + scaled_width // 2, poster_center_y + scaled_height // 2)

# Define the perspective transformation points for the scaled poster
poster_perspective_shifted_pts = np.float32([top_left_scaled, top_right_scaled, bottom_right_scaled, bottom_left_scaled])

# Correct definition of marker_corners using the 4 corner points
marker_corners = np.float32([
    [corner_points[0][0], corner_points[0][1]],  # Top-left corner
    [corner_points[1][0], corner_points[1][1]],  # Top-right corner
    [corner_points[2][0], corner_points[2][1]],  # Bottom-right corner
    [corner_points[3][0], corner_points[3][1]]])  # Bottom-left corner




# Perspective transformation matrix
matrix = cv2.getPerspectiveTransform(poster_perspective_shifted_pts, marker_corners)
print("Perspective Transform Matrix:\n", matrix)

# Apply the perspective transformation to the display image (poster)
warped_poster = cv2.warpPerspective(poster_image, matrix, (image.shape[1], image.shape[0])) 


#MASKING#
gray = cv2.cvtColor(warped_poster, cv2.COLOR_BGR2GRAY)
non_black_mask = gray > 0
mask = np.zeros_like(warped_poster)
mask[non_black_mask] = [255, 255, 255]
inv_mask=cv2.bitwise_not(mask)
wall_with_no_poster=cv2.bitwise_and(image,inv_mask)
final_image=cv2.bitwise_or(wall_with_no_poster,warped_poster)


final_aruco_image= cv2.imwrite("aruco_image_id.jpg", aruco_image)
#Saving the final image
save_path = "\rE:\Mechatronics\CV Task 1\Result images"
cv2.imwrite("output_image_wall1.jpg",final_image)
print("Augmented image saved as 'output_image_wall1.jpg'")
cv2.imwrite("")
cv2.waitKey()
cv2.destroyAllWindows()



