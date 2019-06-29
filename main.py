from cv2 import cv2
import numpy as np
import random

display_size = (640,480)
number_of_points = 200 # Number of random points to generate

# Creating numpy array of random points
points = np.zeros((number_of_points,2), dtype=np.float32)
for i in range(0,number_of_points):
    point_x = random.gauss((i+1)*3.5, 12)
    point_y = random.gauss((i+1)*2.5, 12)
    points[i,:] = [point_x, point_y]

# Initializing display with ones
display = np.ones((display_size[1],display_size[0],3), dtype=np.uint8)

# Random line params
m = random.random()
b = random.random()

# Initializing grad values from zero
m_grad = 0
b_grad = 0

# Learning rate and iterations
learning_rate = 0.000001
iterations = 1000

# Iterations begin
for i in range(iterations):
    # Calculating new values of m and b from gradients
    m = m - learning_rate * m_grad
    b = b + learning_rate * b_grad

    # Making canvas white
    display[:,:,:] = 255

    # Drawing random points
    for point in points:
        cv2.circle(display, (int(point[0]),int(point[1])), 3, (255, 0, 0), -1)
    
    # Drawing line
    p1 = (0, int(b))
    p2 = (display_size[0], int(m*display_size[0]+b))
    cv2.line(display, p1, p2, (0,0,255), 2)

    # Displaying
    cv2.imshow('Display', cv2.flip(display, 0))
    cv2.waitKey(20)

    # Finding predictions
    y_pred = m * points[:,0] + b

    # Finding difference
    diff = points[:,1] - y_pred

    # Using Mean square error loss ==> (sum((y[i]-ypred[i])**2))/n

    # Calculating gradient w.r.t m
    # Dm = (-2/n)*sum(x[i]*(y[i]-y_pred[i]))
    m_grad = (-2/points.shape[0]) * np.sum(diff * points[:,0])

    # Calculating gradient w.r.t b
    # Db = (-2/n)*sum(y[i]-y_pred[i])
    b_grad = (-2/points.shape[0]) * np.sum(diff)