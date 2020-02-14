import numpy as np
import cv2
from shapely.geometry import LineString
import math 
import string

def drawBorderLine(a, b):
    length = 40
    vX0 = b[0] - a[0]; vY0 = b[1] - a[1]
    mag = math.sqrt(vX0*vX0 + vY0*vY0)
    vX = vX0 / mag; vY = vY0 / mag
    temp = vX; vX = -vY; vY = temp
    z0 = (int(a[0]+vX0/2), int(a[1]+vY0/2))
    z1 = (int(a[0]+vX0/2 - vX * length), int(a[1] +vY0/2- vY * length))
    cv2.line(frame, a, b, (255, 255, 0), 2)
    cv2.arrowedLine(frame, z0, z1, (0, 255, 0), 2)
    cv2.putText(frame, "Out", z1, 0, 1, (0, 255, 0), 1)

# arr_symb = string.ascii_letters.split(",")
# smb = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
# sss = ''.join(np.random.choice(smb, 5))
# print(sss)
# arr_symb = ''.join([ for n in range(5)]) 
# arr_symb = ''.join([chr(i) for i in range(32,127)])
# print(arr_symb)
# arr = np.random.choice(arr_symb,4)
# strr = ''.join()  
# print("strr=",strr)

p0=[1,1]
angle_base = 0
frame = np.zeros((300, 400, 3), np.uint8)

border_line = [(200, 100), (200, 300)]
drawBorderLine(border_line[0], border_line[1])
# track_line = [(190, 280), (210, 10)]
track_line = [(190, 10), (210, 310)]
# track_line = [(210, 10), (190, 280)]
# track_line = [(210, 280), (190, 10)]
cv2.arrowedLine(frame, track_line[0], track_line[1], (0, 0, 255), 2)
border_line_str = LineString([border_line[0], border_line[1]]) 
track_line_str = LineString([track_line[0], track_line[1]])
if(track_line_str.intersection(border_line_str)):
    # detect a path direction compare to a border line
    # position = sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))
    X = track_line[1][0]
    Y = track_line[1][1]
    border_line_a = (border_line[1][0] - border_line[0][0])
    border_line_b = (border_line[1][1] - border_line[0][1])
    out_loc =  (border_line_a * (Y - border_line[0][1]) -  border_line_b * (X - border_line[0][0])) < 0
    # v1_u = np.array(track_line)
    # v2_u = np.array(border_line)
    # v1_u = v1_u[1]-v1_u[0]
    # v2_u = v2_u[1]-v2_u[0]
    # det = v1_u[0]*v2_u[1] - v2_u[0]*v1_u[1]
    # angle = np.arctan2(det, np.dot(v1_u, v2_u))
    # print (position, np.degrees(angle))
    print (out_loc,X,Y)
    
cv2.imshow("preview", frame)
key = cv2.waitKey(0)
cv2.destroyAllWindows()
