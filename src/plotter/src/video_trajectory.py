


import cv2
import numpy as np
import matplotlib.pyplot as plt

from numpy import ones,vstack
from numpy.linalg import lstsq



path_video = '/media/auto/ffb3ec49-7cca-4ff7-ab97-3d15f32926cb/course/thesis/videos/estimator/lidar_estimation_vel0.8/'
video_file = 'VID_20210609_190615.mp4'

write_video = True

# # Create a VideoCapture object and read from input file
# # If the input is the camera, pass 0 instead of the video file name
# cap = cv2.VideoCapture(path + video_file)

# # Check if camera opened successfully
# if (cap.isOpened()== False): 
#     print("Error opening video stream or file")

# # Read until video is completed
# while(cap.isOpened()):
#       # Capture frame-by-frame
#     ret, frame = cap.read()
# #     frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25) 
#     if ret == True:

#         # Display the resulting frame
# #         cv2.namedWindow('image',WINDOW_NORMAL)
# #         cv2.resizeWindow('image', 600,600)
#         cv2.imshow('Frame',frame)

#         # Press Q on keyboard to  exit
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break

#     # Break the loop
#     else: 
#         break

# # When everything done, release the video capture object
# cap.release()

# # Closes all the frames
# cv2.destroyAllWindows()

path = '/home/auto/Desktop/autonomus_vehicle_project/project/development/proto/plotter/estimator/lidar_estimation_vel0.8'
file_name = '/frame.jpg'
frame = cv2.imread(path+file_name)


# plt.figure(figsize = (20,20))
# plt.imshow(frame)
# plt.show()

# cv2.namedWindow("frame", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
# frame_im = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
# cv2.imshow('frame', frame_im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ##raw points
# pt1 =  [641.275, 210.98]
# pt2 =  [1013.42, 96.09] 
# pt3 =  [1795.31, 557.315]
# pt4 =  [1347.95, 852.257]

pt1 =  [641.275, 210.98]
pt2 =  [1013.42, 96.09] 
pt3 =  [1795.31, 557.315]
pt4 =  [1347.95, 852.257]


# ##raw points
# pt1 =  [1111, 440]
# pt2 =  [1828, 213] 
# pt3 =  [3240, 1235]
# pt4 =  [2528, 1724]

pts = np.array([pt1, pt2, pt3, pt4])

def find_line(pt1, pt2):
    points = [pt1,pt2]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    print("Line Solution is y = {m}x + {c}".format(m=m,c=c))
    return m, c


border_size = int(frame.shape[0]*0.3)

img = cv2.copyMakeBorder( frame, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT)
# print img.shape

# y, x = img.shape[:2] 


#################### CROP LINE ####################
offset_top = 300.0 #increasing will shift downwards

offset_bottom = 580.0  #increasing will shift downwards

offset_left = 950.0 #increasing will shift left

offset_right = -300.0 #increasing will shift left


m1, c1 = find_line(pt1, pt2)
#want to get c1 bias in a way that putting x = width of image give y >= 0.0 
c1 = int(c1 + offset_top)

tpt1_l1 = (0,int(c1))
tpt2_l1 = (img.shape[1],int(m1*img.shape[1] + c1))


m2, c2 = find_line(pt2, pt3)
#want to get c2 bias in a way that putting x = width of image give y >= 0.0 
# c2 = -m2*img.shape[1] + offset_bottom
c2 = int(c2 + offset_right)

tpt1_l2 = (-int(c2/m2),0)
tpt2_l2 = (img.shape[1],int(m2*img.shape[1] + c2))


m3, c3 = find_line(pt4, pt3)
#want to get c2 bias in a way that putting x = width of image give y >= 0.0 
# c2 = -m2*img.shape[1] + offset_bottom
c3 = int(c3 + offset_bottom)

tpt1_l3 = (int((img.shape[0] - c3)/m3),img.shape[0])
tpt2_l3 = (img.shape[1],int(m3*img.shape[1] + c3))


m4, c4 = find_line(pt1, pt4)
#want to get c2 bias in a way that putting x = width of image give y >= 0.0 
# c2 = -m2*img.shape[1] + offset_bottom
c4 = int(c4 + offset_left)

tpt1_l4 = (0,c4)
tpt2_l4 = (int((img.shape[0] - c4)/m4),img.shape[0])


line1 = [m1, c1]
line2 = [m2, c2]
line3 = [m3, c3]
line4 = [m4, c4]

x1 = (c1-c2) / (m2-m1)
y1 = m1 * x1 + c1

x2 = (c2-c3) / (m3-m2)
y2 = m2 * x2 + c2

x3 = (c3-c4) / (m4-m3)
y3 = m3 * x3 + c3

x4 = (c4-c1) / (m1-m4)
y4 = m4 * x4 + c4

seed_pts = np.array([[x4, y4],[x1, y1], [x2, y2], [x3, y3]])
print 'crop points = ', seed_pts

# ## Extracting points for perspective transform

# pts = np.array([tpt1, tpt2, tpt3, tpt4])
pts = np.array([[int(pt1[0]), int(pt1[1])], [int(pt2[0]), int(pt2[1])],\
                [int(pt3[0]), int(pt3[1])], [int(pt4[0]), int(pt3[1]) ]])  + border_size


# cv2.line(img,tpt1_l1,tpt2_l1,(255,0,0),15) #top, r
# cv2.line(img,tpt1_l2,tpt2_l2,(0,255,0),15) #right, g
# cv2.line(img,tpt1_l3,tpt2_l3,(0,0,255),15) #bottom, b
# cv2.line(img,tpt1_l4,tpt2_l4,(255,0,255),15) #left, m
# cv2.polylines(img,[pts],True,(0,255,255))
# plt.figure()
# plt.scatter(seed_pts[:,0], seed_pts[:,1])
# plt.imshow(img)
# plt.show()




cap = cv2.VideoCapture(path_video + video_file)
 
frame_width = img.shape[1]
frame_height = img.shape[0]

if write_video == True:
    out = cv2.VideoWriter(path_video + '/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

counter = 0

while True:
      
    ret, frame = cap.read()
    img = cv2.copyMakeBorder( frame, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT)

    pts1 = np.float32([[x4, y4],[x1, y1], [x2, y2], [x3, y3]])
    pts2 = np.float32([[0, 0], [img.shape[1], 0],[img.shape[1], img.shape[0]], [0, img.shape[0]]])
      
#     # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
    if write_video == True:

        out.write(result)
  
    # cv2.namedWindow("frame1", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
    # imS = cv2.resize(result, (0,0), fx=0.8, fy=0.8)                   # Resize image


    # cv2.imshow('frame', frame) # Inital Captureqqqq
    # cv2.imshow('frame1', imS) # Transformed Capture
  
    if cv2.waitKey(24) == 27:
        break
    
    print "counter", counter
    counter += 1
cap.release()

if write_video == True:
    out.release()

cv2.destroyAllWindows()


# # In[20]:


# cv2.destroyAllWindows()


# # In[20]:


# [x1, y1], [x2, y2], [x3, y3], [x4, y4]


# # In[104]:


# cv2.destroyAllWindows()


# # In[87]:


# plt.figure()
# plt.imshow(result)
# plt.show()


# # In[83]:


# map = Map() 


# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure

# fig = Figure()
# canvas = FigureCanvas(fig)
# ax = fig.gca()

# # ax.text(0.0,0.0,"Test", fontsize=45)
# # ax.axis('off')

#    # draw the canvas, cache the renderer


# Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
# # Points1 = np.zeros((Points, 2))
# # Points2 = np.zeros((Points, 2))
# # Points0 = np.zeros((Points, 2))
# Points1 = np.zeros((Points, 3))
# Points2 = np.zeros((Points, 3))
# Points0 = np.zeros((Points, 3))

# for i in range(0, int(Points)):
#     Points1[i, :] = map.getGlobalPosition(i * 0.1, map.halfWidth)
#     Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.halfWidth)
#     Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

# plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o') #points on center track
# ax.plot(Points1[:, 0], Points1[:, 1], color = 'black' , linewidth  = 3) # inner track
# ax.plot(Points2[:, 0], Points2[:, 1], color = 'black' , linewidth  = 3) #outer track
# ax.plot(Points0[:, 0], Points0[:, 1], '--' ,color = 'darkgray' , linewidth  = 2) #center track

# # plt.xlim(-1.6,3.0)
# # plt.ylim(-0.5,2.8)




# # ax.xlabel(r"X Position")
# # ax.ylabel(r"Y Position")
# # plt.imshow(result, alphaimage = np.fromstring(canvas.tostring_rgb(), dtype='uint8')=0.0)

# # plt.xlim(2.0,2.8)
# # plt.ylim(0.6,1.8)
# # ax.show()
# fig.canvas.draw()    
# image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

# # plt.savefig('./images/sampling_rate_comparison.png',dpi = 300, bbox_inches = 'tight')


# # In[84]:


# map.PointAndTangent[:, 0], map.PointAndTangent[:, 1]


# # In[82]:


# plt.figure()
# plt.imshow(image)
# plt.show()


# # In[79]:


# import matplotlib.pyplot
# import numpy
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
# # Generate a figure with matplotlib</font>
# figure = matplotlib.pyplot.figure(  )
# plot   = figure.add_subplot ( 111 )
 
# # draw a cardinal sine plot
# x = numpy.arange ( 0, 100, 0.1 )
# y = numpy.sin ( x ) / x
# plot.plot ( x, y )


# def fig2data ( fig ):
#     """
#     @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
#     @param fig a matplotlib figure
#     @return a numpy 3D array of RGBA values
#     """
#     # draw the renderer
#     fig.canvas.draw ( )
 
#     # Get the RGBA buffer from the figure
#     w,h = fig.canvas.get_width_height()
#     buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=numpy.uint8 )
#     buf.shape = ( w, h,4 )
 
#     # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
#     buf = np.roll ( buf, 3, axis = 2 )
#     return buf


# import Image
 
# def fig2img ( fig ):
#     """
#     @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
#     @param fig a matplotlib figure
#     @return a Python Imaging Library ( PIL ) image
#     """
#     # put the figure pixmap into a numpy array
#     buf = fig2data ( fig )
#     w, h, d = buf.shape
#     return Image.fromstring( "RGBA", ( w ,h ), buf.tostring( ) )

# data = fig2img(figure)


# # In[72]:


# plt.figure()
# plt.imshow(image)
# plt.show()


# # In[69]:


# image


# # In[562]:


# import cv2
# import numpy as np

# img = frame

# pts = np.array([[864, 651], [1016, 581], [1205, 667], [1058, 759]], dtype=np.float32)
# for pt in pts:
#     cv2.circle(img, tuple(pt.astype(np.int)), 1, (0,0,255), -1)

# # compute IPM matrix and apply it
# ipm_pts = np.array([[448,609], [580,609], [580,741], [448,741]], dtype=np.float32)
# ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)
# ipm = cv2.warpPerspective(img, ipm_matrix, img.shape[:2][::-1])

# # display (or save) images
# cv2.imshow('img', img)
# cv2.imshow('ipm', ipm)
# cv2.waitKey()


# # In[351]:


# pts1


# # In[352]:


# pts2


# # In[571]:


# cv2.destroyAllWindows()


# # In[ ]:


# y = -0.542056074766x + 646.85046729


# # In[337]:


# tpt3


# # In[338]:


# tpt4


# # ### Perspecrtive transfrom

# # In[14]:


# def order_points(pts):
# 	# initialzie a list of coordinates that will be ordered
# 	# such that the first entry in the list is the top-left,
# 	# the second entry is the top-right, the third is the
# 	# bottom-right, and the fourth is the bottom-left
# 	rect = np.zeros((4, 2), dtype = "float32")
# 	# the top-left point will have the smallest sum, whereas
# 	# the bottom-right point will have the largest sum
# 	s = pts.sum(axis = 1)
# 	rect[0] = pts[np.argmin(s)]
# 	rect[2] = pts[np.argmax(s)]
# 	# now, compute the difference between the points, the
# 	# top-right point will have the smallest difference,
# 	# whereas the bottom-left will have the largest difference
# 	diff = np.diff(pts, axis = 1)
# 	rect[1] = pts[np.argmin(diff)]
# 	rect[3] = pts[np.argmax(diff)]
# 	# return the ordered coordinates
# 	return rect

# def four_point_transform(image, pts):
# 	# obtain a consistent order of the points and unpack them
# 	# individually
# 	rect = order_points(pts)
# 	(tl, tr, br, bl) = rect
# 	# compute the width of the new image, which will be the
# 	# maximum distance between bottom-right and bottom-left
# 	# x-coordiates or the top-right and top-left x-coordinates
# 	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
# 	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
# 	maxWidth = max(int(widthA), int(widthB))
# 	# compute the height of the new image, which will be the
# 	# maximum distance between the top-right and bottom-right
# 	# y-coordinates or the top-left and bottom-left y-coordinates
# 	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
# 	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
# 	maxHeight = max(int(heightA), int(heightB))
# 	# now that we have the dimensions of the new image, construct
# 	# the set of destination points to obtain a "birds eye view",
# 	# (i.e. top-down view) of the image, again specifying points
# 	# in the top-left, top-right, bottom-right, and bottom-left
# 	# order
# 	dst = np.array([
# 		[0, 0],
# 		[maxWidth - 1, 0],
# 		[maxWidth - 1, maxHeight - 1],
# 		[0, maxHeight - 1]], dtype = "float32")
# 	# compute the perspective transform matrix and then apply it
# 	M = cv2.getPerspectiveTransform(rect, dst)
# 	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
# 	# return the warped image
# 	return warped


# # In[ ]:


# pt1 =  (364, 142)
# pt2 =  (557, 76)
# pt3 =  (793, 217)
# pt4 =  (579, 333)

# # pt1 =  (142, 364)
# # pt2 =  (76, 557)
# # pt3 =  (217, 793)
# # pt4 =  (333, 579)




# pts = np.array([pt1, pt2, pt3, pt4])
# wraped = four_point_transform(frame, pts)


# # In[21]:


# plt.figure()
# plt.imshow(wraped)
# plt.show()




# # In[22]:


# frame.shape


# # In[514]:


# import cv2 
# import numpy as np 
  
# # Turn on Laptop's webcam

# cap = cv2.VideoCapture(path + video_file)
 
  
# while True:
      
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)  
#     # Locate points of the documents or object which you want to transform
#     pts1 = np.float32([[364, 142],[557, 76],[793, 217],[579, 333]])
# #     pts1 = np.float32([tpt1,tpt2,tpt3,tpt4])
#     pts2 = np.float32([[0, 0], [960, 0],[960, 540], [0, 540]])
# #     pts2 = np.float32([[0, 0], [540, 0], [0, 960], [540, 960]])
      
#     # Apply Perspective Transform Algorithm
#     matrix = cv2.getPerspectiveTransform(pts1, pts2)
#     result = cv2.warpPerspective(frame, matrix, (960, 540))
# #     result = cv2.warpPerspective(frame, matrix, ( 540, 960,))
    
#     # Wrap the transformed image
  
#     cv2.imshow('frame', frame) # Inital Capture
#     cv2.imshow('frame1', result) # Transformed Capture
  
#     if cv2.waitKey(24) == 27:
#         break
  
# cap.release()
# cv2.destroyAllWindows()


# # In[515]:


# cv2.destroyAllWindows()


# # In[49]:


# frame.shape


# # In[48]:


# img = frame.copy()
# rows,cols,ch = img.shape
# pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
# pts2 = np.float32([[0,0],[frame.shape[1],0],[0,300],[300,300]])
# M = cv2.getPerspectiveTransform(pts1,pts2)
# dst = cv2.warpPerspective(img,M,(frame.shape[0],frame.shape[1]))
# plt.figure()
# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()


# # In[82]:


# img = cv2.imread('sudoku.jpg')
# rows,cols,ch = img.shape
# pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
# pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
# M = cv2.getPerspectiveTransform(pts1,pts2)
# dst = cv2.warpPerspective(img,M,(300,300))
# plt.figure()
# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()


# # In[34]:


# class Map():
#     """map object
#     Attributes:
#         getGlobalPosition: convert position from (s, ey) to (X,Y)
#     """
#     def __init__(self, flagTrackShape = 0):

#         """ Nos interesa que el planner tenga una pista algo mas reducida de la real
#         para conservar algo de robustez y no salirnos de la pista en el primer segundo. """

#         # self.halfWidth  = rospy.get_param("halfWidth")
#         self.halfWidth  = 0.25

#         self.slack      = 0.15
#         # selectedTrack   = rospy.get_param("trackShape")
#         selectedTrack   = "oval_iri"

#         if selectedTrack == "3110":
#             spec = np.array([[60 * 0.03, 0],
#                              [80 * 0.03, +80 * 0.03 * 2 / np.pi],
#                              [20 * 0.03, 0],
#                              [80 * 0.03, +80 * 0.03 * 2 / np.pi],
#                              [40 * 0.03, -40 * 0.03 * 10 / np.pi],
#                              [60 * 0.03, +60 * 0.03 * 5 / np.pi],
#                              [40 * 0.03, -40 * 0.03 * 10 / np.pi],
#                              [80 * 0.03, +80 * 0.03 * 2 / np.pi],
#                              [20 * 0.03, 0],
#                              [80 * 0.03, +80 * 0.03 * 2 / np.pi],
#                              [80 * 0.03, 0]])

#         elif selectedTrack == "oval":
#             spec = np.array([[1.0, 0],
#                              [4.5, 4.5 / np.pi],
#                              [2.0, 0],
#                              [4.5, 4.5 / np.pi],
#                              [1.0, 0]])

#         elif selectedTrack == "L_shape":
#             lengthCurve     = 4.5
#             spec = np.array([[1.0, 0],
#                              [lengthCurve, lengthCurve / np.pi],
#                              [lengthCurve/2,-lengthCurve / np.pi ],
#                              [lengthCurve, lengthCurve / np.pi],
#                              [lengthCurve / np.pi *2, 0],
#                              [lengthCurve/2, lengthCurve / np.pi]])


#         elif selectedTrack == "oval_iri_old":
#             spec = 1.0*np.array([[1.25, 0],
#                              [3.5, 3.5 / np.pi],
#                              [1.25, 0],
#                              [3.5, 3.5 / np.pi]])

#         elif selectedTrack == "oval_iri":
#             spec = 1.0*np.array([[1.34, 0],
#                              [1.125*np.pi, 1.125],
#                              [1.34, 0],
#                              [1.14*np.pi, 1.125]])


#         # Now given the above segments we compute the (x, y) points of the track and the angle of the tangent vector (psi) at
#         # these points. For each segment we compute the (x, y, psi) coordinate at the last point of the segment. Furthermore,
#         # we compute also the cumulative s at the starting point of the segment at signed curvature
#         # PointAndTangent = [x, y, psi, cumulative s, segment length, signed curvature]

#         PointAndTangent = np.zeros((spec.shape[0] + 1, 6))
#         for i in range(0, spec.shape[0]):
#             if spec[i, 1] == 0.0:              # If the current segment is a straight line
#                 l = spec[i, 0]                 # Length of the segments
#                 if i == 0:
#                     ang = 0                          # Angle of the tangent vector at the starting point of the segment
#                     x = 0 + l * np.cos(ang)          # x coordinate of the last point of the segment
#                     y = 0 + l * np.sin(ang)          # y coordinate of the last point of the segment
#                 else:
#                     ang = PointAndTangent[i - 1, 2]                 # Angle of the tangent vector at the starting point of the segment
#                     x = PointAndTangent[i-1, 0] + l * np.cos(ang)  # x coordinate of the last point of the segment
#                     y = PointAndTangent[i-1, 1] + l * np.sin(ang)  # y coordinate of the last point of the segment
#                 psi = ang  # Angle of the tangent vector at the last point of the segment

#                 # # With the above information create the new line
#                 # if i == 0:
#                 #     NewLine = np.array([x, y, psi, PointAndTangent[i, 3], l, 0])
#                 # else:
#                 #     NewLine = np.array([x, y, psi, PointAndTangent[i, 3] + PointAndTangent[i, 4], l, 0])
#                 #
#                 # PointAndTangent[i + 1, :] = NewLine  # Write the new info

#                 if i == 0:
#                     NewLine = np.array([x, y, psi, PointAndTangent[i, 3], l, 0])
#                 else:
#                     NewLine = np.array([x, y, psi, PointAndTangent[i-1, 3] + PointAndTangent[i-1, 4], l, 0])

#                 PointAndTangent[i, :] = NewLine  # Write the new info
#             else:
#                 l = spec[i, 0]                 # Length of the segment
#                 r = spec[i, 1]                 # Radius of curvature


#                 if r >= 0:
#                     direction = 1
#                 else:
#                     direction = -1

#                 if i == 0:
#                     ang = 0                                                      # Angle of the tangent vector at the
#                                                                                  # starting point of the segment
#                     CenterX = 0                               + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
#                     CenterY = 0                               + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle
#                 else:
#                     ang = PointAndTangent[i - 1, 2]                              # Angle of the tangent vector at the
#                                                                                  # starting point of the segment
#                     CenterX = PointAndTangent[i-1, 0]                               + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
#                     CenterY = PointAndTangent[i-1, 1]                               + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle

#                 spanAng = l / np.abs(r)  # Angle spanned by the circle
#                 psi = wrap(ang + spanAng * np.sign(r))  # Angle of the tangent vector at the last point of the segment

#                 angleNormal = wrap((direction * np.pi / 2 + ang))
#                 angle = -(np.pi - np.abs(angleNormal)) * (sign(angleNormal))
#                 x = CenterX + np.abs(r) * np.cos(
#                     angle + direction * spanAng)  # x coordinate of the last point of the segment
#                 y = CenterY + np.abs(r) * np.sin(
#                     angle + direction * spanAng)  # y coordinate of the last point of the segment

#                 # With the above information create the new line
#                 # plt.plot(CenterX, CenterY, 'bo')
#                 # plt.plot(x, y, 'ro')

#                 # if i == 0:
#                 #     NewLine = np.array([x, y, psi, PointAndTangent[i, 3], l, 1 / r])
#                 # else:
#                 #     NewLine = np.array([x, y, psi, PointAndTangent[i, 3] + PointAndTangent[i, 4], l, 1 / r])
#                 #
#                 # PointAndTangent[i + 1, :] = NewLine  # Write the new info

#                 if i == 0:
#                     NewLine = np.array([x, y, psi, PointAndTangent[i, 3], l, 1 / r])
#                 else:
#                     NewLine = np.array([x, y, psi, PointAndTangent[i-1, 3] + PointAndTangent[i-1, 4], l, 1 / r])

#                 PointAndTangent[i, :] = NewLine  # Write the new info
#             # plt.plot(x, y, 'or')

#         # Now update info on last point
#         # xs = PointAndTangent[PointAndTangent.shape[0] - 2, 0]
#         # ys = PointAndTangent[PointAndTangent.shape[0] - 2, 1]
#         # xf = PointAndTangent[0, 0]
#         # yf = PointAndTangent[0, 1]
#         # psif = PointAndTangent[PointAndTangent.shape[0] - 2, 2]
#         #
#         # # plt.plot(xf, yf, 'or')
#         # # plt.show()
#         # l = np.sqrt((xf - xs) ** 2 + (yf - ys) ** 2)
#         #
#         # NewLine = np.array([xf, yf, psif, PointAndTangent[PointAndTangent.shape[0] - 2, 3] + PointAndTangent[
#         #     PointAndTangent.shape[0] - 2, 4], l, 0])
#         # PointAndTangent[-1, :] = NewLine


#         xs = PointAndTangent[-2, 0]
#         ys = PointAndTangent[-2, 1]
#         xf = 0
#         yf = 0
#         psif = 0

#         # plt.plot(xf, yf, 'or')
#         # plt.show()
#         l = np.sqrt((xf - xs) ** 2 + (yf - ys) ** 2)

#         NewLine = np.array([xf, yf, psif, PointAndTangent[-2, 3] + PointAndTangent[-2, 4], l, 0])
#         PointAndTangent[-1, :] = NewLine

#         self.PointAndTangent = PointAndTangent
#         self.TrackLength = PointAndTangent[-1, 3] + PointAndTangent[-1, 4]


#     def getGlobalPosition(self, s, ey):
#         """coordinate transformation from curvilinear reference frame (e, ey) to inertial reference frame (X, Y)
#         (s, ey): position in the curvilinear reference frame
#         """

#         # wrap s along the track
#         while (s > self.TrackLength):
#             s = s - self.TrackLength

#         # Compute the segment in which system is evolving
#         PointAndTangent = self.PointAndTangent

#         index = np.all([[s >= PointAndTangent[:, 3]], [s < PointAndTangent[:, 3] + PointAndTangent[:, 4]]], axis=0)
#         ##  i = int(np.where(np.squeeze(index))[0])
#         i = np.where(np.squeeze(index))[0]

#         if PointAndTangent[i, 5] == 0.0:  # If segment is a straight line
#             # Extract the first final and initial point of the segment
#             xf = PointAndTangent[i, 0]
#             yf = PointAndTangent[i, 1]
#             xs = PointAndTangent[i - 1, 0]
#             ys = PointAndTangent[i - 1, 1]
#             psi = PointAndTangent[i, 2]

#             # Compute the segment length
#             deltaL = PointAndTangent[i, 4]
#             reltaL = s - PointAndTangent[i, 3]

#             # Do the linear combination
#             x = (1 - reltaL / deltaL) * xs + reltaL / deltaL * xf + ey * np.cos(psi + np.pi / 2)
#             y = (1 - reltaL / deltaL) * ys + reltaL / deltaL * yf + ey * np.sin(psi + np.pi / 2)
#             theta = psi
#         else:
#             r = 1 / PointAndTangent[i, 5]  # Extract curvature
#             ang = PointAndTangent[i - 1, 2]  # Extract angle of the tangent at the initial point (i-1)
#             # Compute the center of the arc
#             if r >= 0:
#                 direction = 1
#             else:
#                 direction = -1

#             CenterX = PointAndTangent[i - 1, 0]                       + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
#             CenterY = PointAndTangent[i - 1, 1]                       + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle

#             spanAng = (s - PointAndTangent[i, 3]) / (np.pi * np.abs(r)) * np.pi

#             angleNormal = wrap((direction * np.pi / 2 + ang))
#             angle = -(np.pi - np.abs(angleNormal)) * (sign(angleNormal))

#             x = CenterX + (np.abs(r) - direction * ey) * np.cos(
#                 angle + direction * spanAng)  # x coordinate of the last point of the segment
#             y = CenterY + (np.abs(r) - direction * ey) * np.sin(
#                 angle + direction * spanAng)  # y coordinate of the last point of the segment
#             theta = ang + direction * spanAng

#         return np.squeeze(x), np.squeeze(y), np.squeeze(theta)



#     def getGlobalPosition_Racing(self, ex, ey, xd, yd, psid):
#         """coordinate transformation from curvilinear reference frame (ex, ey) to inertial reference frame (X, Y)
#         based on inverse of error computation for racing:
#             ex      = +(x-xd)*np.cos(psid) + (y-yd)*np.sin(psid)
#             ey      = -(x-xd)*np.sin(psid) + (y-yd)*np.cos(psid)
#             epsi    = wrap(psi-psid)
#         """

#         # x = ex*np.cos(psid) - ey*np.sin(psid) + xd
#         x = xd
#         y = (ey - xd*np.sin(psid) + yd*np.cos(psid) + x*np.sin(psid)) / np.cos(psid)

#         return x, y




#     def getLocalPosition(self, x, y, psi):
#         """coordinate transformation from inertial reference frame (X, Y) to curvilinear reference frame (s, ey)
#         (X, Y): position in the inertial reference frame
#         """
#         PointAndTangent = self.PointAndTangent
#         CompletedFlag = 0



#         for i in range(0, PointAndTangent.shape[0]):
#             if CompletedFlag == 1:
#                 break

#             if PointAndTangent[i, 5] == 0.0:  # If segment is a straight line
#                 # Extract the first final and initial point of the segment
#                 xf = PointAndTangent[i, 0]
#                 yf = PointAndTangent[i, 1]
#                 xs = PointAndTangent[i - 1, 0]
#                 ys = PointAndTangent[i - 1, 1]

#                 psi_unwrap = np.unwrap([PointAndTangent[i - 1, 2], psi])[1]
#                 epsi = psi_unwrap - PointAndTangent[i - 1, 2]

#                 # Check if on the segment using angles
#                 if (la.norm(np.array([xs, ys]) - np.array([x, y]))) == 0:
#                     s  = PointAndTangent[i, 3]
#                     ey = 0
#                     CompletedFlag = 1

#                 elif (la.norm(np.array([xf, yf]) - np.array([x, y]))) == 0:
#                     s = PointAndTangent[i, 3] + PointAndTangent[i, 4]
#                     ey = 0
#                     CompletedFlag = 1
#                 else:
#                     if np.abs(computeAngle( [x,y] , [xs, ys], [xf, yf])) <= np.pi/2 and np.abs(computeAngle( [x,y] , [xf, yf], [xs, ys])) <= np.pi/2:
#                         v1 = np.array([x,y]) - np.array([xs, ys])
#                         angle = computeAngle( [xf,yf] , [xs, ys], [x, y])
#                         s_local = la.norm(v1) * np.cos(angle)
#                         s       = s_local + PointAndTangent[i, 3]
#                         ey      = la.norm(v1) * np.sin(angle)

#                         if np.abs(ey)<= self.halfWidth + self.slack:
#                             CompletedFlag = 1

#             else:
#                 xf = PointAndTangent[i, 0]
#                 yf = PointAndTangent[i, 1]
#                 xs = PointAndTangent[i - 1, 0]
#                 ys = PointAndTangent[i - 1, 1]

#                 r = 1 / PointAndTangent[i, 5]  # Extract curvature
#                 if r >= 0:
#                     direction = 1
#                 else:
#                     direction = -1

#                 ang = PointAndTangent[i - 1, 2]  # Extract angle of the tangent at the initial point (i-1)

#                 # Compute the center of the arc
#                 CenterX = xs + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
#                 CenterY = ys + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle

#                 # Check if on the segment using angles
#                 if (la.norm(np.array([xs, ys]) - np.array([x, y]))) == 0:
#                     ey = 0
#                     psi_unwrap = np.unwrap([ang, psi])[1]
#                     epsi = psi_unwrap - ang
#                     s = PointAndTangent[i, 3]
#                     CompletedFlag = 1
#                 elif (la.norm(np.array([xf, yf]) - np.array([x, y]))) == 0:
#                     s = PointAndTangent[i, 3] + PointAndTangent[i, 4]
#                     ey = 0
#                     psi_unwrap = np.unwrap([PointAndTangent[i, 2], psi])[1]
#                     epsi = psi_unwrap - PointAndTangent[i, 2]
#                     CompletedFlag = 1
#                 else:
#                     arc1 = PointAndTangent[i, 4] * PointAndTangent[i, 5]
#                     arc2 = computeAngle([xs, ys], [CenterX, CenterY], [x, y])
#                     if np.sign(arc1) == np.sign(arc2) and np.abs(arc1) >= np.abs(arc2):
#                         v = np.array([x, y]) - np.array([CenterX, CenterY])
#                         s_local = np.abs(arc2)*np.abs(r)
#                         s    = s_local + PointAndTangent[i, 3]
#                         ey   = -np.sign(direction) * (la.norm(v) - np.abs(r))
#                         psi_unwrap = np.unwrap([ang + arc2, psi])[1]
#                         epsi = psi_unwrap - (ang + arc2)

#                         if np.abs(ey) <= self.halfWidth + self.slack: # OUT OF TRACK!!
#                             CompletedFlag = 1

#         # if epsi>1.0:
#         #     print "epsi Greater then 1.0"
#         #     pdb.set_trace()

#         if CompletedFlag == 0:
#             s    = 10000
#             ey   = 10000
#             epsi = 10000
#             #print "Error!! POINT OUT OF THE TRACK!!!! <=================="
#             # pdb.set_trace()

#         return s, ey, epsi, CompletedFlag


    
# def wrap(angle):
#     if angle < -np.pi:
#         w_angle = 2 * np.pi + angle
#     elif angle > np.pi:
#         w_angle = angle - 2 * np.pi
#     else:
#         w_angle = angle

#     return w_angle


# def sign(a):
#     if a >= 0:
#         res = 1
#     else:
#         res = -1

#     return res


# def unityTestChangeOfCoordinates(map, ClosedLoopData):
#     """For each point in ClosedLoopData change (X, Y) into (s, ey) and back to (X, Y) to check accurancy
#     """
#     TestResult = 1
#     for i in range(0, ClosedLoopData.x.shape[0]):
#         xdat = ClosedLoopData.x
#         xglobdat = ClosedLoopData.x_glob

#         s, ey, _, _ = map.getLocalPosition(xglobdat[i, 4], xglobdat[i, 5], xglobdat[i, 3])
#         v1 = np.array([s, ey])
#         v2 = np.array(xdat[i, 4:6])
#         v3 = np.array(map.getGlobalPosition(v1[0], v1[1]))
#         v4 = np.array([xglobdat[i, 4], xglobdat[i, 5]])
#         # print v1, v2, np.dot(v1 - v2, v1 - v2), np.dot(v3 - v4, v3 - v4)

#         if np.dot(v3 - v4, v3 - v4) > 0.00000001:
#             TestResult = 0
#             print "ERROR", v1, v2, v3, v4
#             pdb.set_trace()
#             v1 = np.array(map.getLocalPosition(xglobdat[i, 4], xglobdat[i, 5]))
#             v2 = np.array(xdat[i, 4:6])
#             v3 = np.array(map.getGlobalPosition(v1[0], v1[1]))
#             v4 = np.array([xglobdat[i, 4], xglobdat[i, 5]])
#             print np.dot(v3 - v4, v3 - v4)
#             pdb.set_trace()

#     if TestResult == 1:
#         print "Change of coordinates test passed!"

