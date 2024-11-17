
import cv2
import matplotlib.pyplot as plt

# Load the photograph (use the correct file path)
image_path = "D://DL LAB//WhatsApp Image 2024-08-24 at 00.10.56.jpeg"  # Provide the full path to your image
pixels = cv2.imread(image_path)

# Convert BGR image (used by OpenCV) to RGB (used by Matplotlib)
pixels_rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)

# Load the pre-trained model (provide the full path to XML file if needed)
cascade_path = 'D://DL LAB//haarcascade_frontalface_default.xml'  # Full path to Haar Cascade
classifier = cv2.CascadeClassifier(cascade_path)

# Perform face detection
bboxes = classifier.detectMultiScale(pixels_rgb)

# Check if any faces were detected
if len(bboxes) == 0:
    print("No faces detected")
else:
    # Print bounding box for each detected face
    for box in bboxes:
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # Draw a rectangle over the pixels
        cv2.rectangle(pixels_rgb, (x, y), (x2, y2), (255, 0, 0), 2)  # Blue rectangle in RGB

    # Use Matplotlib to display the image
    plt.imshow(pixels_rgb)
    plt.title('Face Detection')
    plt.axis('off')  # Hide axes
    plt.show()

# cascade_path = "D://DL LAB//haarcascade_frontalface_default.xml"
# classifier = cv2.CascadeClassifier(cascade_path)

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("camera is not opened")
#     exit()

# print("press q for quiting")

# while True:
#     ret,frame = cap.read()

#     if not ret:
#         print("something is wrong")
#         break

#     frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

#     bboxes = classifier.detectMultiScale(frame_rgb)

#     for (x,y,w,h) in bboxes:
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

#     cv2.imshow('real time detection',frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()