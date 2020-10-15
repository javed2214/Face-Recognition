from cv2 import cv2
import face_recognition as fc

image1 = fc.load_image_file('images/BarackObama1.jpg')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

image2 = fc.load_image_file('images/BarackObama2.jpg')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

faceLoc1 = fc.face_locations(image1)[0]
faceLoc2 = fc.face_locations(image2)[0]

encoding1 = fc.face_encodings(image1)[0]
encoding2 = fc.face_encodings(image2)[0]

print(faceLoc1)
print(faceLoc2)

image1_x1y1 = (faceLoc1[3], faceLoc1[0])
image1_x2y2 = (faceLoc1[1], faceLoc1[2])

image2_x1y1 = (faceLoc2[3], faceLoc2[0])
image2_x2y2 = (faceLoc2[1], faceLoc2[2])

res = fc.compare_faces([encoding1], encoding2)
print(res)

cv2.rectangle(image1, image1_x1y1, image1_x2y2, (255, 0, 255), 2)
cv2.rectangle(image2, image2_x1y1, image2_x2y2, (255, 0, 255), 2)

cv2.putText(image2, f'{res}', (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

cv2.imshow('Barack Obama - 1', image1)
cv2.imshow('Barack Obama - 2', image2)


cv2.waitKey(0)