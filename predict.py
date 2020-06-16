from keras.models import load_model
import cv2
import numpy as np

model = load_model('model3.h5')

wheel = cv2.imread('wheel.jpg')
wheel_center = tuple(np.array(wheel.shape[1::-1]) / 2)

smoothed_angle = 0

for i in range(45406):
	frame = cv2.imread('driving_dataset/' + str(i) + '.jpg')
	cv2.imshow('frames', frame)
	img = cv2.GaussianBlur(cv2.resize(frame, (200, 66)), (3,3), cv2.BORDER_DEFAULT)

	rad = model.predict(img.reshape(1, 66, 200, 3))
	degrees = rad * (180 / np.pi)
	smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
	rot_mat = cv2.getRotationMatrix2D(wheel_center, -smoothed_angle, 1)
	result = cv2.warpAffine(wheel, rot_mat, wheel.shape[1::-1], flags=cv2.INTER_LINEAR)
	cv2.imshow('steering wheel', result)
	if cv2.waitKey(1) == ord('q'):
		break
cv2.destroyAllWindows()