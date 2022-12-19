import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
from datetime import date, datetime, timedelta


np.set_printoptions(threshold=np.inf)

# green, yellow, red, dark red
bg_color = [[148, 200, 97], [225, 131, 49], [211, 45, 31], [146, 36, 29]]
# color similarity threshold
threshold = 6000


def num2deg(xtile, ytile, zoom):
	n = math.pow(2, zoom)
	lon_deg = xtile / n * 360.0 - 180.0
	lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
	lat_deg = math.degrees(lat_rad)
	return (lat_deg, lon_deg)


def timeCalc(daynum):
	start = datetime(2017, 11, 1)
	# UTC -> Local time
	start = start - timedelta(hours=5)
	test = start + timedelta(days=daynum - 1)
	# Monday is 0 and Sunday is 6.
	d = test.weekday()
	h = test.hour
	m = test.minute
	if (m % 10 > 5):
		m += (10 - m % 10)
	else:
		m -= (m % 10)
	idx = d * (144) + h * 6 + m
	idx %= 1008
	if d == 6:
		d = 0
	else:
		d += 1
	return idx


def calc_diff(pixel, i):
	return (
			       (pixel[0] - bg_color[i][0]) ** 2
			       + (pixel[1] - bg_color[i][1]) ** 2
			       + (pixel[2] - bg_color[i][2]) ** 2
	       ) < threshold


# Generate Conv filter by the lng_seq and lat_seq

def DDA(x1, y1, x2, y2, point_list):
	dx = x2 - x1
	dy = y2 - y1
	steps = max(abs(dx), abs(dy))
	x_inc = dx / steps
	y_inc = dy / steps
	x = x1
	y = y1
	for i in range(steps):
		point_list.append((int(x), int(y)))
		x += x_inc
		y += y_inc


def gen_Filter(max_lat_delta, max_lng_delta, rest_point_delta):
	max_delta = max(max_lat_delta, max_lng_delta)

	# Convert the absolute coordinates of the trajectory to relative coordinates
	min_val = 0
	for i in range(len(rest_point_delta)):
		min_val = min(min_val, rest_point_delta[i][0], rest_point_delta[i][1])
	
	for i in range(len(rest_point_delta)):
		rest_point_delta[i] = (rest_point_delta[i][0] - min_val, rest_point_delta[i][1] - min_val)
	
	interpolated_point = []
	for i in range(len(rest_point_delta) - 1):
		DDA(rest_point_delta[i][0], rest_point_delta[i][1],
		    rest_point_delta[i + 1][0], rest_point_delta[i + 1][1], interpolated_point)
	
	width = int(max(max_delta + 1 - min_val, 12))
	kernel_size = (width, width)
	kernel = np.zeros(kernel_size, np.uint8)
	for point in interpolated_point:
		kernel[point[0], point[1]] = 1
	# delet all zero row
	kernel = kernel[~np.all(kernel == 0, axis=1)]
	# delet all zero col
	kernel = kernel[:, ~np.all(kernel == 0, axis=0)]
	
	# padding to same size
	x_len = kernel.shape[0]
	y_len = kernel.shape[1]
	
	if x_len > y_len:
		# padding y equally both side
		padding = (x_len - y_len) // 2
		kernel = np.pad(kernel, ((0, 0), (padding, padding)), 'constant')
	elif x_len < y_len:
		# padding x equally both side
		padding = (y_len - x_len) // 2
		kernel = np.pad(kernel, ((padding, padding), (0, 0)), 'constant')
	
	x_len = kernel.shape[0]
	y_len = kernel.shape[1]
	if x_len != y_len:
		if x_len < y_len:
			# delete one col
			kernel = kernel[:, 1:]
		else:
			# delete one row
			kernel = kernel[1:, :]
	return kernel


def image_process(lng: float, lat: float, idx, output=None):
	# cv.line(logo, (lng, 0), (lng, h), (0, 255, 255), thickness=2)
	# cv.line(logo, (0, lat), (w, lat), (255, 0, 0), thickness=2)
	try:
		path = "/Users/alexwell/Desktop/DAT295-Road-pattern-matching/"
		logo = cv.imread(path + 'output/8756_8768_12124_12137_z15_t' +
		                 str(idx * 600) + '.png')
		logo = cv.cvtColor(logo, cv.COLOR_BGR2RGB)
		
		h, w = logo.shape[0:2]
		x = 12
		plt.imshow(logo)
		# print(h * lng - x, h * lng + x, w * lat - x, w * lat + x)
		lng = int(h * lng)
		lat = int(w * lat)
		cropped = logo[max(0, lat - x): min(w, lat + x),
		          max(0, lng - x): min(h, lng + x)]  # [y0:y1, x0:x1]
		# plt.imshow(cropped)
		# cv.imwrite("./tmp/" + output + ".png",
		#            cv.cvtColor(logo, cv.COLOR_RGB2BGR))
		# cv.line(logo, (0, 0), (w, h), (0, 255, 0), thickness=2)
		# cv.line(logo, (0, h), (w, 0), (0, 255, 0), thickness=2)
		
		t = [0, 0, 0, 0]
		for i in range(cropped.shape[0]):
			for j in range(cropped.shape[1]):
				if (cropped[i][j][0] == 255 and cropped[i][j][1] == 255 and cropped[i][j][2] == 255):
					continue
				for k in range(len(bg_color)):
					if calc_diff(cropped[i][j], k):
						t[k] += 1
		return t.index(max(t))
	except:
		return -1


def detectDirection(kernel, rest_point_delta):
	# Calculate the vector from the last point to the first point and normalize it
	vector_original = [rest_point_delta[-1][0] - rest_point_delta[0][0],
	                   rest_point_delta[-1][1] - rest_point_delta[0][1]]
	vector_len = math.sqrt(vector_original[0] ** 2 + vector_original[1] ** 2)
	vector = [vector_original[0] / vector_len, vector_original[1] / vector_len]
	print(vector)
	
	# Rotate the vector 90 degrees counterclockwise
	vector_rotated = np.array([-vector[1], vector[0]])
	'''
		Generate new kernel.
		Original point change to 2
		New points equal 1
	'''
	kernel_new = np.zeros(kernel.shape, dtype=np.uint8)
	i, j = 0, 0
	for i in range(len(kernel)):
		for j in range(len(kernel[0])):
			if kernel[i][j] == 1:
				kernel_new[i][j] = 2
				# rotated points should move 11 pixels on the rotated vector direction
				cord_x = i + int(vector_rotated[0] * 11)
				cord_y = j + int(vector_rotated[1] * 11)
				if cord_x >= 0 and cord_x < len(kernel) and cord_y >= 0 and cord_y < len(kernel[0]):
					kernel_new[cord_x][cord_y] = 1
				
				cord_x = i + int(vector_rotated[0] * 6)
				cord_y = j + int(vector_rotated[1] * 6)
				if cord_x >= 0 and cord_x < len(kernel) and cord_y >= 0 and cord_y < len(kernel[0]):
					kernel_new[cord_x][cord_y] = 1
	
	return kernel_new


def point_delta(lng_seq, lat_seq, idx):
	path = "/Users/alexwell/Desktop/DAT295-Road-pattern-matching/"
	logo = cv.imread(path + 'output/8756_8768_12124_12137_z15_t' +
	                 str(idx * 600) + '.png')
	logo = cv.cvtColor(logo, cv.COLOR_BGR2RGB)
	h, w = logo.shape[0:2]
	
	lng = lng_seq[0]
	lat = lat_seq[0]
	
	lng = int(h * lng)
	lat = int(w * lat)
	
	rest_point_delta = [(0, 0)]
	max_delta = 0
	max_lat_delta = 0
	max_lng_delta = 0
	for i in range(1, len(lng_seq)):
		lng_ = int(h * lng_seq[i])  # x
		lat_ = int(w * lat_seq[i])  # y
		max_lat_delta = max(max_lat_delta, abs(lat_ - lat))
		max_lng_delta = max(max_lng_delta, abs(lng_ - lng))
		max_delta = max(max_delta, max(max_lat_delta, max_lng_delta))
		if max_delta < 100:
			rest_point_delta.append((lng_ - lng, lat_ - lat))
	rest_point_delta = set(rest_point_delta)
	rest_point_delta = list(rest_point_delta)
	return rest_point_delta, max_delta


def image_process_position_seq(lng_seq, lat_seq, idx, rest_point_delta, max_delta, flag=False, origin=False):
	# cv.line(logo, (lng, 0), (lng, h), (0, 255, 255), thickness=2)
	# cv.line(logo, (0, lat), (w, lat), (255, 0, 0), thickness=2)
	try:
		path = "/Users/alexwell/Desktop/DAT295-Road-pattern-matching/"
		logo = cv.imread(path + 'output/8756_8768_12124_12137_z15_t' +
		                 str(idx * 600) + '.png')
		logo = cv.cvtColor(logo, cv.COLOR_BGR2RGB)
		
		h, w = logo.shape[0:2]
		print(h, w)
		
		lng = lng_seq[0]
		lat = lat_seq[0]
		
		lng = int(w * lng)
		lat = int(h * lat)
		
		max_delta = 0
		max_lat_delta = 0
		max_lng_delta = 0
		rest_point_delta = []
		for i in range(1, len(lng_seq)):
			lng_ = int(w * lng_seq[i])  # y 经度
			lat_ = int(h * lat_seq[i])  # x 纬度
			max_lat_delta = max(max_lat_delta, abs(lat_ - lat))
			max_lng_delta = max(max_lng_delta, abs(lng_ - lng))
			max_delta = max(max_delta, max(max_lat_delta, max_lng_delta))
			rest_point_delta.append((lat_ - lat, lng_ - lng))  # 上下是x，左右是y
		
		# rest_point_delta.append((lng_-lng, lat_-lat)) # 上下是x，左右是y
		rest_point_delta = set(rest_point_delta)
		rest_point_delta = list(rest_point_delta)
		
		# print("Length of rest_point_delta: ", len(rest_point_delta))
		# print(rest_point_delta)
		# print("Max delta: ", max_delta)
		
		img_half_width = max(max_delta + 1, 60)
		point_on_map = np.zeros((2 * img_half_width, 2 * img_half_width))
		for point in rest_point_delta:
			point_on_map[point[0] + img_half_width][point[1] + img_half_width] = 1
		cropped = logo[max(0, lat - img_half_width): min(h, lat + img_half_width),
		          max(0, lng - img_half_width): min(w, lng + img_half_width)]
		# print("cropped shape: ", cropped.shape)
		
		# Generate filter kernel
		kernel = gen_Filter(max_lat_delta, max_lng_delta, rest_point_delta[:])
		# Improve the kernel and mask
		direct_kernel = detectDirection(kernel, rest_point_delta)
		# Extract road
		cropped_road = np.zeros((cropped.shape[0], cropped.shape[1]))
		for i in range(cropped.shape[0]):
			for j in range(cropped.shape[1]):
				# Skip white pixel
				if (cropped[i][j][0] == 255 and cropped[i][j][1] == 255 and cropped[i][j][2] == 255):
					continue
				for k in range(len(bg_color)):
					if calc_diff(cropped[i][j], k):
						cropped_road[i][j] = 1
						break
		
		# Apply filter
		# print("Kernel size: ", kernel.shape)
		# print("Image size: ", cropped_road.shape)
		output = cv.filter2D(cropped_road, 1, direct_kernel)  # Ouptput is the mask of intrerested area
		original_kernel_output = cv.filter2D(cropped_road, 1, kernel)
		threshold = int(direct_kernel.sum() * 0.65)
		original_threshold = int(kernel.sum() * 0.65)

		# Without interest area
		output[output < threshold] = 0
		output[output >= threshold] = 1

		original_kernel_output[original_kernel_output < original_threshold] = 0
		original_kernel_output[original_kernel_output >= original_threshold] = 1
		
		# Center crop to w*w
		center_width = 24
		center_crop_output = output[img_half_width - center_width // 2:img_half_width + center_width // 2,
		                     img_half_width - center_width // 2:img_half_width + center_width // 2]
		center_crop_image = cropped[img_half_width - center_width // 2:img_half_width + center_width // 2,
		                    img_half_width - center_width // 2:img_half_width + center_width // 2]
		center_crop_original_kernel_output = original_kernel_output[img_half_width - center_width // 2:img_half_width + center_width // 2,
		                     img_half_width - center_width // 2:img_half_width + center_width // 2]
		
		show_img_flag = flag
		if show_img_flag:
			# Mix the point on map with the cropped image
			fig = plt.figure(figsize=[10, 10])
			ax = fig.add_subplot(121)
			ax.imshow(cropped)
			ax.imshow(output, alpha=0.5)
			# ax.imshow(center_crop_image)
			# ax.imshow(center_crop_output, alpha=0.5)
			# subtitle
			ax.set_title("Directed Interested road")
			
			ax = fig.add_subplot(122)
			ax.imshow(cropped)
			ax.imshow(original_kernel_output, alpha=0.5)
			ax.set_title("Orignial Interested road")
			plt.show()
		
		t = [0, 0, 0, 0]
		for i in range(center_crop_image.shape[0]):
			for j in range(center_crop_image.shape[1]):
				if (center_crop_image[i][j][0] == 255 and center_crop_image[i][j][1] == 255 and center_crop_image[i][j][
					2] == 255) or center_crop_output[i][j] == 0:
					continue
				for k in range(len(bg_color)):
					if calc_diff(center_crop_image[i][j], k):
						t[k] += 1
		dirtect_res = t.index(max(t))

		t = [0, 0, 0, 0]
		for i in range(center_crop_image.shape[0]):
			for j in range(center_crop_image.shape[1]):
				if (center_crop_image[i][j][0] == 255 and center_crop_image[i][j][1] == 255 and center_crop_image[i][j][
					2] == 255) or center_crop_original_kernel_output[i][j] == 0:
					continue
				for k in range(len(bg_color)):
					if calc_diff(center_crop_image[i][j], k):
						t[k] += 1
		original_res = t.index(max(t))
		return dirtect_res, original_res
	except Exception as e:
		raise e


def center_crop(img, dim):
	"""Returns center cropped image

  Args:
  img: image to be center cropped
  dim: dimensions (width, height) to be cropped from center
  """
	width, height = img.shape[1], img.shape[0]
	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
	mid_x, mid_y = int(width / 2), int(height / 2)
	cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
	crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
	return crop_img


def crop_center_np(img, cropx, cropy):
	y, x = img.shape
	startx = x // 2 - cropx // 2
	starty = y // 2 - cropy // 2
	return img[startx:startx + cropx, starty:starty + cropy]


def show_trajectory(lng_seq, lat_seq, idx):
	logo = cv.imread('./output/8756_8768_12124_12137_z15_t' +
	                 str(idx * 600) + '.png')
	logo = cv.cvtColor(logo, cv.COLOR_BGR2RGB)
	
	# times = 0.5
	# print(logo.shape)
	# logo = cv.resize(logo, (0, 0), fx = times, fy = times)
	# print(logo.shape)
	h, w = logo.shape[0:2]
	# h是长边 3328，w是短边 3072
	print(h, w)
	
	lng = lng_seq[0]
	lat = lat_seq[0]
	
	# lat 是上下 对应x， lng是左右 对应y
	# x对应h， y对应w
	lng = int(w * lng)
	lat = int(h * lat)
	
	max_delta = 0
	max_lat_delta = 0
	max_lng_delta = 0
	rest_point_delta = []
	for i in range(1, len(lng_seq)):
		lng_ = int(w * lng_seq[i])  # y 经度
		lat_ = int(h * lat_seq[i])  # x 纬度
		max_lat_delta = max(max_lat_delta, abs(lat_ - lat))
		max_lng_delta = max(max_lng_delta, abs(lng_ - lng))
		max_delta = max(max_delta, max(max_lat_delta, max_lng_delta))
		rest_point_delta.append((lat_ - lat, lng_ - lng))  # 上下是x，左右是y
	
	# rest_point_delta.append((lng_-lng, lat_-lat)) # 上下是x，左右是y
	rest_point_delta = set(rest_point_delta)
	rest_point_delta = list(rest_point_delta)
	print("Length of rest_point_delta: ", len(rest_point_delta))
	print(rest_point_delta)
	print("Max delta: ", max_delta)
	
	img_half_width = max(max_delta + 200, 60)
	point_on_map = np.zeros((2 * img_half_width, 2 * img_half_width))
	for point in rest_point_delta:
		point_on_map[point[0] + img_half_width][point[1] + img_half_width] = 1
	cropped = logo[max(0, lat - img_half_width): min(h, lat + img_half_width),
	          max(0, lng - img_half_width): min(w, lng + img_half_width)]
	
	# # point_img = Image.fromarray(point_on_map)
	# origin_w, origin_h = point_on_map.shape
	# times = 1
	# point_on_map = cv.resize(point_on_map, (int(point_on_map.shape[0]*times), int(point_on_map.shape[1]*times)))
	
	# # center crop to original size
	# point_on_map = crop_center_np(point_on_map, origin_w, origin_h)
	
	# point_on_map = np.array(point_img_resize)
	fig = plt.figure(figsize=[10, 10])
	ax = fig.add_subplot(111)
	ax.imshow(cropped)
	ax.imshow(point_on_map, alpha=0.5)
	ax.set_title("Actual trajectory")
	plt.show()


def show_trajectory_color(lng_seq, lat_seq, idx):
	path = "/Users/alexwell/Desktop/DAT295-Road-pattern-matching/"
	logo = cv.imread(path + 'output/8756_8768_12124_12137_z15_t' +
	                 str(idx * 600) + '.png')
	logo = cv.cvtColor(logo, cv.COLOR_BGR2RGB)
	
	# times = 0.5
	# print(logo.shape)
	# logo = cv.resize(logo, (0, 0), fx = times, fy = times)
	# print(logo.shape)
	h, w = logo.shape[0:2]
	# h是长边 3328，w是短边 3072
	print(h, w)
	
	lng = lng_seq[0]
	lat = lat_seq[0]
	
	# lat 是上下 对应x， lng是左右 对应y
	# x对应h， y对应w
	lng = int(w * lng)
	lat = int(h * lat)
	
	max_delta = 0
	max_lat_delta = 0
	max_lng_delta = 0
	rest_point_delta = []
	for i in range(1, len(lng_seq)):
		lng_ = int(w * lng_seq[i])  # y 经度
		lat_ = int(h * lat_seq[i])  # x 纬度
		max_lat_delta = max(max_lat_delta, abs(lat_ - lat))
		max_lng_delta = max(max_lng_delta, abs(lng_ - lng))
		max_delta = max(max_delta, max(max_lat_delta, max_lng_delta))
		rest_point_delta.append((lat_ - lat, lng_ - lng))  # 上下是x，左右是y
	
	# rest_point_delta.append((lng_-lng, lat_-lat)) # 上下是x，左右是y
	rest_point_delta = set(rest_point_delta)
	rest_point_delta = list(rest_point_delta)
	print("Length of rest_point_delta: ", len(rest_point_delta))
	print(rest_point_delta)
	print("Max delta: ", max_delta)
	
	img_half_width = max(max_delta + 20, 60)
	point_on_map = np.zeros((2 * img_half_width, 2 * img_half_width))
	for point in rest_point_delta:
		point_on_map[point[0] + img_half_width][point[1] + img_half_width] = 1
	cropped = logo[max(0, lat - img_half_width): min(h, lat + img_half_width),
	          max(0, lng - img_half_width): min(w, lng + img_half_width)]
	
	# # point_img = Image.fromarray(point_on_map)
	# origin_w, origin_h = point_on_map.shape
	# times = 1
	# point_on_map = cv.resize(point_on_map, (int(point_on_map.shape[0]*times), int(point_on_map.shape[1]*times)))
	
	# # center crop to original size
	# point_on_map = crop_center_np(point_on_map, origin_w, origin_h)
	
	# point_on_map = np.array(point_img_resize)
	fig = plt.figure(figsize=[5, 5])
	ax = fig.add_subplot(111)
	ax.imshow(cropped)
	ax.imshow(point_on_map, alpha=0.5)
	ax.set_title("Actual trajectory")
	# plt.show()
	color_code = input("Please input color code:\n0: Green, 1: Orange, 2: Red, 3: Dark Red, 4: Exit\n")
	fig.canvas.draw()
	fig.canvas.flush_events()
	plt.close(fig)
	# if color_code != "0" or color_code != "1" or color_code != "2" or color_code != "3" or color_code != "4":
	#     return -1
	if color_code == '':
		return -1
	return int(color_code)
