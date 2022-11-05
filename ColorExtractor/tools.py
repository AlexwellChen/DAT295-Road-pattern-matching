import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)

# green, yellow, red, dark red
bg_color = [[148, 200, 97], [225, 131, 49], [211, 45, 31],  [146, 36, 29]]
# color similarity threshold
threshold = 6000

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
    # cv.line(logo, (lng, 0), (lng, h), (0, 255, 255), thickness=2)
    # cv.line(logo, (0, lat), (w, lat), (255, 0, 0), thickness=2)
        
    # lng = lng_seq[0]
    # lat = lat_seq[0]
    
    # lng = int(h*lng)
    # lat = int(w*lat)
    
    # rest_point_delta = [(0, 0)]
    # max_lng_delta = 0
    # max_lat_delta = 0
    # for i in range(1, len(lng_seq)):
    #     lng_ = int(h*lng_seq[i])
    #     lat_ = int(w*lat_seq[i])
    #     print(lat_, lng_)
    #     max_lat_delta = max(max_lat_delta, abs(lat_-lat))
    #     max_lng_delta = max(max_lng_delta, abs(lng_-lng))
    #     rest_point_delta.append((lat_ - lat, lng_ - lng))
    # rest_point_delta = list(set(rest_point_delta))
    max_delta = max(max_lat_delta, max_lng_delta)
    # if max_delta % 2 != 0:
    #     max_delta += 1
    
    # direction = -1
    # if max_lat_delta > max_lng_delta:
    #     rest_point_delta.sort(key=lambda x: x[0])
    #     if len(rest_point_delta) != 1:
    #         dx = rest_point_delta[0][0] - rest_point_delta[1][0]
    #         if dx < 0:
    #             direction = 1 # down
    #         elif dx > 0:
    #             direction = 2 # up
    # else:
    #     rest_point_delta.sort(key=lambda x: x[1])
    #     print(rest_point_delta)
    # DDA interpolation
    
    interpolated_point = []
    for i in range(len(rest_point_delta) - 1):
        DDA(rest_point_delta[i][0], rest_point_delta[i][1],
            rest_point_delta[i+1][0], rest_point_delta[i+1][1], interpolated_point)
    interpolated_point.append(rest_point_delta[-1])
    interpolated_point = list(set(interpolated_point))
    min_val = 0
    for i in range(len(rest_point_delta)):
        min_val = min(min_val, rest_point_delta[i][0], rest_point_delta[i][1])

    for i in range(len(rest_point_delta)):
        rest_point_delta[i] = (rest_point_delta[i][0] - min_val, rest_point_delta[i][1] - min_val)
    
    interpolated_point = []
    print("rest_point_delta")
    print(rest_point_delta)
    for i in range(len(rest_point_delta) - 1):
        DDA(rest_point_delta[i][0], rest_point_delta[i][1],
            rest_point_delta[i+1][0], rest_point_delta[i+1][1], interpolated_point)

    width = max(max_delta + 1 - min_val, 12)
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
    print(kernel.shape)
    print(kernel)
    return kernel


def image_process(lng: float, lat: float, idx, output=None):
    # cv.line(logo, (lng, 0), (lng, h), (0, 255, 255), thickness=2)
    # cv.line(logo, (0, lat), (w, lat), (255, 0, 0), thickness=2)
    try:
        logo = cv.imread('./output/8756_8768_12124_12137_z15_t' +
                         str(idx*600)+'.png')
        logo = cv.cvtColor(logo, cv.COLOR_BGR2RGB)
        h, w = logo.shape[0:2]
        x = 12
        plt.imshow(logo)
        print(h*lng-x, h*lng+x, w*lat-x, w*lat+x)
        lng = int(h*lng)
        lat = int(w*lat)
        cropped = logo[max(0, lat-x): min(w, lat+x),
                       max(0, lng-x): min(h, lng+x)]  # [y0:y1, x0:x1]
        plt.imshow(cropped)
        # cv.imwrite("./tmp/" + output + ".png",
        #            cv.cvtColor(logo, cv.COLOR_RGB2BGR))
        # cv.line(logo, (0, 0), (w, h), (0, 255, 0), thickness=2)
        # cv.line(logo, (0, h), (w, 0), (0, 255, 0), thickness=2)

        t = [0, 0, 0, 0]
        for i in range(cropped.shape[0]):
            for j in range(cropped.shape[1]):
                if(cropped[i][j][0] == 255 and cropped[i][j][1] == 255 and cropped[i][j][2] == 255):
                    continue
                for k in range(len(bg_color)):
                    if calc_diff(cropped[i][j], k):
                        t[k] += 1
        return t.index(max(t))
    except:
        return -1


def checkRoadType(mask):
    pass

def point_delta(lng_seq, lat_seq, idx):
    logo = cv.imread('./output/8756_8768_12124_12137_z15_t' +
                         str(idx*600)+'.png')
    logo = cv.cvtColor(logo, cv.COLOR_BGR2RGB)
    h, w = logo.shape[0:2]
        
    lng = lng_seq[0]
    lat = lat_seq[0]
    
    lng = int(h*lng)
    lat = int(w*lat)

    rest_point_delta = [(0, 0)]
    max_delta = 0
    max_lat_delta = 0
    max_lng_delta = 0
    for i in range(1, len(lng_seq)):
        lng_ = int(h*lng_seq[i]) # x
        lat_ = int(w*lat_seq[i]) # y
        max_lat_delta = max(max_lat_delta, abs(lat_-lat))
        max_lng_delta = max(max_lng_delta, abs(lng_-lng))
        max_delta = max(max_delta, max(max_lat_delta, max_lng_delta))
        if max_delta < 100:
            rest_point_delta.append((lng_-lng, lat_-lat))
    rest_point_delta = set(rest_point_delta)
    rest_point_delta = list(rest_point_delta)
    return rest_point_delta, max_delta
    
def image_process_position_seq(lng_seq, lat_seq, idx, rest_point_delta, max_delta,  output=None):
    # cv.line(logo, (lng, 0), (lng, h), (0, 255, 255), thickness=2)
    # cv.line(logo, (0, lat), (w, lat), (255, 0, 0), thickness=2)
    try:
        logo = cv.imread('./output/8756_8768_12124_12137_z15_t' +
                         str(idx*600)+'.png')
        logo = cv.cvtColor(logo, cv.COLOR_BGR2RGB)
        
        h, w = logo.shape[0:2]
            
        lng = lng_seq[0]
        lat = lat_seq[0]
        
        lng = int(h*lng)
        lat = int(w*lat)
            
        print("First point: ", lng, lat)
        rest_point_delta = [(0, 0)]
        max_delta = 0
        max_lat_delta = 0
        max_lng_delta = 0
        for i in range(1, len(lng_seq)):
            lng_ = int(h*lng_seq[i]) # x
            lat_ = int(w*lat_seq[i]) # y
            max_lat_delta = max(max_lat_delta, abs(lat_-lat))
            max_lng_delta = max(max_lng_delta, abs(lng_-lng))
            max_delta = max(max_delta, max(max_lat_delta, max_lng_delta))
            if max_delta < 100:
                rest_point_delta.append((lng_-lng, lat_-lat))
        rest_point_delta = set(rest_point_delta)
        rest_point_delta = list(rest_point_delta)

        print("point delta: ", rest_point_delta)
        print("Max delta: ", max_delta)
        
        img_half_width = max(max_delta + 1, 50)
        point_on_map = np.zeros((2*img_half_width, 2*img_half_width))
        for point in rest_point_delta:
            point_on_map[point[0]+img_half_width][point[1]+img_half_width] = 1
        cropped = logo[max(0, lat-img_half_width): min(w, lat+img_half_width),
                       max(0, lng-img_half_width): min(h, lng+img_half_width)]  # [y0:y1, x0:x1]
        
        # Generate filter kernel
        kernel = gen_Filter(max_lat_delta, max_lng_delta, rest_point_delta[:])
        # Extract road
        cropped_road = np.zeros((cropped.shape[0], cropped.shape[1]))
        for i in range(cropped.shape[0]):
            for j in range(cropped.shape[1]):
                # Skip white pixel
                if(cropped[i][j][0] == 255 and cropped[i][j][1] == 255 and cropped[i][j][2] == 255):
                    continue
                for k in range(len(bg_color)):
                    if calc_diff(cropped[i][j], k):
                        cropped_road[i][j] = 1
                        break
        
        # Apply filter
        print("Kernel size: ", kernel.shape)
        print("Image size: ", cropped_road.shape)

        output = cv.filter2D(cropped_road, 1, kernel) # Ouptput is the mask of intrerested area
        threshold = int(kernel.sum() * 0.6)

        # Without interest area
        output[output < threshold] = 0
        output[output >= threshold] = 1
        
        # # With interest area
        # detect_range = 0
        # for i in range(len(output)):
        #     for j in range(len(output[0])):
        #         if output[i][j] >= threshold and abs(i - img_half_width) < max_delta + detect_range and abs(j - img_half_width) < max_delta + detect_range:
        #             output[i][j] = 1
        #         else:
        #             output[i][j] = 0

        # Todo: Determine the type of road


        # Todo: Determine which direction the road is going
        

        # Center crop to 24*24
        center_crop_output = output[img_half_width-12:img_half_width+12, img_half_width-12:img_half_width+12]
        center_crop_image = cropped[img_half_width-12:img_half_width+12, img_half_width-12:img_half_width+12]
        
        # Mix the point on map with the cropped image
        fig = plt.figure(figsize=[10,10])
        ax = fig.add_subplot(121)
        ax.imshow(cropped)
        ax.imshow(output, alpha=0.5)
        # ax.imshow(center_crop_image)
        # ax.imshow(center_crop_output, alpha=0.5)
        # subtitle
        ax.set_title("Interested road")

        ax = fig.add_subplot(122)
        ax.imshow(cropped)
        ax.imshow(point_on_map, alpha=0.5)
        ax.set_title("Actual trajectory")
        plt.show()
        
        t = [0, 0, 0, 0]
        for i in range(cropped.shape[0]):
            for j in range(cropped.shape[1]):
                if(cropped[i][j][0] == 255 and cropped[i][j][1] == 255 and cropped[i][j][2] == 255):
                    continue
                for k in range(len(bg_color)):
                    if calc_diff(cropped[i][j], k):
                        t[k] += 1
        return t.index(max(t))
    except:
        return -1