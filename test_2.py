import cv2
import matplotlib.pyplot as plt
import os
import random

if not os.path.exists("TrainImages"):
    os.mkdir("TrainImages")

if not os.path.exists("TestImages"):
    os.mkdir("TestImages")

file_name_list = list()
for file_name in os.listdir('im'):
    split_by_under = file_name.split('.')[0].split('_')
    # print(split_by_under)
    if split_by_under[-1]=='1':
        file_name_list.append(file_name)

test_iamge_files_count = int(len(file_name_list)-(len(file_name_list)*0.2))

test_iamge_files = random.sample(file_name_list,test_iamge_files_count)

for file_name in test_iamge_files:
    path = os.getcwd()
    image_path = os.path.join("/im/",file_name)
    split_by_slash = image_path.split('/')[-1]
    img = cv2.imread(path+image_path)


# print(test_iamge_files)
    if not os.path.exists(f"TestImages/{split_by_slash.split('.')[0]}"):
        os.mkdir(f"TestImages/{split_by_slash.split('.')[0]}")
        os.mkdir(f"TestImages/{split_by_slash.split('.')[0]}/images")

    segment_image_path = f"TestImages/{split_by_slash.split('.')[0]}/images/{split_by_slash.split('.')[0]}.jpg"
    x = cv2.imwrite(segment_image_path, img)

for file_name in file_name_list:


    path = os.getcwd()
    image_path = os.path.join("/im/",file_name)
    split_by_slash = image_path.split('/')[-1]
    img = cv2.imread(path+image_path)


    if not os.path.exists(f"TrainImages/{split_by_slash.split('.')[0]}"):
        os.mkdir(f"TrainImages/{split_by_slash.split('.')[0]}")
        os.mkdir(f"TrainImages/{split_by_slash.split('.')[0]}/images")
        os.mkdir(f"TrainImages/{split_by_slash.split('.')[0]}/masks")



    segment_image_path  = f"TrainImages/{split_by_slash.split('.')[0]}/images/{split_by_slash.split('.')[0]}.jpg"
    x = cv2.imwrite(segment_image_path,img)


    rows,cols, _ = img.shape
    # print("Rows",rows)
    # print("Cols",cols)

    f = open(f"{path}/xyc/{split_by_slash.split('.')[0]}.xyc","r")
    content_of_file = f.read()

    # print(content_of_file)
    splited_by_n = content_of_file.split('\n')

    x_y_cor = list()
    for i in splited_by_n:
        if len(i) > 0:
            splited_by_t = tuple(i.split('\t'))
            x_y_cor.append(splited_by_t)

    x_y_cor

    after_plus_minus_list = list()
    for tup in x_y_cor:
        # print(tup)

        temp_plus_lst = list()
        temp_minus_lst = list()
        for values in tup:
            temp_plus_lst.append(int(values) + 50)
            temp_minus_lst.append(int(values) - 50)

        after_plus_minus_list.append([temp_minus_lst, temp_plus_lst])

    if len(after_plus_minus_list)<=0:
        print("This image has not much blast cells")

    for i,j in after_plus_minus_list:
      cv2.rectangle(img, tuple(i), tuple(j), (0,0,0), 1)

    image_seg = img.copy()

    if not os.path.exists('TrainImages'):
        os.mkdir('TrainImages')
    import PIL

    for minus, plus in after_plus_minus_list:
        # print("(", plus[-1], ":", minus[-1], ")", ",(", plus[0], ":", minus[0], ")", )
        x1 = plus[-1]
        y1 = minus[-1]

        x2 = plus[0]
        y2 = minus[0]

        plt.subplot(2, 6, 2)
        plt.title(str(minus))
        plt.axis("off")
        roi = image_seg[y1:x1, y2:x2]

        # x = cv2.imwrite(f"{path}/TrainImages/{split_by_slash.split('.')[0]}_{str(minus)}.jpg", roi)
        # seg_image_name = f"TrainImages/{split_by_slash.split('.')[0]}/masks/{split_by_slash.split('.')[0]}.jpg"
        seg_image_name =f"{path}/TrainImages/{split_by_slash.split('.')[0]}_({str(plus[-1])},{str(minus[-1])}),.jpg"
        x = cv2.imwrite(seg_image_name,roi)


        im1 = PIL.Image.open("im/Bl.png")
        im2 = PIL.Image.open(seg_image_name)
        im2.convert('1')

        im1.paste(im2,(plus[-1],minus[-1]))
        mask_image_name = f"TrainImages/{split_by_slash.split('.')[0]}/masks/{split_by_slash.split('.')[0]}_({str(plus[-1])},{str(minus[-1])}.png"

        im1.save(mask_image_name, quality=95)
        os.remove(seg_image_name)

