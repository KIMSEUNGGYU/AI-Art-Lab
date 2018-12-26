#-*- coding: utf-8 -*-

from PIL import Image
import os
import sys

count = 0
def image_crop( infilename , save_path):
    """
    image file 와 crop한이미지를 저장할 path 을 입력받아 crop_img를 저장한다.
    :param infilename:
        crop할 대상 image file 입력으로 넣는다.
    :param save_path:
        crop_image file의 저장 경로를 넣는다.
    :return:
    """

    img = Image.open( infilename )
    (img_h, img_w) = img.size

    # crop 할 사이즈 : grid_w, grid_h
    grid_w = 256 # crop width
    grid_h = 256 # crop height
    range_w = (int)(img_w/grid_w)
    range_h = (int)(img_h/grid_h)

    i = 0

    if not os.path.exists('data'):
        print(' [NO DATA DIRECTORY] ')
        sys.exit(1)
    if not os.path.exists('crop'):
        print(' [NO CROP DIRECTORY ] => make crop directory')
        os.makedirs('crop')

    save_path = save_path + str(count)+'/'
    if not os.path.exists(save_path):
        # print(' [NO CROP DIRECTORY ] => make crop directory')
        os.makedirs(save_path)

    global count
    count += 1

    for w in range(range_w):
        for h in range(range_h):
            bbox = (h*grid_h, w*grid_w, (h+1)*(grid_h), (w+1)*(grid_w))
            # print(h*grid_h, w*grid_w, (h+1)*(grid_h), (w+1)*(grid_w))
            # 가로 세로 시작, 가로 세로 끝
            crop_img = img.crop(bbox)

            fname = "{:05d}_{:05d}.jpg".format(count ,i)

            savename = save_path + fname
            crop_img.save(savename)
            print('save file ' + savename + '....')
            i += 1

if __name__ == '__main__':
    for (path, dir, files) in os.walk("./data"):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.png':
                image_crop('./data/'+filename, './crop/')
