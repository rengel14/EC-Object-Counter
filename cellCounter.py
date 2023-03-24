import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

print('Starting process to count cells in images.')
print('Image sets should be saved into a folder named "input" in the same folder as this file. Each set saved into its own subfolder')
print('Results will be output into a folder called "output" along with a csv file containing the counts')

temp = input('Enter the lower bound micron to exclude spots smaller than this number, or just hit enter for default of 8: ')
low_bound = 8 if temp.strip() == '' else float(temp)

temp = input('Enter the higher bound micron to exclude spots larger than this number, or just hit enter for default of 500: ')
high_bound = 500 if temp.strip() == '' else float(temp)

temp = input('Enter the number of microns per pixel, or just hit Enter for default of 2.40467')
pixelToMicron = 2.40367 if temp.strip() == '' else float(temp)
pixelToMicron = pixelToMicron ** 2

baseDir = ''
input = 'input/'
output = 'output_low=' + str(low_bound) + '_high=' + str(high_bound) + "_v3"
folders = os.listdir(baseDir + input)
df = pd.DataFrame({'folderName' : [], 'fileName': [], 'cellCount': []})

low_bound = low_bound *  pixelToMicron
high_bound = high_bound *  pixelToMicron

try:
    os.mkdir(baseDir + output)
    print('Created folder ' + baseDir + output)    
except FileExistsError as e:
    print('Skipping Folder Generation')
for folder in folders:
    try:
        os.mkdir(baseDir + output + '/' + folder)
        print('Created folder ' + baseDir + output + '/' + folder)
    except FileExistsError as e:
        print('Skipping Folder Generation')
    files = os.listdir(baseDir + input + folder)
    for file in files:
        if ".tif" not in file:
            continue
        image = cv2.imread(baseDir + input + folder + '/' + file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if(np.average(image) > 40):
            image = cv2.bitwise_not(image)

        blur = cv2.GaussianBlur(image, (5, 5), 0)

        canny = cv2.Canny(blur, 1, 5, 3)

        dilated = cv2.dilate(canny, (1,1), iterations=5)

        (cnt, hierarchy) = cv2.findContours(
            dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        rowList = [folder, file]

        temp = list()
        for i in cnt:
            if(cv2.contourArea(i) >= low_bound and cv2.contourArea(i) <= high_bound):
                temp.append(i)
        cnt=tuple(temp)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 1)
        
        fig=plt.figure(figsize=[9.8, 7.2])
        ax=fig.add_subplot(1,1,1)
        plt.axis('off')
        plt.imshow(result)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(baseDir + output + '/' + folder + '/' + file[:-4] + "_output" + ".tif", bbox_inches=extent, dpi=300)
        plt.close()
        rowList.append(str(len(cnt)))
        df.loc[len(df)] = rowList
        print("Completed " + file)
        
df.to_csv(baseDir + output + '/' + output + '.csv', index=False)
