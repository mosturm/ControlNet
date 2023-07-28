import os
import glob
import cv2
import numpy as np



def check_split(path,t,value):
    xl,yl,rl,idel,split_l,s_pr_l,t_vl= np.loadtxt(path+'pos_GT.txt',skiprows=1, delimiter='\t', usecols=(0,1,2,3,4,5,6), unpack=True)
    id_t = idel[t_vl==t]
    parent = split_l[t_vl==t]
    num= int(np.max(idel))
    ide=value
    

    par = parent[id_t==ide]
    print('ide,par',ide,parent,t)
    if par != 0:
        print('ide,par',ide,par)
        return True,par[0]

    else:
        return False,0
    

    





def connect_matching_dots(img1, img2, path, t, cells, test=False):
    # Find unique pixel values in both images
    print('poss_val',np.linspace(100,255,12))
    unique_values_1 = np.unique(img1)
    unique_values_2 = np.unique(img2)

    # Create an output image for dots and lines separately
    output_img_dots = np.zeros_like(img1)
    output_img_lines = np.zeros_like(img1)
    output_img_splits = np.zeros_like(img1)

    # Initialize 3-channel image with zeros
    output_img_dots = np.stack((output_img_dots,)*3, axis=-1)
    output_img_lines = np.stack((output_img_lines,)*3, axis=-1)
    output_img_splits = np.stack((np.zeros_like(img1),)*3, axis=-1)

    # Draw dots in blue
    output_img_dots[:, :, 2] = cells  # RGB color for Blue 

    # Draw lines and circles in green
    for value in unique_values_1[unique_values_1 > 0]:
        # Find the coordinates of matching dots in both images
        coordinates_img1 = np.argwhere(img1 == value)
        coordinates_img2 = np.argwhere(img2 == value)

        if coordinates_img2.size > 0: 
            coord1 = np.mean(coordinates_img1, axis=0, dtype=int)
            coord2 = np.mean(coordinates_img2, axis=0, dtype=int)
            print('working...',coord1,coord2,value)
            cv2.line(output_img_lines, tuple(coord1[::-1]), tuple(coord2[::-1]), color=[0, 255, 0], thickness=2)
            cv2.circle(output_img_lines, tuple(coord2[::-1]), 3, color=[0, 255, 0], thickness=-1)  # Filled circle
        else:
            cs = check_split(path, t, value)
            print('cs', cs)
            if cs[0]:
                coordinates_img1 = np.argwhere(img1 == value)
                coordinates_img2 = np.argwhere(img2 == cs[1])
                if coordinates_img2.size > 0: 
                    coord1 = np.mean(coordinates_img1, axis=0, dtype=int)
                    coord2 = np.mean(coordinates_img2, axis=0, dtype=int)
                    
                    cv2.line(output_img_splits, tuple(coord1[::-1]), tuple(coord2[::-1]), color=[255, 0, 0], thickness=2)
                    cv2.circle(output_img_splits, tuple(coord2[::-1]), 3, color=[255, 0, 0], thickness=-1)  # Filled circle

    # Stack images along the channel dimension: Red for splits, Green for lines, Blue for dots
    output_img = output_img_dots + output_img_lines + output_img_splits
    return output_img




def ctc_2_track(i, res_path, id_path, save_path):
     # File names
    x_img_path = os.path.join(res_path, f"{i}.png")
    #y_img_path = os.path.join(id_path, f"{i}.jpg")
    #y_img2_path = os.path.join(id_path, f"{i-1}.jpg")

    # Load the images
    x_img = cv2.imread(x_img_path, 0)
    #y_img = cv2.imread(y_img_path, 0)
    #y_img2 = cv2.imread(y_img2_path, 0)

    # Error checking
    #if x_img is None or y_img is None or y_img2 is None:
    #    raise ValueError('One or more images could not be read')

    # Call connect_matching_dots function
     
    out = connect_matching_coords(x_img, x_img, id_path, i, x_img)

    output_file = os.path.join(save_path, f"{i}.jpg")
    cv2.imwrite(output_file, out)

    print('save_path',output_file)




    return 0




    
def revert_val_id(start,end,value,num):
    
    c_list = np.linspace(start,end,int(num))
    ind = np.abs(c_list - value).argmin()
    ind += 1

    return ind




def connect_matching_coords(img1, img2, path, t, cells, test=False):



    xl,yl,rl,idel,split_l,s_pr_l,t_vl= np.loadtxt(path+'pos_GT.txt',skiprows=1, delimiter='\t', usecols=(0,1,2,3,4,5,6), unpack=True)
    y0 = xl[t_vl==t]
    y1 = xl[t_vl==(t-1)]

    x0 = 1-yl[t_vl==t]
    x1 = 1-yl[t_vl==(t-1)]

    id0 = idel[t_vl==t]
    id1 = idel[t_vl==(t-1)]











    # Find unique pixel values in both images



    # Create an output image for dots and lines separately
    output_img_dots = np.zeros_like(img1)
    output_img_lines = np.zeros_like(img1)
    output_img_splits = np.zeros_like(img1)

    # Initialize 3-channel image with zeros
    output_img_dots = np.stack((output_img_dots,)*3, axis=-1)
    output_img_lines = np.stack((output_img_lines,)*3, axis=-1)
    output_img_splits = np.stack((np.zeros_like(img1),)*3, axis=-1)

    # Draw dots in blue
    output_img_dots[:, :, 2] = cells  # RGB color for Blue 

    # Draw lines and circles in green
    for value in id0:

        # Find the coordinates of matching dots in both images
        coordinates_img1 = (x0[id0==value][0]*512,y0[id0==value][0]*512)
        try:
            
            coordinates_img2 = (x1[id1==value][0]*512,y1[id1==value][0]*512)
            coord1 = tuple(map(int, coordinates_img1))
            coord2 = tuple(map(int, coordinates_img2))
            print('working...',coord1,coord2,value)
            cv2.line(output_img_lines, tuple(coord1[::-1]), tuple(coord2[::-1]), color=[55, 200, 0], thickness=3)
            cv2.circle(output_img_lines, tuple(coord2[::-1]), 8, color=[0, 255, 0], thickness=-1)  # Filled circle

        
        except:
            cs = check_split(path, t, value)
            print('cs', cs)
            if cs[0]:
                coordinates_img2 = (x1[id1==cs[1]][0]*512,y1[id1==cs[1]][0]*512)
                coord1 = tuple(map(int, coordinates_img1))
                coord2 = tuple(map(int, coordinates_img2))
                    
                cv2.line(output_img_splits, tuple(coord1[::-1]), tuple(coord2[::-1]), color=[200, 55, 0], thickness=3)
                cv2.circle(output_img_splits, tuple(coord2[::-1]), 8, color=[255, 0, 0], thickness=-1)  # Filled circle

       

    # Stack images along the channel dimension: Red for splits, Green for lines, Blue for dots
    output_img = output_img_dots + output_img_lines + output_img_splits
    return output_img

