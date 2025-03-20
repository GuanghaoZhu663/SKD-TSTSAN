import cv2
import os
import numpy as np
import pandas as pd

def pol2cart(rho, phi): #Convert polar coordinates to cartesian coordinates for computation of optical strain
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def computeStrain(u, v):
    u_x= u - pd.DataFrame(u).shift(-1, axis=1)
    v_y= v - pd.DataFrame(v).shift(-1, axis=0)
    u_y= u - pd.DataFrame(u).shift(-1, axis=0)
    v_x= v - pd.DataFrame(v).shift(-1, axis=1)
    os = np.array(np.sqrt(u_x**2 + v_y**2 + 1/2 * (u_y+v_x)**2).ffill(axis=1).ffill(axis=0))
    return os

def calculate_optical_flow(img1, img2):
    frame1 = cv2.imread(img1, 0)
    frame2 = cv2.imread(img2, 0)

    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(frame1, frame2, None)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    u, v = pol2cart(magnitude, angle)
    os_ = computeStrain(u, v)

    final_u = cv2.resize(u, (48, 48))
    final_v = cv2.resize(v, (48, 48))
    final_os = cv2.resize(os_, (48, 48))

    if ((np.max(final_u) - np.min(final_u))==0):
        normalized_u = final_u.astype(np.uint8)
    else:
        normalized_u = ((final_u - np.min(final_u)) / (np.max(final_u) - np.min(final_u)) * 255).astype(np.uint8)

    if ((np.max(final_v) - np.min(final_v))==0):
        normalized_v = final_v.astype(np.uint8)
    else:
        normalized_v = ((final_v - np.min(final_v)) / (np.max(final_v) - np.min(final_v)) * 255).astype(np.uint8)

    if ((np.max(final_os) - np.min(final_os))==0):
        normalized_os = final_os.astype(np.uint8)
    else:
        normalized_os = ((final_os - np.min(final_os)) / (np.max(final_os) - np.min(final_os)) * 255).astype(np.uint8)

    return normalized_u, normalized_v, normalized_os

def main(input_folder, output_folder):
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        out_folder_path = os.path.join(output_folder, folder_name)

        os.makedirs(out_folder_path, exist_ok=True)

        onset_img = [img for img in os.listdir(folder_path) if img.endswith("onset.jpg")]
        apex_img = [img for img in os.listdir(folder_path) if img.endswith("apex.jpg")]
        offset_img = [img for img in os.listdir(folder_path) if img.endswith("offset.jpg")]


        for i in range(len(apex_img)):
            flow_1_u, flow_1_v, flow_1_os = calculate_optical_flow(os.path.join(folder_path, onset_img[i]), os.path.join(folder_path, apex_img[i]))
            flow_2_u, flow_2_v, flow_2_os = calculate_optical_flow(os.path.join(folder_path, apex_img[i]), os.path.join(folder_path, offset_img[i]))

            output_filename_1_u = "_".join(onset_img[i].split('_')[:-1]) + "_1_u" + '.jpg'
            output_filename_1_v = "_".join(onset_img[i].split('_')[:-1]) + "_1_v" + '.jpg'
            output_filename_2_u = "_".join(onset_img[i].split('_')[:-1]) + "_2_u" + '.jpg'
            output_filename_2_v = "_".join(onset_img[i].split('_')[:-1]) + "_2_v" + '.jpg'
            output_path_1_u = os.path.join(out_folder_path, output_filename_1_u)
            output_path_1_v = os.path.join(out_folder_path, output_filename_1_v)
            output_path_2_u = os.path.join(out_folder_path, output_filename_2_u)
            output_path_2_v = os.path.join(out_folder_path, output_filename_2_v)

            cv2.imwrite(output_path_1_u, flow_1_u)
            cv2.imwrite(output_path_1_v, flow_1_v)
            cv2.imwrite(output_path_2_u, flow_2_u)
            cv2.imwrite(output_path_2_v, flow_2_v)



if __name__ == "__main__":
    input_folder = 'Dataset/CASME2_onset_apex_offset_retinaface'
    output_folder = "Dataset/CASME2_optflow_retinaface"
    main(input_folder, output_folder)
