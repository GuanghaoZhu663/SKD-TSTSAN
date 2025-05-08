import glob
import cv2
from RetinaFace.tools import FaceDetector
import os


def crop_images_CASME2_retinaface():
    face_det_model_path = "RetinaFace/Resnet50_Final.pth"
    face_detection = FaceDetector(face_det_model_path)

    main_folder_path = "Dataset/CASME2_RAW_selected"

    for sub_folder_name in os.listdir(main_folder_path):
        sub_folder_path = os.path.join(main_folder_path, sub_folder_name)

        if os.path.isdir(sub_folder_path):
            print(f"Processing sub-folder: {sub_folder_name}")

            for sub_sub_folder_name in os.listdir(sub_folder_path):
                sub_sub_folder_path = os.path.join(sub_folder_path, sub_sub_folder_name)

                if os.path.isdir(sub_sub_folder_path):
                    print(f"Processing sub-sub-folder: {sub_sub_folder_name}")

                    index = 0
                    face_left = 0
                    face_right = 0
                    face_top = 0
                    face_bottom = 0
                    for img_file_path in glob.glob(os.path.join(sub_sub_folder_path, '*.jpg')):
                        if index == 0:
                            image = cv2.imread(img_file_path)

                            face_left, face_top, face_right, face_bottom = face_detection.cal(image)

                            face = image[face_top:face_bottom + 1, face_left:face_right + 1, :]
                            face = cv2.resize(face, (128, 128))

                            cv2.imwrite(img_file_path, face)
                        else:
                            image = cv2.imread(img_file_path)

                            face = image[face_top:face_bottom + 1, face_left:face_right + 1, :]
                            face = cv2.resize(face, (128, 128))

                            cv2.imwrite(img_file_path, face)
                        index += 1
    print("Face cropping and saving complete.")


if __name__ == '__main__':
    crop_images_CASME2_retinaface()
