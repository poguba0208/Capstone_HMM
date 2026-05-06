import cv2
from sixdrepnet import SixDRepNet

pose_model = SixDRepNet(gpu_id=0)

def get_head_pose(face_crop):
    if face_crop is None:
        print("크롭된 얼굴 이미지가 없습니다.")
        return None
    
    pitch, yaw, roll = pose_model.predict(face_crop)
    
    print(f"yaw (좌우): {yaw[0]:.2f}°")
    print(f"pitch (상하): {pitch[0]:.2f}°")
    
    return yaw[0], pitch[0]

if __name__ == "__main__":
    from detect_face import detect_face
    
    result = detect_face('/workspace/capstone_pipeline/test_images/test_sulyoon.jpg')
    if result:
        face_crop, face_ratio = result
        pose = get_head_pose(face_crop)
        if pose:
            yaw, pitch = pose
            print("head_pose 완료!")