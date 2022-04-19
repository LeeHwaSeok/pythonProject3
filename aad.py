# 프로젝트 메인 [ Open CV ]
import dlib
import cv2
import math
import numpy as np

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
  return result

ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))


detector = dlib.get_frontal_face_detector()      #얼굴 감지 /  dlib.get_frontal_face_detector -> 얼굴에 바운딩 박스 생성
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') #얼굴 예측 / 68개의 점으로 예측한다.


#객체 생성
vid_in = cv2.VideoCapture(0)
# "---" 비디오 파일로 쓰고싶으면 아래와 같이 사용하쇼
#vid_in = cv2.VideoCapture("baby_vid.mp4")

#빋아온 이미지를 반복
# -> 동영상 처럼 보이게
while True:
    # 동영상 프레임 갖고오기
    # ret = 1이면 프레임을 정상적으로 받아옴 / 0이면 반대
    ret, image_o = vid_in.read()
    #imgae_o로 받아온 프레임 출력

   # 비디오 사이즈 재정리   화면 해상도 4:3 비 -> 640 ; 480
    image = cv2.resize(image_o, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    roi = image         # roi 선언
    rotate_angle = 0.0  #rotate_angle 선언


    # 얼굴 가져오기 (up-sampling=1)
    face_detector = detector(img_gray, 1)
    #face 숫자 count
#    print("The number of faces detected : {}".format(len(face_detector)))
    # 받아온 얼굴 수만큼 반복
    # 하나의 루프는 하나의 얼굴입니다.
    for face in face_detector:
        # 얼굴을 직사각형으로 감싸기
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()),(0, 255, 0), 3)
        roi = image[face.top():face.bottom(),face.left():face.right()]   #원본임 조심
        print("TOP{},BOTTOM{},LEFT{},RIGHT{}".
              format(face.top(),face.bottom(),face.left(),face.right()))
        #경계박스 roi 좌표 출력력


        #경계값을 기준으로 x,y축 계산
        roi_x_axis = face.bottom()-face.top()
        roi_y_axis = face.right()-face.left()
        roi_area = roi_x_axis * roi_y_axis        #-> 연산이 많이 들어가서 넓이보다는 한 면을 사용하는게 효율적임.
        roi_per = roi_y_axis/480 * 100
        print("x축 값{}, y축 값{}, roi넓이{}, 화면차지 비율{:.2f}%".
              format(roi_x_axis,roi_y_axis,roi_area,roi_per))
        #사각형 넓이 식은 x*y 임.


        if roi_per > 50:
            print("화면에 가깝습니다. 화면 차지 비율{}",roi_per)
            cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 4)
        elif roi_per < 30:
            print("화면에서 멉미다./n 화면 차지 비율{}",roi_per)
            cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 4)

        else :
            pass
       # 얼굴을 예측하고 numpy 배열로 변환
        landmarks = predictor(image, face)  # 얼굴에서 68개 점 찾기

        #랜드마크 리스트 만들기
        landmark_list = []
        # append (x, y) in landmark_list
        for p in landmarks.parts():
            landmark_list.append([p.x, p.y])
            cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), -1)
            tan_theta = (landmarks.part(30).x - landmarks.part(27).x) / (landmarks.part(30).y- landmarks.part(27).y)
            theta = np.arctan(tan_theta)
            rotate_angle = theta * 180 / math.pi
    #    cv2.circle(image, (landmarks.part(30).x, landmarks.part(30).y), 5, (255, 0, 255), -1)# 딸기 코

    img_rotate = rotate_image(roi,rotate_angle)
    cv2.imshow('result', image)
    cv2.imshow("roi",roi)
    cv2.imshow("rote",img_rotate)

    # wait for keyboard input
    key = cv2.waitKey(1)

    # if esc,
    if key == 27:
        break

vid_in.release()


