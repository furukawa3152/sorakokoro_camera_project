import cv2
from mtcnn.mtcnn import MTCNN
from PIL import Image
import streamlit as st
import numpy as np


def sorakoko_judge(test_img):
    from keras.models import load_model
    import numpy as np
    from keras_preprocessing.image import img_to_array, load_img

    sorakoko_model = "sorakokoro_face_model.h5"
    model = load_model(sorakoko_model)

    # img_path = (test_img)
    # img = img_to_array(load_img(img_path, target_size=(50,50)))
    img = cv2.resize(test_img, (50, 50))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_nad = img_to_array(img) / 256
    img_nad = img_nad[None, ...]

    label = ["kokoro", "sora"]
    pred = model.predict(img_nad, batch_size=1, verbose=0)
    score = np.max(pred)
    pred_label = label[np.argmax(pred[0])]
    return (pred_label, score)


def face_detect_MTCNN(img):
    # img = cv2.imread(img)
    b_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detect = MTCNN()
    faces = detect.detect_faces(b_img)
    face_result = []
    for i in faces:
        if i["confidence"] > 0.8:
            face_result.append(i)
    if len(face_result) == 0:
        return (img, "顔を検出出来ません。")
    else:
        (x, y, w, h) = face_result[0]["box"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # cv2.imshow("test",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        face_image = img[y:y + h, x:x + w]
        name, count = sorakoko_judge(face_image)
        if name == "kokoro":
            jname = "こころちゃん"
        elif name == "sora":
            jname = "そらちゃん"
        count = int(count * 10000) / 100
        return (img, f"{count}%{jname}です")


def scale_to_width(img, width):  # PIL画像をアスペクト比を固定してリサイズする。
    im_height = img.height
    im_width = img.width
    aspect = im_height / im_width
    if im_height > im_width:
        return img.resize((width, int(width * aspect)))
    else:
        return img.resize((int(width / aspect), width))


if __name__ == '__main__':
    st.title("～～～そらここカメラだよ!～～～")
    uploaded_file_h = st.file_uploader("写真を入れてね", type=["png", "jpg","jpeg"], accept_multiple_files=False)
    if uploaded_file_h is not None:
        image = Image.open(uploaded_file_h)
        image = scale_to_width(image, 1000)  # リサイズ
        image = np.array(image.convert("RGB"))
        image = cv2.cvtColor(image, 1)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = face_detect_MTCNN(image)

        result_image = image[0]
        im_height = result_image.shape[1]
        im_width = result_image.shape[0]
        aspect = im_height / im_width
        result_image = cv2.resize(result_image, (int(500 * aspect), 500))
        comment = image[1]

        st.image(result_image, caption=comment)

    uploaded_file = st.file_uploader("回転してしまうときはこちらから", type=["png", "jpg","jpeg"], accept_multiple_files=False)
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = scale_to_width(image, 1000)  # リサイズ
        image = np.array(image.convert("RGB"))
        image = cv2.cvtColor(image, 1)
        image = face_detect_MTCNN(image)

        result_image = image[0]
        im_height = result_image.shape[1]
        im_width = result_image.shape[0]
        aspect = im_height / im_width
        result_image = cv2.resize(result_image, (int(500 * aspect), 500))
        comment = image[1]

        st.image(result_image, caption=comment)

