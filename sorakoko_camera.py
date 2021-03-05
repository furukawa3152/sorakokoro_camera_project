import cv2
from mtcnn.mtcnn import MTCNN
from PIL import Image
import streamlit as st
import numpy as np

def sorakoko_judge(test_img):
    from keras.models import load_model
    import numpy as np
    from keras_preprocessing.image import img_to_array, load_img

    sorakoko_model = r"C:\Users\user\PycharmProjects\sorakokoro_camera_project\sorakokoro_face_model.h5"
    model = load_model(sorakoko_model)

    # img_path = (test_img)
    # img = img_to_array(load_img(img_path, target_size=(50,50)))
    img = cv2.resize(test_img,(50,50))
    img_nad = img_to_array(img)/255
    img_nad = img_nad[None, ...]

    label = ["kokoro","sora"]
    pred = model.predict(img_nad, batch_size=1, verbose=0)
    score = np.max(pred)
    pred_label = label[np.argmax(pred[0])]
    return (pred_label,score)

def face_detect_MTCNN(img):
    # img = cv2.imread(img)
    b_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detect =MTCNN()
    faces =  detect.detect_faces(b_img)
    (x, y, w, h) = faces[0]["box"]
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # cv2.imshow("test",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    face_image = img[y:y + h, x:x + w]
    name,count = sorakoko_judge(face_image)
    return (img,name,count)

def scale_to_width(img,width):  # PIL画像をアスペクト比を固定してリサイズする。
    im_height = img.height
    im_width = img.width
    aspect = im_height / im_width
    if im_height > im_width:
        return img.resize((width, int(width * aspect)))
    else:
        return img.resize((int(width / aspect), width))


if __name__ == '__main__':
    st.title("そらここカメラだよ！")
    uploaded_file_h = st.file_uploader("撮影は縦で！", type=["png", "jpg"], accept_multiple_files=False)
    if uploaded_file_h is not None:
        image = Image.open(uploaded_file_h)
        image = scale_to_width(image, 800)  # リサイズ
        image = np.array(image.convert("RGB"))
        image = cv2.cvtColor(image, 1)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = face_detect_MTCNN(image)
        if image[1] == "kokoro":
            name = "こころちゃん"
        elif image[1] == "sora":
            name = "そらちゃん"
        count = int(image[2]*100)

        st.image(image[0],caption=f"{count}%{name}です")




