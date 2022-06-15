from tensorflow import keras
import cv2
import numpy as np
app = FastAPI()
############ url initialization ############
model_url="/home/hoola/code/mobilenet_parkinson/models/trained_model"

############ load model ##########
def load_model(model_url):
    model=keras.models.load_model(model_url)
    return model
########### predictions ########## 
def parkinson(image_url):
    dim=(224,224)
    file=cv2.imread(image_url)
    print(file)
    img=cv2.resize(file,dim)
    image=np.expand_dims(img,axis=0)
    image=image/255
    model=keras.models.load_model("/home/hoola/code/mobilenet_parkinson/models/trained_model")
    pred=model.predict(x=image, verbose=0)
    prediction=np.argmax(pred)
    if prediction ==0:
        label="Healthy"
    else:
        label="Parkinson"
    return label

@app.get("/")
def hello():
    return "Hello from Api"


@app.post("/prediction")
async def interpret(image_url):
    #print(await request.json())
    print(image_url)
    prediction=parkinson(image_url)
    return prediction;

