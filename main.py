import torch
import torch.nn as nn
from torchvision.transforms import v2
import cv2

#Define the model
class Emote_detector(nn.Module):
  def __init__(self):
    super().__init__()

    # Block 1
    self.block1=nn.Sequential(
        nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,stride=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    # Block 2
    self.block2=nn.Sequential(
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1,stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.Dropout2d(p=0.1)
    )
    # Block 3
    self.block3=nn.Sequential(
        nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1,stride=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1,stride=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.Dropout2d(p=0.2)
    )
    # Decision Block
    self.classifier=nn.Sequential(
        # nn.AdaptiveAvgPool2d((1,1)),  the model started to underfit cuz of this so for now removed it
        nn.Flatten(),
        nn.Linear(in_features=256*6*6,out_features=256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=256,out_features=7)
    )

  def forward(self,x):
    x=self.block1(x)
    x=self.block2(x)
    x=self.block3(x)
    x=self.classifier(x)
    return x
  
def main():
    # we dont train the model so change the device from cuda to cpu
    device=torch.device("cpu")
    model=Emote_detector()

    #load modal parameters
    model.load_state_dict(torch.load("model_params/emotion_detector.pth",map_location=device)) # if we put slash in front of model_params then it will start looking from the system directory so remove it
    model.eval() #disables dropout so we can use the full model

    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # transformin an image to image tensor
    transform=v2.Compose([
       v2.ToImage(),
       v2.ToDtype(torch.float32,scale=True),
       v2.Normalize([0.5],[0.5]),
       v2.Resize((48,48))
    ])

    # we need to detect faces from the image so using a pretrained ai model for that
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap=cv2.VideoCapture(0) # 0 means our laptop camera

    print("Press q to quit")

    #Webcam loop
    while True:
       ret,frame=cap.read()
       if not ret:
          break
       
       #we need grayscale colors for our model
       gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

       #detecting faces 
       faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

       for (x,y,w,h) in faces:
          cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

          roi_gray=gray[y:y+h,x:x+w]

          img_tensor=transform(roi_gray).unsqueeze(0)

          with torch.no_grad():
             logits=model(img_tensor)
             _,index=torch.max(logits,1)
             emotion_name=emotions[index.item()]
             cv2.putText(frame, emotion_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

       cv2.imshow("Emote Detector",frame)

       if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

        
if __name__ == "__main__":
    main()
