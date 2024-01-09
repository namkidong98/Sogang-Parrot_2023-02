## 프로젝트명 : CUB-200-2011 Dataset Classification

### 목표
CUB-200-2011 Dataset를 바탕으로 200 종의 새 이미지가 주어졌을 때 해당 새가 어떤 종인지 분류한다

<img width="800" src="https://github.com/namkidong98/Sogang-Parrot_2023-02/assets/113520117/48a37d43-8651-45fd-a8c9-5c1f0a2a3641">

<br>

### 프로젝트 기간
2023.12.22 ~ 2024.01.05

<br>

## 프로젝트 코드 요약

train dataset 공유 링크   
https://drive.google.com/file/d/1sYdfrvVbWYK2gwaItAY67a2lNYieZ12e/view?usp=drive_link

### 1. Augmentation_Code.ipynb
1. data_transforms을 이용하여 원본 데이터셋을 로드한다
2. albumentation 라이브러리를 바탕으로 augmentation 함수를 만든다
```python
def augmentation(data_dir, num_class, class_names, new_per_origin):
  # new_per_origin: 각 원본 이미지당 생성할 이미지의 개수
  total = 0
  for i in range(num_class):
    file_path = data_dir + '/train/' + class_names[i] # 각 클래스별 폴더 경로 지정
    file_names = os.listdir(file_path) # 각 클래스별 폴더 내의 파일들의 이름
    total_origin_image_num = len(file_names)

    new_file_path = data_dir + '/augmented/' + class_names[i]
    if not os.path.exists(new_file_path):
      os.makedirs(new_file_path)

    print("current_directory:", class_names[i]) # 진행 상황을 확인

    augment_cnt = 0
    images_to_use = total_origin_image_num  # 해당 폴더에서 Augmentation에 사용할 이미지 개수
    for idx in range(images_to_use):
      file_name = file_names[idx]
      origin_image_path = file_path + '/' + file_name # 해당 파일의 경로를 설정
      origin_image = plt.imread(origin_image_path) # 이미지로 불러오기
      for _ in range(new_per_origin):
        # albumentations_transform으로 변환된 이미지 생성
        transformed = albumentations_transform(image=origin_image)
        transformed_image = transformed['image']
        transformed_image = Image.fromarray(np.uint8(transformed_image))

        # transformed_image.save(file_path + '/augmented_' + str(augment_cnt) + '.jpg') # 기존 파일에 저장한 경우
        transformed_image.save(new_file_path + '/augmented_' + str(augment_cnt) + '.jpg') # 새로 분리해서 저장
        augment_cnt += 1
    total += augment_cnt
  print("Augmented Data:", total)
```
3. augmented dataset을 삭제하여 원본 dataset으로 돌리는 delete_augmented 함수를 만든다
```python
def delete_augmented(data_dir, num_class, class_names):
  prefixes = ['augmented_']
  for i in range(num_class):
    # file_path = data_dir + '/train/' + class_names[i] # 기존 파일에 생성한 경우
    file_path = data_dir + '/augmented/' + class_names[i]
    file_names = os.listdir(file_path) # 각 클래스별 폴더 내의 파일들의 이름

    print("current_directory:", class_names[i]) # 진행 상황을 확인
    for file_name in file_names:
      for prefix in prefixes:
        if file_name.startswith(prefix): # 해당 prefix로 시작하는 파일이면
          file_path_to_delete = os.path.join(file_path, file_name)
          os.remove(file_path_to_delete) # 삭제하기
```
4. augmentation 함수를 실행시킨 후 전체 데이터셋의 개수를 확인한다
```python
# 사용법 : 함수 이름(경로, 클래스 개수, 클래스 이름, {각 이미지당 증강할 이미지 개수})
delete_augmented('/content/drive/MyDrive/Parrot DL Project', 200, class_names)
augmentation('/content/drive/MyDrive/Parrot DL Project', 200, class_names, 1) # 200개의 클래스에 대해 각 이미지당 1개씩 augmented image 생성
```

<br>

### 2. EfficientNet_TL.ipynb
1. 원본 데이터셋을 로드하고 7:3의 비율로 train, valid로 구분한다
2. augmented dataset을 로드하고 train에 Concat하여 전체 train_dataset을 만든다
```python
# 데이터셋을 train과 valid로 나누기
origin_train_size = int(0.7 * len(full_dataset))
valid_size = len(full_dataset) - origin_train_size
origin_train_dataset, valid_dataset = random_split(full_dataset, [origin_train_size, valid_size])

# augmented를 origin_train에 더해서 train_dataset을 만든다
train_dataset = ConcatDataset([origin_train_dataset, augmented_dataset])

# 데이터 로더
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
```

<br>

3. 원본 이미지와 증강된 이미지 시각화 비교

<img width='700' src='https://github.com/namkidong98/Sogang-Parrot_2023-02/assets/113520117/cfc1b290-b2aa-4e9a-aec5-6a6d0f4b5288'>
<img width='700' src='https://github.com/namkidong98/Sogang-Parrot_2023-02/assets/113520117/90ec09fc-9ca6-48d5-a11a-2f8eb061d49d'>

<br> 

4. pretrain된 EfficientNet을 base model로 EfficientNetCustom 모델을 생성하고 장치에 올린다
```python
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

class EfficientNetCustom(nn.Module):
    def __init__(self, freeze=False):
        super(EfficientNetCustom, self).__init__()
        self.pretrained_efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

        if freeze:
            for param in self.pretrained_efficientnet.parameters():
                param.requires_grad = False

        # 예측에 사용되는 EfficientNet의 마지막 계층들을 새로이 정의
        self.pretrained_efficientnet._fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 200)
        )

    def forward(self, input):
        return self.pretrained_efficientnet(input)

# 모델 생성 및 장치로 이동
model = EfficientNetCustom(freeze=False).to(device)
```

5. Loss Fuction, Optimizer, Scheduler를 설정하고 학습을 진행한다
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.2) # 손실함수 설정
optimizer = optim.Adam(model.parameters(), lr=0.00005) # Optimizer 설정
step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75) # Scheduler 설정
```

6. Train과 Valid 각각에 대한 Loss, Accuracy Graph를 그린다

<img width='700' src="https://github.com/namkidong98/Sogang-Parrot_2023-02/assets/113520117/46f1c1cb-b44b-4213-9d59-0ec8411023fb">

<br>

7. 모델을 저장한다(model.pth)

### 3. model.pth
2번의 코드를 바탕으로 학습을 완료한 모델을 저장한 파일

<br> 

### 4. try : 최종 파일 이전의 시도와 정확도를 표시
1. 65 : train data를 로드할 때 랜덤하게 변형하는 방식을 사용, EfficientNet을 사용, lr=0.0001을 사용
2. 75 : train data를 그대로 로드, resnet을 사용, lr=0.00001로 감소 시킴으로써 성능의 향상을 확인
3. 80 : train data를 그대로 로드, EfficientNet을 사용, lr=0.00005 --> EfficientNet과 lr 조절로 성능 향상
4. 88 : Data_Augmentation 코드로 데이터셋 크기를 증가시킴, but valid에도 augmented가 포함되는 문제 발생

<br>

## 생각해볼 점 & 배운 점
- Data Augmentation을 많이 한다고 성능이 높아지는 것 같지는 않다. 원본 데이터의 50%, 100%, 300%를 시도했는데 큰 성능 차이는 없었던 것 같다

- Data Augmentation을 할 때 valid, test는 원본 이미지로 하고 train에만 augmented data가 포함될 수 있도록 주의해야 한다

- pretrained model을 사용한 Transfer learning에서는 평소보다 **매우 낮은 learning rate**를 주어야 한다
