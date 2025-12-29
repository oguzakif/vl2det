from typing import Any
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import random
import scipy.io
import PIL


class RandomCropAndResize(object):
    def __init__(self, min_crop_size, max_crop_size, final_size):
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.final_size = final_size

    def __call__(self, img):
        # Set random crop size
        crop_size = random.randint(self.min_crop_size, self.max_crop_size)

        # RandomCrop 
        img = transforms.RandomCrop(crop_size)(img)

        # Final resize
        img = transforms.Resize((self.final_size, self.final_size))(img)

        return img


class KDDataset(Dataset):
    def __init__(self, data_name, data_root, attributes, split='train'):
        self.root_dir = data_root
        self.split = split
        self.transform = self.get_transform(split)
        self.dataloader = self.get_dataloader(data_name)
        self.attributes = attributes #prompt_tmpl, class_num, classes

    def get_transform(self, split):
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])       
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((256, 256)),
                #transforms.Pad(10),
                #transforms.Pad(20),
                transforms.RandomCrop((224, 224)),
                #transforms.CenterCrop((224, 224)),
                RandomCropAndResize(min_crop_size=192, max_crop_size=224, final_size=224),
                transforms.RandomRotation(60, interpolation = PIL.Image.BILINEAR),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.00001, 10)),
                transforms.ColorJitter(brightness=.3, contrast=.1, saturation=.1, hue=.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                #transforms.RandomErasing(p=0.5, scale=(0.3, 0.5), ratio=(0.3, 3.3),),
                ])
        else:
            return transforms.Compose([
                        # transforms.Scale(256),
                        # transforms.CenterCrop(224),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        normalize,
                    ])

    def get_dataloader(self, data_name):
        if data_name == '0_CUB_200_2011':
            return CUB200Dataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '1_FGVC_AIRCRAFT':
            return AircraftDataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '2_NABirds':
            return NABirdsDataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '3_DTD':
            return DTDDataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '4_OxfordIIITPet':
            return OxfordIIPetDataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '5_StanfordDogs':
            return StanfordDogsDataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '6_StanfordCars':
            return StanfordCarsDataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '7_CALTECH101':
            return Caltech101Dataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '8_CALTECH256':
            return Caltech256Dataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '9_GTSRB':
            return GTSRBTestDataset(self.root_dir, split=self.split, transform=self.transform)

    def __len__(self):
        return len(self.dataloader)
        
    def __getitem__(self, idx):
        return self.dataloader[idx]
    



class GTSRBTestDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        if split == 'train':
            csv_file = os.path.join(root_dir, 'Train.csv')
        else:
            csv_file = os.path.join(root_dir, 'Test.csv')

        self.data_frame = pd.read_csv(csv_file, skiprows=1)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.root_dir + '/' + self.data_frame.iloc[idx, 7].strip()
        image = Image.open(img_path)
        label = int(self.data_frame.iloc[idx, 6])

        if self.transform:
            image = self.transform(image)

        return image, label


class CUB200Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):

        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.split = split
        
        splits = []
        with open(os.path.join(root_dir, 'train_test_split.txt'), 'r') as f:
            for line in f:
                _, is_train = line.strip().split(' ')
                splits.append(int(is_train))
        
        with open(os.path.join(root_dir, 'images.txt'), 'r') as f:
            for idx, line in enumerate(f):
                img_id, img_path = line.strip().split(' ')
                if (split == "train" and splits[idx] == 1) or (split == "test" and splits[idx] == 0):
                    self.image_paths.append(os.path.join(root_dir, 'images', img_path))
                
        with open(os.path.join(root_dir, 'image_class_labels.txt'), 'r') as f:
            for idx, line in enumerate(f):
                _, label = line.strip().split(' ')
                if (split == "train" and splits[idx] == 1) or (split == "test" and splits[idx] == 0):
                    self.labels.append(int(label))
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        label = self.labels[idx] - 1 # label은 0부터 시작
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class DTDDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): DTD root directory
            split (str): "train" 또는 "test".
            transform (callable, optional): transform.
        """
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.split = split
        img_path = os.path.join(root_dir, 'images')
        
        splits = {}

        split_file = os.path.join(root_dir, 'train_test_split.txt')
        if not os.path.isfile(split_file):
            with open(split_file, 'w') as f:
                for category in os.listdir(img_path):
                    category_path = os.path.join(img_path, category)
                    if os.path.isdir(category_path):
                        for img_name in os.listdir(category_path):
                            if img_name.endswith('.jpg'):
                                if random.random() < 0.8:
                                    f.write(f"{img_name} train\n")
                                else:
                                    f.write(f"{img_name} test\n")

        with open(split_file, 'r') as f:
            for line in f:
                img_name, s = line.strip().split()
                splits[img_name] = s

        for label, category in enumerate(os.listdir(img_path)):
            category_path = os.path.join(img_path, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    if img_name.endswith('.jpg') and splits[img_name] == split:
                        self.image_paths.append(os.path.join(category_path, img_name))
                        self.labels.append(label)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class AircraftDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):

        assert split in ["train", "test", "val"], "split should be 'train', 'test', or 'val'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.split = split
        
        # 레이블 매핑
        label_map = {}
        with open(os.path.join(root_dir, 'data', 'variants.txt'), 'r') as f:
            for idx, line in enumerate(f):
                class_name = line.strip()
                label_map[class_name] = idx

        # 분할 파일 읽기
        with open(os.path.join(root_dir, 'data', f'images_variant_{split}.txt'), 'r') as f:
            for line in f:
                line_split = line.strip().split(' ')
                if len(line_split) == 2:
                    img_name, class_name = line.strip().split()
                else:
                    img_name = line_split[0]
                    class_name = ' '.join(line_split[1:])

                self.image_paths.append(os.path.join(root_dir, 'data', 'images', img_name + '.jpg'))
                self.labels.append(label_map[class_name])
                
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class NABirdsDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        
        with open(os.path.join(root_dir, f'images.txt'), 'r') as f:
            for line in f:
                img_id, img_path = line.strip().split()
                self.image_paths.append(os.path.join(root_dir, 'images', img_path))

        with open(os.path.join(root_dir, f'image_class_labels.txt'), 'r') as f:
            for line in f:
                _, label = line.strip().split()
                self.labels.append(int(label) - 1)  
        
        split_list = []
        with open(os.path.join(root_dir, f'train_test_split.txt'), 'r') as f:
            for line in f:
                _, is_train = line.strip().split()
                split_list.append(is_train)

        if split == "train":
            self.image_paths = [self.image_paths[i] for i, s in enumerate(split_list) if s == '1']
            self.labels = [self.labels[i] for i, s in enumerate(split_list) if s == '1']
        else:
            self.image_paths = [self.image_paths[i] for i, s in enumerate(split_list) if s == '0']
            self.labels = [self.labels[i] for i, s in enumerate(split_list) if s == '0']

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class OxfordIIPetDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Oxford-IIIT Pet 데이터셋의 디렉터리 경로.
            split (str): "train" 또는 "test".
            transform (callable, optional): 적용할 변환(Optional).
        """
        assert split in ["train", "test"], "split should be either 'train' or 'test'"

        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}

        if split == "train":
            split_file = os.path.join(root_dir, "annotations", "trainval.txt")
        else:
            split_file = os.path.join(root_dir, "annotations", "test.txt")

        # split 파일에서 이미지 경로와 레이블 정보 읽기
        with open(split_file, 'r') as file:
            for line in file:
                img_name, label_id, _, _ = line.strip().split()
                img_path = os.path.join(root_dir, "images", img_name + ".jpg")
                if img_name.split('_')[0] not in self.label_to_idx:
                    self.label_to_idx[img_name.split('_')[0]] = len(self.label_to_idx)
                self.image_paths.append(img_path)
                self.labels.append(self.label_to_idx[img_name.split('_')[0]])
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CalTech256Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): CalTech256 데이터셋의 디렉터리 경로.
            split (str): "train" 또는 "test".
            transform (callable, optional): 적용할 변환(Optional).
        """
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        
        split_file = os.path.join(root_dir, "train_test_split.txt")
        
        # 파일이 없으면 새로 생성
        if not os.path.exists(split_file):
            with open(split_file, 'w') as file:
                categories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith("BACKGROUND")])
                for category in categories:
                    category_path = os.path.join(root_dir, category)
                    images_in_category = sorted([img for img in os.listdir(category_path) if img.endswith('.jpg')])
                    random.shuffle(images_in_category)
                    
                    split_idx = int(len(images_in_category) * 0.8)
                    train_images = images_in_category[:split_idx]
                    test_images = images_in_category[split_idx:]
                    
                    for img in train_images:
                        file.write(f"train {os.path.join(category, img)} {category}\n")
                    for img in test_images:
                        file.write(f"test {os.path.join(category, img)} {category}\n")
        
        # train_test_split.txt에서 이미지 경로와 레이블 정보 읽기
        with open(split_file, 'r') as file:
            for line in file:
                mode, img_path, category = line.strip().split()
                if mode == split:
                    if category not in self.label_to_idx:
                        self.label_to_idx[category] = len(self.label_to_idx)
                    self.image_paths.append(os.path.join(root_dir, img_path))
                    self.labels.append(self.label_to_idx[category])
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CalTech101Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): CalTech101 데이터셋의 디렉터리 경로.
            split (str): "train" 또는 "test".
            transform (callable, optional): 적용할 변환(Optional).
        """
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        
        split_file = os.path.join(root_dir, "train_test_split.txt")
        
        # 파일이 없으면 새로 생성
        if not os.path.exists(split_file):
            with open(split_file, 'w') as file:
                categories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
                for category in categories:
                    category_path = os.path.join(root_dir, category)
                    images_in_category = sorted([img for img in os.listdir(category_path) if img.endswith('.jpg')])
                    random.shuffle(images_in_category)
                    
                    split_idx = int(len(images_in_category) * 0.8)
                    train_images = images_in_category[:split_idx]
                    test_images = images_in_category[split_idx:]
                    
                    for img in train_images:
                        file.write(f"train {os.path.join(category, img)} {category}\n")
                    for img in test_images:
                        file.write(f"test {os.path.join(category, img)} {category}\n")
        
        # train_test_split.txt에서 이미지 경로와 레이블 정보 읽기
        with open(split_file, 'r') as file:
            for line in file:
                mode, img_path, category = line.strip().split()
                if mode == split:
                    if category not in self.label_to_idx:
                        self.label_to_idx[category] = len(self.label_to_idx)
                    self.image_paths.append(os.path.join(root_dir, img_path))
                    self.labels.append(self.label_to_idx[category])
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class StanfordCarsDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Stanford Cars 데이터셋의 디렉터리 경로.
            split (str): "train" 또는 "test".
            transform (callable, optional): 적용할 변환(Optional).
        """
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.df = pd.read_csv(os.path.join(root_dir, f"{split}.csv"))
        self.img_dir = os.path.join(root_dir, "car_ims")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 1])  # 2nd column for image filename
        image = Image.open(img_name).convert('RGB')
        label = int(self.df.iloc[idx, 0])  # 1st column for label
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

        
class StanfordDogsDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Stanford Dogs 데이터셋의 디렉터리 경로.
            split (str): "train" 또는 "test".
            transform (callable, optional): 적용할 변환(Optional).
        """
        assert split in ["train", "test"], "split should be either 'train' or 'test'"

        self.split_mat = scipy.io.loadmat(os.path.join(root_dir, f"{split}_list.mat"))
        self.file_list = [item[0][0] for item in self.split_mat['file_list']]
        self.labels = [item[0] - 1 for item in self.split_mat['labels']]  # Convert 1-based index to 0-based
        
        self.img_dir = os.path.join(root_dir, "Images")
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.file_list[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class Caltech101Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Caltech 256 데이터셋의 디렉터리 경로.
            split (str): "train" 또는 "test".
            transform (callable, optional): 적용할 변환(Optional).
        """
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.imgs_path = os.path.join(root_dir, '101_ObjectCategories')
        self.image_paths = []
        self.labels = []
        self.class_names = sorted([d for d in os.listdir(self.imgs_path) if os.path.isdir(os.path.join(self.imgs_path, d))])
        self.transform = transform
        self.split = split
        
        train_indices = []
        test_indices = []
        
        # Load images and labels by class and split within each class
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.imgs_path, class_name)
            class_image_paths = [os.path.join(class_dir, img_name) for img_name in os.listdir(class_dir)]
            class_labels = [class_idx] * len(class_image_paths)
            
            class_train_size = int(0.8 * len(class_image_paths))
            class_train_indices = list(range(len(self.image_paths), len(self.image_paths) + class_train_size))
            class_test_indices = list(range(len(self.image_paths) + class_train_size, len(self.image_paths) + len(class_image_paths)))
            
            self.image_paths.extend(class_image_paths)
            self.labels.extend(class_labels)
            
            train_indices.extend(class_train_indices)
            test_indices.extend(class_test_indices)
        
        split_file = os.path.join(root_dir, "train_test_split.txt")
        if not os.path.exists(split_file):
            with open(split_file, 'w') as f:
                for idx in train_indices:
                    f.write(f"train,{self.image_paths[idx]}\n")
                for idx in test_indices:
                    f.write(f"test,{self.image_paths[idx]}\n")
        
        split_indices = train_indices if split == "train" else test_indices
        self.image_paths = [self.image_paths[i] for i in split_indices]
        self.labels = [self.labels[i] for i in split_indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class Caltech256Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Caltech 256 데이터셋의 디렉터리 경로.
            split (str): "train" 또는 "test".
            transform (callable, optional): 적용할 변환(Optional).
        """
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.imgs_path = os.path.join(root_dir, '256_ObjectCategories')
        self.image_paths = []
        self.labels = []
        self.class_names = sorted([d for d in os.listdir(self.imgs_path) if os.path.isdir(os.path.join(self.imgs_path, d))])
        self.transform = transform
        self.split = split
        
        train_indices = []
        test_indices = []
        
        # Load images and labels by class and split within each class
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.imgs_path, class_name)
            class_image_paths = [os.path.join(class_dir, img_name) for img_name in os.listdir(class_dir)]
            class_labels = [class_idx] * len(class_image_paths)
            
            class_train_size = int(0.8 * len(class_image_paths))
            class_train_indices = list(range(len(self.image_paths), len(self.image_paths) + class_train_size))
            class_test_indices = list(range(len(self.image_paths) + class_train_size, len(self.image_paths) + len(class_image_paths)))
            
            self.image_paths.extend(class_image_paths)
            self.labels.extend(class_labels)
            
            train_indices.extend(class_train_indices)
            test_indices.extend(class_test_indices)
        
        split_file = os.path.join(root_dir, "train_test_split.txt")
        if not os.path.exists(split_file):
            with open(split_file, 'w') as f:
                for idx in train_indices:
                    f.write(f"train,{self.image_paths[idx]}\n")
                for idx in test_indices:
                    f.write(f"test,{self.image_paths[idx]}\n")
        
        split_indices = train_indices if split == "train" else test_indices
        self.image_paths = [self.image_paths[i] for i in split_indices]
        self.labels = [self.labels[i] for i in split_indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label