import torchvision.transforms as transforms

def makeDefaultTransforms(img_crop_size=224, img_resize=256):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(img_resize),
            transforms.RandomResizedCrop(img_crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(img_resize),
            transforms.CenterCrop(img_crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms

def makeNewTransforms():
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize(512),
            transforms.RandomResizedCrop(448), # 随机选取原输入图片中的448×448的部分
            transforms.RandomHorizontalFlip(), # 随机旋转一定角度
            transforms.ToTensor(),             # 转化为tensor矩阵
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize(512),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    return data_transform


def makeAggresiveTransforms(img_crop_size=512, img_resize=448, transform_type='all', persp_distortion_scale=0.25, rotation_range=(-10.0,10.0)):
    
    test = transforms.Compose([
                transforms.Resize(img_resize),
                transforms.CenterCrop(img_crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    if transform_type == 'perspective':
        train = transforms.Compose([
                transforms.Resize(img_resize),
                transforms.RandomResizedCrop(img_crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomPerspective(distortion_scale=persp_distortion_scale, p=0.5, interpolation=3),# fill=0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    elif transform_type == 'rotation':
        train = transforms.Compose([
                transforms.Resize(img_resize),
                transforms.RandomResizedCrop(img_crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(rotation_range, resample=False, expand=False, center=None, fill=None),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
            
    elif transform_type == 'all':
        train = transforms.Compose([
                transforms.Resize(img_resize),
                transforms.RandomResizedCrop(img_crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomPerspective(distortion_scale=persp_distortion_scale, p=0.5, interpolation=3), # fill=0),
                transforms.RandomRotation(rotation_range, expand=False, center=None, fill=None),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    return {'train': train,'test': test} 