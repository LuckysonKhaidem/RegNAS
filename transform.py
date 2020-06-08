from scipy.ndimage import affine_transform
import numpy as np
import nibabel as nib
from nilearn.plotting import plot_img
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.transform import Rotation
from models import SpatialTransformer
import torch

def elastic_transform(image,alpha, sigma):
    shape = image.shape
    random_state = np.random.RandomState(None)
    dx = gaussian_filter((random_state.rand(*shape)*2 - 1),sigma)*alpha
    dy = gaussian_filter((random_state.rand(*shape)*2 - 1),sigma)*alpha
    dz = np.zeros_like(dx)
    x,y,z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy,(-1,1)),np.reshape(x + dx,(-1,1)), np.reshape(z,(-1,1))
    temp =  map_coordinates(image,indices)
    tr =  temp.reshape(shape)
    
    displacement  = np.concatenate((dx,dy,dz),axis=0)

    displacement = displacement.reshape((3,) + shape)
    return tr, displacement


def get_affine_transformation_matrix():
    a = np.random.uniform(0,np.pi/6, size = (3,)) #rotation angles
    c = np.random.uniform(0.75,1.25, size = (3,)) #scaling factors
    l = np.random.uniform(-0.02,0.02,size = (3,)) #translation factors

    c_m = np.eye(4) # scale matrix
    c_m[0][0] = c[0]; c_m[1][1] = c[1]; c_m[2][2] = c[2]
    l_m = np.eye(4) # translation matrix
    l_m[0][3] = l[0]; l_m[1][3] = l[1]; l_m[2][3] = l[2]
    # l_m[0][3] = 10
    r_m = np.eye(4)
    r_m[:3,:3] = Rotation.from_rotvec(a).as_dcm() # rotation matrix
    M = r_m @ c_m @ l_m
    # M = l_m
 
    return M

def get_affine_displacement(shape, M):
    displacement = np.zeros((3,) + shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                vec = np.array([i,j,k,1]).reshape((4,1))
                new_vec = M @ vec
                d = new_vec - vec
                displacement[0][i][j][k] = d[0]
                displacement[1][i][j][k] = d[1]
                displacement[2][i][j][k] = d[2]

    return displacement


def random_transform(image):
    M = get_affine_transformation_matrix()
    tr_image = affine_transform(image,M, mode = "constant")
    affine_displacement  = get_affine_displacement(image.shape,M)
    alpha = np.random.uniform(low = 0,high = 1000)
    sigma = np.random.uniform(low = 11,high = 13 )
    tr_image, elastic_displacement = elastic_transform(tr_image, alpha, sigma)
    displacement = affine_displacement + elastic_displacement
    return tr_image,displacement

def run_stn(image, displacement):
    stn = SpatialTransformer(image.shape)
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    displacement = torch.from_numpy(displacement).unsqueeze(0).float()
    tr_image = stn(image,displacement)
    return tr_image.detach().numpy()[0][0]

def test():
    x = nib.load("/Users/luckysonkhaidem/school-work/research/Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz")
    image = x.get_fdata()
    transformed_image,displacement = random_transform(image)
    # test_img = run_stn(image,displacement)
    # new_image = nib.Nifti1Image(transformed_image, affine=np.eye(4))
    # new_image.to_filename("deformed.nii")
    # new_image = nib.Nifti1Image(test_img, affine=np.eye(4))
    # new_image.to_filename("deformed_1.nii")
    # plot_img("Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz")
    # plot_img("deformed.nii")
    # plot_img("deformed_1.nii")
    # plt.show()

test()