from scipy.ndimage import affine_transform
import numpy as np
import nibabel as nib
from nilearn.plotting import plot_img
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.transform import Rotation

def elastic_transform(image,alpha, sigma):
    shape = image.shape
    random_state = np.random.RandomState(None)
    dx = gaussian_filter((random_state.rand(*shape)*2 - 1),sigma)*alpha
    dy = gaussian_filter((random_state.rand(*shape)*2 - 1),sigma)*alpha
    dz = np.zeros_like(dx)
    x,y,z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy,(-1,1)),np.reshape(x + dx,(-1,1)), np.reshape(z,(-1,1))
    return map_coordinates(image,indices).reshape(shape)

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

def random_transform(image):
    M = get_affine_transformation_matrix()
    image = affine_transform(image,M, mode = "constant")
    alpha = np.random.uniform(low = 0,high = 1000)
    sigma = np.random.uniform(low = 11,high = 13 )
    image = elastic_transform(image, alpha, sigma)
    return image

def test():
    x = nib.load("/Users/luckysonkhaidem/school-work/research/Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz")
    image = x.get_fdata()
    transformed_image = random_transform(image)
    new_image = nib.Nifti1Image(transformed_image, affine=np.eye(4))
    new_image.to_filename("deformed.nii")
    plot_img("Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz")
    plot_img("deformed.nii")
    plt.show()

test()