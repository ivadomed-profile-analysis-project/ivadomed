import logging
import os
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from ivadomed import image as imed_image


AXIS_DCT = {'sagittal': 0, 'coronal': 1, 'axial': 2}

# List of classification models (ie not segmentation output)
CLASSIFIER_LIST = ['resnet18', 'densenet121']

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_task(model_name):
    return "classification" if model_name in CLASSIFIER_LIST else "segmentation"


def cuda(input_var, cuda_available=True, non_blocking=False):
    """Passes input_var to GPU.

    Args:
        input_var (Tensor): either a tensor or a list of tensors.
        cuda_available (bool): If False, then return identity
        non_blocking (bool):

    Returns:
        Tensor
    """
    if cuda_available:
        if isinstance(input_var, list):
            return [t.cuda(non_blocking=non_blocking) for t in input_var]
        else:
            return input_var.cuda(non_blocking=non_blocking)
    else:
        return input_var


class HookBasedFeatureExtractor(nn.Module):
    """This function extracts feature maps from given layer. Helpful to observe where the attention of the network is
    focused.

    https://github.com/ozan-oktay/Attention-Gated-Networks/tree/a96edb72622274f6705097d70cfaa7f2bf818a5a

    Args:
        submodule (nn.Module): Trained model.
        layername (str): Name of the layer where features need to be extracted (layer of interest).
        upscale (bool): If True output is rescaled to initial size.

    Attributes:
        submodule (nn.Module): Trained model.
        layername (str):  Name of the layer where features need to be extracted (layer of interest).
        outputs_size (list): List of output sizes.
        outputs (list): List of outputs containing the features of the given layer.
        inputs (list): List of inputs.
        inputs_size (list): List of input sizes.
    """

    def __init__(self, submodule, layername, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layername = layername
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        assert (isinstance(i, tuple))
        self.inputs = [i[index].data.clone() for index in range(len(i))]
        self.inputs_size = [input.size() for input in self.inputs]

        print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, m, i, o):
        assert (isinstance(i, tuple))
        self.outputs = [o[index].data.clone() for index in range(len(o))]
        self.outputs_size = [output.size() for output in self.outputs]
        print('Output Array Size: ', self.outputs_size)

    def forward(self, x):
        target_layer = self.submodule._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)
        self.submodule(x)
        h_inp.remove()
        h_out.remove()

        return self.inputs, self.outputs


def save_feature_map(batch, layer_name, log_directory, model, test_input, slice_axis):
    """Save model feature maps.

    Args:
        batch (dict):
        layer_name (str):
        log_directory (str): Output folder.
        model (nn.Module): Network.
        test_input (Tensor):
        slice_axis (int): Indicates the axis used for the 2D slice extraction: Sagittal: 0, Coronal: 1, Axial: 2.
    """
    if not os.path.exists(os.path.join(log_directory, layer_name)):
        os.mkdir(os.path.join(log_directory, layer_name))

    # Save for subject in batch
    for i in range(batch['input'].size(0)):
        inp_fmap, out_fmap = \
            HookBasedFeatureExtractor(model, layer_name, False).forward(Variable(test_input[i][None,]))

        # Display the input image and Down_sample the input image
        orig_input_img = test_input[i][None,].cpu().numpy()
        upsampled_attention = F.interpolate(out_fmap[1],
                                            size=test_input[i][None,].size()[2:],
                                            mode='trilinear',
                                            align_corners=True).data.cpu().numpy()

        path = batch["input_metadata"][0][i]["input_filenames"]

        basename = path.split('/')[-1]
        save_directory = os.path.join(log_directory, layer_name, basename)

        # Write the attentions to a nifti image
        nib_ref = nib.load(path)
        nib_ref_can = nib.as_closest_canonical(nib_ref)
        oriented_image = imed_image.reorient_image(orig_input_img[0, 0, :, :, :], slice_axis, nib_ref, nib_ref_can)

        nib_pred = nib.Nifti1Image(oriented_image, nib_ref.affine)
        nib.save(nib_pred, save_directory)

        basename = basename.split(".")[0] + "_att.nii.gz"
        save_directory = os.path.join(log_directory, layer_name, basename)
        attention_map = imed_image.reorient_image(upsampled_attention[0, 0, :, :, :], slice_axis, nib_ref, nib_ref_can)
        nib_pred = nib.Nifti1Image(attention_map, nib_ref.affine)

        nib.save(nib_pred, save_directory)


class SliceFilter(object):
    """Filter 2D slices from dataset.

    If a sample does not meet certain conditions, it is discarded from the dataset.

    Args:
        filter_empty_mask (bool): If True, samples where all voxel labels are zeros are discarded.
        filter_empty_input (bool): If True, samples where all voxel intensities are zeros are discarded.

    Attributes:
        filter_empty_mask (bool): If True, samples where all voxel labels are zeros are discarded.
        filter_empty_input (bool): If True, samples where all voxel intensities are zeros are discarded.
    """

    def __init__(self, filter_empty_mask=True,
                 filter_empty_input=True,
                 filter_classification=False, classifier_path=None, device=None, cuda_available=None):
        self.filter_empty_mask = filter_empty_mask
        self.filter_empty_input = filter_empty_input
        self.filter_classification = filter_classification
        self.device = device
        self.cuda_available = cuda_available

        if self.filter_classification:
            if cuda_available:
                self.classifier = torch.load(classifier_path, map_location=device)
            else:
                self.classifier = torch.load(classifier_path, map_location='cpu')

    def __call__(self, sample):
        input_data, gt_data = sample['input'], sample['gt']

        if self.filter_empty_mask:
            if not np.any(gt_data):
                return False

        if self.filter_empty_input:
            # Filter set of images if one of them is empty or filled with constant value (i.e. std == 0)
            if np.any([img.std() == 0 for img in input_data]):
                return False

        if self.filter_classification:
            if not np.all([int(
                    self.classifier(cuda(torch.from_numpy(img.copy()).unsqueeze(0).unsqueeze(0), self.cuda_available)))
                           for img in input_data]):
                return False

        return True


def unstack_tensors(sample):
    """Unstack tensors.

    Args:
        sample (Tensor):

    Returns:
        list: list of Tensors.
    """
    list_tensor = []
    for i in range(sample.shape[1]):
        list_tensor.append(sample[:, i, ].unsqueeze(1))
    return list_tensor


def save_onnx_model(model, inputs, model_path):
    """Convert PyTorch model to ONNX model and save it as `model_path`.

    Args:
        model (nn.Module): PyTorch model.
        inputs (Tensor): Tensor, used to inform shape and axes.
        model_path (str): Output filename for the ONNX model.
    """
    model.eval()
    dynamic_axes = {0: 'batch', 1: 'num_channels', 2: 'height', 3: 'width', 4: 'depth'}
    if len(inputs.shape) == 4:
        del dynamic_axes[4]
    torch.onnx.export(model, inputs, model_path,
                      opset_version=11,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': dynamic_axes, 'output': dynamic_axes})


def define_device(gpu_id):
    """Define the device used for the process of interest.

    Args:
        gpu_id (int): GPU ID.

    Returns:
        Bool, device: True if cuda is available.
    """
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("Cuda is not available.")
        print("Working on {}.".format(device))
    if cuda_available:
        # Set the GPU
        gpu_number = int(gpu_id)
        torch.cuda.set_device(gpu_number)
        print("Using GPU number {}".format(gpu_number))
    return cuda_available, device


def display_selected_model_spec(params):
    """Display in terminal the selected model and its parameters.

    Args:
        params (dict): Keys are param names and values are param values.
    """
    print('\nSelected architecture: {}, with the following parameters:'.format(params["name"]))
    for k in list(params.keys()):
        if k != "name":
            print('\t{}: {}'.format(k, params[k]))


def display_selected_transfoms(params, dataset_type):
    """Display in terminal the selected transforms for a given dataset.

    Args:
        params (dict):
        dataset_type (list): e.g. ['testing'] or ['training', 'validation']
    """
    print('\nSelected transformations for the {} dataset:'.format(dataset_type))
    for k in list(params.keys()):
        print('\t{}: {}'.format(k, params[k]))


def plot_transformed_sample(before, after, list_title=[], fname_out="", cmap="jet"):
    """Utils tool to plot sample before and after transform, for debugging.

    Args:
        before (ndarray): Sample before transform.
        after (ndarray): Sample after transform.
        list_title (list of str): Sub titles of before and after, resp.
        fname_out (str): Output filename where the plot is saved if provided.
        cmap (str): Matplotlib colour map.
    """
    if len(list_title) == 0:
        list_title = ['Sample before transform', 'Sample after transform']

    plt.interactive(False)
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(before, interpolation='nearest', cmap=cmap)
    plt.title(list_title[0], fontsize=20)

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(after, interpolation='nearest', cmap=cmap)
    plt.title(list_title[1], fontsize=20)

    if fname_out:
        plt.savefig(fname_out)
    else:
        matplotlib.use('TkAgg')
        plt.show()


def _git_info(commit_env='IVADOMED_COMMIT', branch_env='IVADOMED_BRANCH'):
    """Get ivadomed version info from GIT.

    This functions retrieves the ivadomed version, commit, branch and installation type.

    Args:
        commit_env (str):
        branch_env (str):
    Returns:
        str, str, str, str: installation type, commit, branch, version.
    """
    ivadomed_commit = os.getenv(commit_env, "unknown")
    ivadomed_branch = os.getenv(branch_env, "unknown")
    if check_exe("git") and os.path.isdir(os.path.join(__ivadomed_dir__, ".git")):
        ivadomed_commit = __get_commit() or ivadomed_commit
        ivadomed_branch = __get_branch() or ivadomed_branch

    if ivadomed_commit != 'unknown':
        install_type = 'git'
    else:
        install_type = 'package'

    path_version = os.path.join(__ivadomed_dir__, 'ivadomed', 'version.txt')
    with open(path_version) as f:
        version_ivadomed = f.read().strip()

    return install_type, ivadomed_commit, ivadomed_branch, version_ivadomed


def check_exe(name):
    """Ensure that a program exists.

    Args:
        name (str): Name or path to program.
    Returns:
        str or None: path of the program or None
    """

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(name)
    if fpath and is_exe(name):
        return fpath
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, name)
            if is_exe(exe_file):
                return exe_file

    return None


def __get_commit(path_to_git_folder=None):
    """Get GIT ivadomed commit.

    Args:
        path_to_git_folder (str): Path to GIT folder.
    Returns:
        str: git commit ID, with trailing '*' if modified.
    """
    if path_to_git_folder is None:
        path_to_git_folder = __ivadomed_dir__
    else:
        path_to_git_folder = os.path.abspath(os.path.expanduser(path_to_git_folder))

    p = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=path_to_git_folder)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        commit = output.decode().strip()
    else:
        commit = "?!?"

    p = subprocess.Popen(["git", "status", "--porcelain"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=path_to_git_folder)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        unclean = True
        for line in output.decode().strip().splitlines():
            line = line.rstrip()
            if line.startswith("??"):  # ignore ignored files, they can't hurt
                continue
            break
        else:
            unclean = False
        if unclean:
            commit += "*"

    return commit


def __get_branch():
    """Get ivadomed branch.

    Args:

    Returns:
        str: ivadomed branch.
    """
    p = subprocess.Popen(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, cwd=__ivadomed_dir__)
    output, _ = p.communicate()
    status = p.returncode

    if status == 0:
        return output.decode().strip()


def _version_string():
    install_type, ivadomed_commit, ivadomed_branch, version_ivadomed = _git_info()
    if install_type == "package":
        return version_ivadomed
    else:
        return "{install_type}-{ivadomed_branch}-{ivadomed_commit}".format(**locals())


__ivadomed_dir__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
__version__ = _version_string()


def init_ivadomed():
    """Initialize the ivadomed for typical terminal usage."""
    # Display ivadomed version
    logger.info('\nivadomed ({})\n'.format(__version__))
