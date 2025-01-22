import os

import cv2
import lpips
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# Initialize LPIPS with AlexNet
lpips_model = lpips.LPIPS(net='alex')


def show_image(image_np):
    """
    Show an image from a numpy array.

    Parameters:
    - image_np: Numpy array of the image to display.

    The input image should be in RGB format. If the image is in BGR format,
    convert it to RGB before passing to this function.
    """
    # Check if the image is in RGBA format and convert to RGB if necessary
    if image_np.shape[2] == 4:
        # Convert RGBA to RGB
        image_np = image_np[:, :, :3]

    plt.imshow(image_np)
    plt.axis('off')  # Hide axis labels and ticks
    plt.show()


def calculate_metrics(image_path_1, image_path_2, alpha_path, gamma_corr=False, mask_pred=False):
    # Read and convert images
    image1 = cv2.cvtColor(cv2.imread(image_path_1), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread(image_path_2), cv2.COLOR_BGR2RGB)

    image1 = image1 / 255.
    image2 = image2 / 255.

    if alpha_path is not None:
        alpha = cv2.imread(alpha_path, cv2.IMREAD_UNCHANGED)
        assert alpha.shape[2] == 4
        alpha = alpha[:, :, 3:]
        alpha = alpha / 255.
        image2 = image2 * alpha + 1.0 * (1 - alpha)
        # the prediction should not be masked (because they are already masked). Except in PIE-Net and Ordinal(they cannot handle background. )
        if mask_pred:
            image1 = image1 * alpha + 1.0 * (1 - alpha)
        # # gamma correction for shading
        if gamma_corr:
            image2 = np.power(image2, 1/2.2)

    H1, W1 = tuple(image1.shape[:2])
    H2, W2 = tuple(image2.shape[:2])
    if not (H1 == H2 and W1 == W2):
        image2 = cv2.resize(image2, (W1, H1))

    # Calculate PSNR and SSIM
    psnr = compare_psnr(image1, image2)
    ssim = compare_ssim(image1, image2, multichannel=True, channel_axis=-1, data_range=1.0)

    # Prepare images for LPIPS
    tensor1 = torch.tensor(image1.transpose(2, 0, 1)).float().unsqueeze(0)
    tensor2 = torch.tensor(image2.transpose(2, 0, 1)).float().unsqueeze(0)

    # Use GPU if available
    if torch.cuda.is_available():
        tensor1 = tensor1.cuda()
        tensor2 = tensor2.cuda()
        lpips_model.cuda()

    # Calculate LPIPS, to [-1.0, 1.0]
    # lpips_value = lpips_model(tensor1 * 2.0 - 1.0, tensor2 * 2.0 - 1.0).item()
    lpips_value = lpips_model(tensor1, tensor2, normalize=True).item()

    # Calculate MSE
    # mse = np.mean((image1 - image2) ** 2)
    mse = mean_squared_error(image1, image2)

    return psnr, ssim, lpips_value, mse


def compare_image_lists(folder1_images, folder2_images, folder_alpha, gamma_corr=False, mask_pred=False):
    # Initialize metric accumulators
    total_psnr, total_ssim, total_lpips, total_mse = 0, 0, 0, 0

    # Ensure both lists have the same number of images
    assert len(folder1_images) == len(folder2_images), "Image lists must be of the same length."
    assert len(folder2_images) == len(folder_alpha)

    # Iterate and calculate metrics for each image pair
    for img1, img2, alpha in zip(folder1_images, folder2_images, folder_alpha):
        psnr, ssim, lpips_val, mse = calculate_metrics(img1, img2, alpha, gamma_corr=gamma_corr, mask_pred=mask_pred)
        total_psnr += psnr
        total_ssim += ssim
        total_lpips += lpips_val
        total_mse += mse

    # Calculate average metrics
    num_images = len(folder1_images)
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    avg_lpips = total_lpips / num_images
    avg_mse = total_mse / num_images

    return avg_psnr, avg_ssim, avg_lpips, avg_mse


def get_output_ours(path, component, dataset_type=None):
    ours_output_key = {
        'Ref': 'o_r',
        'Sha': 'o_s',
        'Img': 'rgb'
    }
    if dataset_type == 'syn_intrinsic':
        images = [os.path.join(path, f'{i}_{ours_output_key[component]}_map.png') for i in range(100)]
    elif dataset_type == 'rene':
        image_num = len([x for x in os.listdir(path) if 'rgb_map.png' in x])
        images = [os.path.join(path, f'{i}_{ours_output_key[component]}_map.png') for i in range(image_num)]
    return images


def get_output_IntrinsicNeRF(path, component, dataset_type=None):
    ours_output_key = {
        'Ref': 'a',
        'Sha': 's',
        'Img': ''

    }
    if dataset_type == 'syn_intrinsic':
        images = [os.path.join(path, f'{ours_output_key[component]}{str(i).zfill(3)}.png') for i in range(100)]
    else:
        raise NotImplementedError
    return images

def get_GT(path, component, dataset_type=None):
    if dataset_type == 'syn_intrinsic':
        alphas = [os.path.join(path, f'{str(i).zfill(3)}_Img.png') for i in range(100)]
        images = [os.path.join(path, f'{str(i).zfill(3)}_{component}.png') for i in range(100)]
    else:
        raise NotImplementedError
    return images, alphas

def get_GT_rene(data_root, meta_fname, light_index=None):
    with open(meta_fname) as file:
        meta = json.load(file)
    if light_index is not None:
        gt_images_list = [os.path.join(data_root, x["file_path"]) for x in meta['frames'] if x['light_index'] in light_index]
    else:
        gt_images_list = [os.path.join(data_root, x["file_path"]) for x in meta['frames']]
    return gt_images_list

def get_output_NRHints(path):
    images = [os.path.join(path, f'rgb_{str(i).zfill(3)}.png') for i in range(100)]
    return images

def get_output_pienet_ordinal(path, component):
    output_key = {
        'Ref': 'ref',
        'Sha': 'sha',
    }
    images = [os.path.join(path, f'{str(i).zfill(3)}_{output_key[component]}.png') for i in range(100)]
    return images


def get_output_InvRender(path, component):
    output_key = {
        'Ref': 'albedo',
        'Img': 'sg_rgb_bg',
    }
    images = [os.path.join(path, f'{output_key[component]}_{str(i)}.png') for i in range(100)]
    return images


def get_output_TensoIR(path, component):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and f.startswith('test_')]
    folders.sort()
    img_path = os.path.join(path, folders[-1])
    if component == 'Ref':
        images = [os.path.join(img_path, f'imgs_test_all/brdf/{str(i).zfill(3)}_albedo.png') for i in range(100)]
    elif component == 'Img':
        images = [os.path.join(img_path, f'imgs_test_all/nvs_with_radiance_field/{str(i).zfill(3)}_rgb_prediction.png') for i in range(100)]
    else:
        raise NotImplementedError
    return images


if __name__ == "__main__":
    # for exp in ['syn_hotdog_b_2_4', 'syn_hotdog_b_2_5', 'syn_hotdog_b_2_6']:
    #     dataset_type = 'syn_intrinsic'
    #     path_prediction = f'/ghome/yyang/PycharmProjects/neuralangelo/logs/{exp}/output_image'
    #     path_gt = '/ghome/yyang/dataset/Relighting_intrinsic/hotdog_intrinsic/test'
    #     results = ''
    #     for component in ['Ref', 'Sha', 'Img']:
    #         images_prediction = get_output_ours(path_prediction, component, dataset_type=dataset_type)
    #         images_gt, alphas_gt = get_GT(path_gt, component, dataset_type=dataset_type)
    #         result = compare_image_lists(images_prediction, images_gt, alphas_gt)
    #         formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
    #         results = results + formatted_result
    #     print(exp, results)

    # for exp in ['syn_drums_b_t1', 'syn_drums_b_t2', 'syn_drums_b_t3', 'syn_drums_b_t4']:
    # # for exp in ['syn_drums_b_t1']:
    #     dataset_type = 'syn_intrinsic'
    #     path_prediction = f'/ghome/yyang/PycharmProjects/neuralangelo/logs_old/{exp}/output_image'
    #     path_gt = '/ghome/yyang/dataset/Relighting_intrinsic/drums_intrinsic/test'
    #     results = ''
    #     for component in ['Ref', 'Sha', 'Img']:
    #         gamma_corr = component == 'Sha'
    #         images_prediction = get_output_ours(path_prediction, component, dataset_type=dataset_type)
    #         images_gt, alphas_gt = get_GT(path_gt, component, dataset_type=dataset_type)
    #         result = compare_image_lists(images_prediction, images_gt, alphas_gt, gamma_corr=gamma_corr)
    #         formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
    #         results = results + formatted_result
    #     print(exp, results)

    # for exp in ['hotdog', 'lego', 'drums', 'FurBall']:
    #     dataset_type = 'syn_intrinsic'
    #     path_prediction = f'/ghome/yyang/PycharmProjects/NRHints/outputs/baseline/{exp}_intrinsic/test_views/step_1000000'
    #     path_gt = f'/ghome/yyang/dataset/Relighting_intrinsic/{exp}_intrinsic/test'
    #     results = ''
    #     for component in ['Img']:
    #         gamma_corr = component == 'Sha'
    #         images_prediction = get_output_NRHints(path_prediction)
    #         images_gt, alphas_gt = get_GT(path_gt, component, dataset_type=dataset_type)
    #         result = compare_image_lists(images_prediction, images_gt, alphas_gt, gamma_corr=gamma_corr)
    #         formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
    #         results = results + formatted_result
    #     print(exp, results)

    # for exp in ['hotdog', 'lego', 'drums', 'FurBall']:
    #     dataset_type = 'syn_intrinsic'
    #     # path_prediction = f'/ghome/yyang/PycharmProjects/PIE-Net/test_output/{exp}_intrinsic/'   # PIE-Net
    #     # path_prediction = f'/ghome/yyang/PycharmProjects/Intrinsic/test_output/{exp}_intrinsic/'   # ordinal
    #     for method in ['PIE-Net', 'ordinal']:
    #         if method == 'PIE-Net':
    #             path_prediction = f'/ghome/yyang/PycharmProjects/PIE-Net/test_output/{exp}_intrinsic/'
    #         elif method == 'ordinal':
    #             path_prediction = f'/ghome/yyang/PycharmProjects/Intrinsic/test_output/{exp}_intrinsic/'
    #         else:
    #             raise NotImplementedError
    #         path_gt = f'/ghome/yyang/dataset/Relighting_intrinsic/{exp}_intrinsic/test'
    #         results = ''
    #         for component in ['Ref', 'Sha']:
    #             gamma_corr = component == 'Sha'
    #             images_prediction = get_output_pienet_ordinal(path_prediction, component)
    #             images_gt, alphas_gt = get_GT(path_gt, component, dataset_type=dataset_type)
    #             result = compare_image_lists(images_prediction, images_gt, alphas_gt, gamma_corr=gamma_corr,
    #                                          mask_pred=True)
    #             formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
    #             results = results + formatted_result
    #         print(exp, method, results)

    # for exp in ['hotdog', 'lego', 'drums', 'FurBall']:
    #     dataset_type = 'syn_intrinsic'
    #     path_prediction = f'/ghome/ramon/PycharmProjects/OpenIllumination_work/third_party/InvRender/relits/Mat-{exp}_intrinsic_NL1/light_origin'
    #     path_gt = f'/ghome/yyang/dataset/Relighting_intrinsic/{exp}_intrinsic_NL1/test'
    #     results = ''
    #     for component in ['Ref', 'Img']:
    #         gamma_corr = component == 'Sha'
    #         images_prediction = get_output_InvRender(path_prediction, component)
    #         images_gt, alphas_gt = get_GT(path_gt, component, dataset_type=dataset_type)
    #         result = compare_image_lists(images_prediction, images_gt, alphas_gt, gamma_corr=gamma_corr)
    #         formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
    #         results = results + formatted_result
    #     print(exp, results)

    # for light_condition in ['NL1', 'NL4']:
    #     for exp in ['hotdog', 'lego', 'drums', 'FurBall']:
    #         dataset_type = 'syn_intrinsic'
    #         pred_folder = 'singleillum' if light_condition == 'NL1' else 'multiillum'
    #         path_prediction = f'/ghome/ramon/PycharmProjects/OpenIllumination_work/third_party/TensoIR/logs/{pred_folder}_syn_{exp}/'
    #         path_gt = f'/ghome/yyang/dataset/Relighting_intrinsic/{exp}_intrinsic_{light_condition}/test'
    #         results = ''
    #         for component in ['Ref', 'Img']:
    #             gamma_corr = component == 'Sha'
    #             images_prediction = get_output_TensoIR(path_prediction, component)
    #             images_gt, alphas_gt = get_GT(path_gt, component, dataset_type=dataset_type)
    #             result = compare_image_lists(images_prediction, images_gt, alphas_gt, gamma_corr=gamma_corr)
    #             formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
    #             results = results + formatted_result
    #         print(light_condition, exp, results)

    exp_list = ['Pikachu', 'Pixiu', 'FurScene', 'Fish']
    for exp in exp_list:
        path_prediction = f'/ghome/yyang/PycharmProjects/neuralangelo/all_results/objects/Ours/{exp}/'
        path_gt = f'/ghome/yyang/dataset/NRHints/{exp}/test'
        num_imgs = len(os.listdir(path_gt))
        print('Num_imgs:', num_imgs)
        images_pred = [os.path.join(path_prediction, f'{str(i)}_rgb_map.png') for i in range(num_imgs)]
        images_gt = [os.path.join(path_gt, f'r_{str(i)}.png') for i in range(num_imgs)]

        result = compare_image_lists(images_pred, images_gt, [None]*len(images_gt), gamma_corr=False)
        formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
        print(formatted_result)

    # exp_pred_list = ['syn_hotdog_b_1', 'syn_lego_b_1', 'syn_drums_b_1', 'syn_FurBall_b_1']
    # exp_gt_list = ['hotdog_intrinsic', 'lego_intrinsic', 'drums_intrinsic', 'FurBall_intrinsic']
    # for exp_pred, exp_gt in zip(exp_pred_list, exp_gt_list):
    #     dataset_type = 'syn_intrinsic'
    #     path_prediction = f'/ghome/yyang/PycharmProjects/neuralangelo/logs/{exp_pred}/output_image'
    #     path_gt = f'/ghome/yyang/dataset/Relighting_intrinsic/{exp_gt}/test'
    #     results = ''
    #     for component in ['Ref', 'Sha', 'Img']:
    #         gamma_corr = component == 'Sha'
    #         images_prediction = get_output_ours(path_prediction, component, dataset_type=dataset_type)
    #         images_gt, alphas_gt = get_GT(path_gt, component, dataset_type=dataset_type)
    #         result = compare_image_lists(images_prediction, images_gt, alphas_gt, gamma_corr=gamma_corr)
    #         formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
    #         results = results + formatted_result
    #     print(exp_pred, results)

    # for path_prediction in ['/ghome/yyang/PycharmProjects/NRHints/outputs/baseline/Pikachu/test_views/step_1000000']:
    #     path_gt = '/ghome/yyang/dataset/NRHints/Pikachu/test'
    #     images_pred = [os.path.join(path_prediction, f'rgb_{str(i).zfill(3)}.png') for i in range(243)]
    #     images_gt = [os.path.join(path_gt, f'r_{str(i)}.png') for i in range(243)]
    #
    #     result = compare_image_lists(images_pred, images_gt, [None]*len(images_gt), gamma_corr=False)
    #     formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
    #     print(formatted_result)

    # exp_pred_list = ['hotdog', 'lego', 'drums', 'FurBall']
    # exp_gt_list = ['hotdog', 'lego', 'drums', 'FurBall']
    # for exp_pred, exp_gt in zip(exp_pred_list, exp_gt_list):
    #     dataset_type = 'syn_intrinsic'
    #     path_prediction = f'/ghome/yyang/PycharmProjects/neuralangelo/all_results/syn/Single_IntrinsicNeRF/{exp_pred}/'
    #     path_gt = f'/ghome/yyang/dataset/Relighting_intrinsic/{exp_gt}_intrinsic_NL1/test/'
    #     results = ''
    #     for component in ['Ref', 'Sha', 'Img']:
    #         gamma_corr = component == 'Sha'
    #         images_prediction = get_output_IntrinsicNeRF(path_prediction, component, dataset_type=dataset_type)
    #         images_gt, alphas_gt = get_GT(path_gt, component, dataset_type=dataset_type)
    #         result = compare_image_lists(images_prediction, images_gt, alphas_gt, gamma_corr=gamma_corr)
    #         formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
    #         results = results + formatted_result
    #     print(exp_pred, results)

    # exp_pred_list = ['savannah', 'apple', 'garden', 'cube']
    # exp_gt_list = ['savannah', 'apple', 'garden', 'cube']
    # for exp_pred, exp_gt in zip(exp_pred_list, exp_gt_list):
    #     path_prediction = f'/ghome/yyang/PycharmProjects/neuralangelo/all_results/rene/All_Ours/{exp_pred}/'
    #
    #     data_root = f'/ghome/yyang/dataset/rene_dataset/{exp_gt}/'
    #     meta_fname = f'/ghome/yyang/PycharmProjects/neuralangelo/dataset_rene/{exp_gt}/test_custom_transforms.json'
    #     gt_images_list = get_GT_rene(data_root, meta_fname)
    #
    #     results = ''
    #     images_prediction = get_output_ours(path_prediction, 'Img',dataset_type='rene')
    #     images_gt = gt_images_list
    #     alphas_gt = [None] * len(images_gt)
    #     result = compare_image_lists(images_prediction, images_gt, alphas_gt)
    #     formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
    #     results = results + formatted_result
    #     print(exp_pred, results)

    # exp_pred_list = ['savannah', 'apple', 'garden', 'cube']
    # exp_gt_list = ['savannah', 'apple', 'garden', 'cube']
    # for exp_pred, exp_gt in zip(exp_pred_list, exp_gt_list):
    #     path_prediction = f'/ghome/yyang/PycharmProjects/neuralangelo/all_results/rene/4lights_Ours/{exp_pred}/'
    #
    #     data_root = f'/ghome/yyang/dataset/rene_dataset/{exp_gt}/'
    #     meta_fname = f'/ghome/yyang/PycharmProjects/neuralangelo/dataset_rene/{exp_gt}/test_custom_transforms.json'
    #     gt_images_list = get_GT_rene(data_root, meta_fname, light_index=[24, 29, 33, 38])
    #
    #     results = ''
    #     images_prediction = get_output_ours(path_prediction, 'Img',dataset_type='rene')
    #     images_gt = gt_images_list
    #     alphas_gt = [None] * len(images_gt)
    #     result = compare_image_lists(images_prediction, images_gt, alphas_gt)
    #     formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
    #     results = results + formatted_result
    #     print(exp_pred, results)

    # exp_pred_list = ['savannah', 'apple', 'garden', 'cube']
    # exp_gt_list = ['savannah', 'apple', 'garden', 'cube']
    # for exp_pred, exp_gt in zip(exp_pred_list, exp_gt_list):
    #     path_prediction = f'/ghome/yyang/PycharmProjects/neuralangelo/all_results/rene/Single_Ours/{exp_pred}/'
    #
    #     data_root = f'/ghome/yyang/dataset/rene_dataset/{exp_gt}/'
    #     meta_fname = f'/ghome/yyang/PycharmProjects/neuralangelo/dataset_rene/{exp_gt}/test_custom_transforms.json'
    #     gt_images_list = get_GT_rene(data_root, meta_fname, light_index=[0])
    #     # print('Number of images:', len(gt_images_list))
    #
    #     results = ''
    #     images_prediction = get_output_ours(path_prediction, 'Img',dataset_type='rene')
    #     images_gt = gt_images_list
    #     alphas_gt = [None] * len(images_gt)
    #     result = compare_image_lists(images_prediction, images_gt, alphas_gt)
    #     formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
    #     results = results + formatted_result
    #     print(exp_pred, results)

    # exp_pred_list = ['savannah', 'apple', 'garden', 'cube']
    # exp_gt_list = ['savannah', 'apple', 'garden', 'cube']
    # for exp_pred, exp_gt in zip(exp_pred_list, exp_gt_list):
    #     path_prediction = f'/ghome/yyang/PycharmProjects/neuralangelo/all_results/rene/Single_TensoIR/{exp_pred}/'
    #
    #     data_root = f'/ghome/yyang/dataset/rene_dataset/{exp_gt}/'
    #     meta_fname = f'/ghome/yyang/PycharmProjects/neuralangelo/dataset_rene/{exp_gt}/test_custom_transforms.json'
    #     gt_images_list = get_GT_rene(data_root, meta_fname, light_index=[0])
    #     # print('Number of images:', len(gt_images_list))
    #
    #     results = ''
    #     path_prediction_image = os.path.join(path_prediction, f'nvs_with_radiance_field')
    #     images_prediction = [x for x in os.listdir(path_prediction_image) if 'rgb_prediction.png' in x]
    #     images_prediction.sort()
    #     images_prediction = [os.path.join(path_prediction_image, x) for x in images_prediction]
    #
    #     images_gt = gt_images_list
    #     alphas_gt = [None] * len(images_gt)
    #     result = compare_image_lists(images_prediction, images_gt, alphas_gt)
    #     formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
    #     results = results + formatted_result
    #     print(exp_pred, results)

    # exp_pred_list = ['savannah', 'apple', 'garden', 'cube']
    # exp_gt_list = ['savannah', 'apple', 'garden', 'cube']
    # for exp_pred, exp_gt in zip(exp_pred_list, exp_gt_list):
    #     path_prediction = f'/ghome/yyang/PycharmProjects/neuralangelo/all_results/rene/3lights_TensoIR/{exp_pred}/'
    #
    #     data_root = f'/ghome/yyang/dataset/rene_dataset/{exp_gt}/'
    #     meta_fname = f'/ghome/yyang/PycharmProjects/neuralangelo/dataset_rene/{exp_gt}/test_custom_transforms.json'
    #     gt_images_list = get_GT_rene(data_root, meta_fname, light_index=[0, 9, 11])
    #     # print('Number of images:', len(gt_images_list))
    #
    #     results = ''
    #     path_prediction_image = os.path.join(path_prediction, f'nvs_with_radiance_field')
    #     images_prediction = [x for x in os.listdir(path_prediction_image) if 'rgb_prediction.png' in x]
    #     images_prediction.sort()
    #     images_prediction = [os.path.join(path_prediction_image, x) for x in images_prediction]
    #
    #     images_gt = gt_images_list
    #     alphas_gt = [None] * len(images_gt)
    #     result = compare_image_lists(images_prediction, images_gt, alphas_gt)
    #     formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
    #     results = results + formatted_result
    #     print(exp_pred, results)

    # exp_pred_list = ['savannah', 'apple', 'garden', 'cube']
    # exp_gt_list = ['savannah', 'apple', 'garden', 'cube']
    # for exp_pred, exp_gt in zip(exp_pred_list, exp_gt_list):
    #     path_prediction = f'/ghome/yyang/PycharmProjects/neuralangelo/all_results/rene/Single_IntrinsicNeRF/{exp_pred}/'
    #
    #     data_root = f'/ghome/yyang/dataset/rene_dataset/{exp_gt}/'
    #     meta_fname = f'/ghome/yyang/PycharmProjects/neuralangelo/dataset_rene/{exp_gt}/test_custom_transforms.json'
    #     gt_images_list = get_GT_rene(data_root, meta_fname, light_index=[0])
    #     # print('Number of images:', len(gt_images_list))
    #
    #     results = ''
    #     images_prediction = [x for x in os.listdir(path_prediction) if 'rgb_' in x]
    #     images_prediction.sort()
    #     images_prediction = [os.path.join(path_prediction, x) for x in images_prediction]
    #
    #     images_gt = gt_images_list
    #     alphas_gt = [None] * len(images_gt)
    #     result = compare_image_lists(images_prediction, images_gt, alphas_gt)
    #     formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
    #     results = results + formatted_result
    #     print(exp_pred, results)

    # exp_pred_list = ['hotdog', 'lego', 'drums', 'FurBall']
    # exp_gt_list = ['hotdog', 'lego', 'drums', 'FurBall']
    # for exp_pred, exp_gt in zip(exp_pred_list, exp_gt_list):
    #     dataset_type = 'syn_intrinsic'
    #     path_prediction = f'/ghome/yyang/PycharmProjects/neuralangelo/all_results/syn/Single_Ours/{exp_pred}/'
    #     path_gt = f'/ghome/yyang/dataset/Relighting_intrinsic/{exp_gt}_intrinsic_NL1/test'
    #     results = ''
    #     for component in ['Ref', 'Sha', 'Img']:
    #         gamma_corr = component == 'Sha'
    #         images_prediction = get_output_ours(path_prediction, component, dataset_type=dataset_type)
    #         images_gt, alphas_gt = get_GT(path_gt, component, dataset_type=dataset_type)
    #         result = compare_image_lists(images_prediction, images_gt, alphas_gt, gamma_corr=gamma_corr)
    #         formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
    #         results = results + formatted_result
    #     print(exp_pred, results)

    # exp_pred_list = ['hotdog', 'lego', 'drums', 'FurBall']
    # exp_gt_list = ['hotdog', 'lego', 'drums', 'FurBall']
    # for exp_pred, exp_gt in zip(exp_pred_list, exp_gt_list):
    #     dataset_type = 'syn_intrinsic'
    #     path_prediction = f'/ghome/yyang/PycharmProjects/neuralangelo/all_results/syn/4lights_Ours/{exp_pred}/'
    #     path_gt = f'/ghome/yyang/dataset/Relighting_intrinsic/{exp_gt}_intrinsic_NL4/test'
    #     results = ''
    #     for component in ['Ref', 'Sha', 'Img']:
    #         gamma_corr = component == 'Sha'
    #         images_prediction = get_output_ours(path_prediction, component, dataset_type=dataset_type)
    #         images_gt, alphas_gt = get_GT(path_gt, component, dataset_type=dataset_type)
    #         result = compare_image_lists(images_prediction, images_gt, alphas_gt, gamma_corr=gamma_corr)
    #         formatted_result = f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f} {result[3]:.4f} "
    #         results = results + formatted_result
    #     print(exp_pred, results)

    pass









